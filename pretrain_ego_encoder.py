"""
Pretrain Ego Encoder via VAE Latent Regression
===============================================
Trains the ego encoder to predict the VAE's motion latent z from the ego
trajectory. This gives the ego encoder a meaningful embedding space *before*
diffusion training begins.

  ego (B,T,2) -> EgoEncoder -> pool -> z_pred (B,1,256)
                                          |
                                        MSE loss
                                          |
  motion (B,T,263) -> frozen VAE.encode -> z_gt  (1,B,256)

Usage:
  python pretrain_ego_encoder.py \
      --cfg ./configs/config_ego_motion.yaml \
      --epochs 2000 \
      --lr 1e-4 \
      --output_dir ./experiments/ego_encoder_pretrain

  # Then use the saved weights in diffusion training:
  python train.py --cfg ./configs/config_ego_motion.yaml
  # (set TRAIN.PRETRAINED_EGO in config, or load manually)
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf

from mld.config import get_module_config, instantiate_from_config
from mld.data.get_data import get_datasets


def build_cfg(args):
    """Build config exactly like train.py does."""
    cfg_base = OmegaConf.load("./configs/base.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg_assets = OmegaConf.load(args.cfg_assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)
    return cfg


def load_vae(cfg, device):
    """Build VAE and load pretrained weights (frozen)."""
    vae = instantiate_from_config(cfg.model.motion_vae)

    vae_ckpt_path = cfg.TRAIN.PRETRAINED_VAE
    if not vae_ckpt_path:
        raise ValueError(
            "TRAIN.PRETRAINED_VAE must be set in the config to pretrain "
            "the ego encoder (we need a frozen VAE as teacher)."
        )

    print(f"Loading VAE from {vae_ckpt_path}")
    state_dict = torch.load(vae_ckpt_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("vae."):
            vae_dict[k.replace("vae.", "")] = v
    vae.load_state_dict(vae_dict, strict=True)

    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    print(f"VAE loaded and frozen ({sum(p.numel() for p in vae.parameters()):.1e} params)")
    return vae


def build_ego_encoder(cfg, device):
    """Instantiate the ego encoder from the same config the diffusion model uses."""
    ego_encoder = instantiate_from_config(cfg.model.ego_encoder)
    ego_encoder.to(device)
    n_params = sum(p.numel() for p in ego_encoder.parameters() if p.requires_grad)
    print(f"Ego encoder: {n_params:,} trainable parameters")
    return ego_encoder


def run_pretraining(args):
    # ── Config ────────────────────────────────────────────────────────────
    cfg = build_cfg(args)

    device = torch.device(
        f"cuda:{cfg.DEVICE[0]}" if cfg.ACCELERATOR == "gpu" and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    torch.manual_seed(cfg.SEED_VALUE)
    np.random.seed(cfg.SEED_VALUE)

    # ── Dataset ───────────────────────────────────────────────────────────
    datasets = get_datasets(cfg, phase="train")
    datamodule = datasets[0]
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Models ────────────────────────────────────────────────────────────
    vae = load_vae(cfg, device)
    ego_encoder = build_ego_encoder(cfg, device)

    latent_dim = cfg.model.latent_dim  # [1, 256]
    z_dim = latent_dim[-1]             # 256

    # The projection head is now BUILT INTO the EgoEncoderPooled class,
    # so no external projection is needed. This ensures the projection
    # weights are part of ego_encoder.state_dict() and get saved/loaded
    # correctly into the diffusion model.
    n_params = sum(p.numel() for p in ego_encoder.parameters() if p.requires_grad)
    print(f"Ego encoder (with built-in projection): {n_params:,} trainable parameters")

    # ── Optimiser ─────────────────────────────────────────────────────────
    lr = args.lr if args.lr is not None else cfg.TRAIN.OPTIM.LR
    params = list(ego_encoder.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=lr * 0.01)

    # ── Output dir ────────────────────────────────────────────────────────
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # Save config for reference
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))
    print(f"Outputs → {output_dir}")

    # ── Use VAE mean (deterministic target) ───────────────────────────────
    use_mean = args.use_mean  # Use mu instead of sampled z as target

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_interval = max(1, args.epochs // 100)

    for epoch in range(1, args.epochs + 1):
        ego_encoder.train()

        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            ego = batch["ego"].to(device)           # (B, T, 2)
            motion = batch["motion"].to(device)     # (B, T, 263)
            lengths = batch["length"]               # list[int]

            # ── Teacher: VAE encodes motion → latent target ───────────
            with torch.no_grad():
                z_gt, dist_gt = vae.encode(motion, lengths)
                # z_gt shape: (n_token, B, z_dim) = (1, B, 256)
                if use_mean:
                    z_target = dist_gt.loc       # (1, B, z_dim) — deterministic
                else:
                    z_target = z_gt              # (1, B, z_dim) — sampled
                z_target = z_target.squeeze(0)   # (B, z_dim)

            # ── Student: ego encoder → pooled embedding (includes projection) ─
            ego_emb = ego_encoder(ego)           # (B, 1, z_dim) for pooled
            z_pred = ego_emb.squeeze(1)          # (B, z_dim)

            # ── Loss ──────────────────────────────────────────────────
            loss = F.mse_loss(z_pred, z_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────
        if epoch % log_interval == 0 or epoch == 1:
            val_loss = evaluate(ego_encoder, vae, val_loader,
                                device, use_mean)
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                save_checkpoint(ego_encoder, optimizer, epoch,
                                val_loss, os.path.join(output_dir, "checkpoints", "best.pt"))
            print(
                f"Epoch {epoch:5d}/{args.epochs} | "
                f"train_loss={avg_train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
                f"{' *' if improved else ''}"
            )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(ego_encoder, optimizer, epoch,
                            avg_train_loss,
                            os.path.join(output_dir, "checkpoints", f"epoch={epoch}.pt"))

    # Final save
    save_checkpoint(ego_encoder, optimizer, args.epochs,
                    avg_train_loss, os.path.join(output_dir, "checkpoints", "last.pt"))
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {os.path.join(output_dir, 'checkpoints')}")
    print(f"\nTo use in diffusion training, load with:")
    print(f"  state = torch.load('{os.path.join(output_dir, 'checkpoints', 'best.pt')}')")
    print(f"  model.ego_encoder.load_state_dict(state['ego_encoder'])")


@torch.no_grad()
def evaluate(ego_encoder, vae, val_loader, device, use_mean):
    ego_encoder.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        ego = batch["ego"].to(device)
        motion = batch["motion"].to(device)
        lengths = batch["length"]

        z_gt, dist_gt = vae.encode(motion, lengths)
        z_target = (dist_gt.loc if use_mean else z_gt).squeeze(0)

        ego_emb = ego_encoder(ego).squeeze(1)  # includes built-in projection
        z_pred = ego_emb

        total_loss += F.mse_loss(z_pred, z_target).item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_checkpoint(ego_encoder, optimizer, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "ego_encoder": ego_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain ego encoder via VAE latent regression"
    )
    parser.add_argument(
        "--cfg", type=str, default="./configs/config_ego_motion.yaml",
        help="Same config file used for diffusion training",
    )
    parser.add_argument(
        "--cfg_assets", type=str, default="./configs/assets.yaml",
        help="Asset config (same as train.py)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2000,
        help="Number of pretraining epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: use TRAIN.OPTIM.LR from config)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./experiments/ego_encoder_pretrain",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--save_every", type=int, default=500,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--use_mean", action="store_true",
        help="Use VAE distribution mean (mu) as target instead of sampled z. "
             "Recommended for more stable targets.",
    )

    args = parser.parse_args()
    run_pretraining(args)


if __name__ == "__main__":
    main()
