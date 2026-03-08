"""
Pretrain Ego Encoder via VAE Latent Regression
===============================================
Trains the ego encoder to predict the VAE's motion latent z from the ego
trajectory. This gives the ego encoder a meaningful embedding space *before*
diffusion training begins.

  ego (B,T,2) -> EgoEncoder -> pool -> z_pred (B,1,256)
                                          |
                                        MSE loss + cosine loss
                                          |
  motion (B,T,263) -> frozen VAE.encode -> z_gt  (1,B,256)

Usage:
  # Full dataset pretraining (recommended):
  python pretrain_ego_encoder.py \
      --cfg ./configs/config_ego_motion.yaml \
      --epochs 200 --batch_size 64 --lr 1e-4 \
      --output_dir ./experiments/ego_encoder_pretrain

  # Quick overfit test (single sample):
  python pretrain_ego_encoder.py \
      --cfg ./configs/config_ego_motion.yaml \
      --overfit --epochs 2000 \
      --output_dir ./experiments/ego_encoder_pretrain_overfit

  # Then use the saved weights in diffusion training:
  #   Set TRAIN.PRETRAINED_EGO in config to the checkpoint path
  python train.py --cfg ./configs/config_ego_motion.yaml
"""

import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from omegaconf import OmegaConf

from mld.config import get_module_config, instantiate_from_config
from mld.data.get_data import get_datasets

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def build_cfg(args):
    """Build config exactly like train.py does, with CLI overrides."""
    cfg_base = OmegaConf.load("./configs/base.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg_assets = OmegaConf.load(args.cfg_assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

    # CLI overrides for dataset behaviour
    if args.overfit is not None:
        cfg.OVERFIT = args.overfit
    else:
        # Default: full-dataset (False) unless config says otherwise
        # This prevents accidentally training on 1 sample with an overfit config
        cfg.OVERFIT = False

    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size

    if args.data_roots is not None:
        cfg.DATASET.EGOMOTION.ROOT = args.data_roots

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

    # Build our own dataloaders with drop_last=False so small datasets
    # (e.g. AVA with 27 samples) don't lose all data when batch_size > n_samples.
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from mld.data.EgoMotion import ego_motion_collate
    effective_bs = min(cfg.TRAIN.BATCH_SIZE, len(datamodule.train_dataset))
    sampler = None
    shuffle = True
    if datamodule.train_dataset.sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=datamodule.train_dataset.sample_weights,
            num_samples=len(datamodule.train_dataset),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(
        datamodule.train_dataset,
        batch_size=effective_bs,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        collate_fn=ego_motion_collate,
        drop_last=False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datamodule.val_dataset,
        batch_size=min(cfg.TRAIN.BATCH_SIZE, len(datamodule.val_dataset)),
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        collate_fn=ego_motion_collate,
        drop_last=False,
        pin_memory=True,
    )
    n_train = len(datamodule.train_dataset)
    n_val = len(datamodule.val_dataset)
    print(f"Train samples: {n_train} ({len(train_loader)} batches, bs={effective_bs}), "
          f"Val samples: {n_val} ({len(val_loader)} batches)")
    print(f"OVERFIT={cfg.OVERFIT}")

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

    # Learning rate schedule: linear warmup + cosine decay
    warmup_epochs = min(args.warmup_epochs, args.epochs // 10)
    if warmup_epochs > 0:
        warmup_scheduler = LambdaLR(
            optimizer, lr_lambda=lambda ep: min(1.0, (ep + 1) / warmup_epochs)
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_epochs, eta_min=lr * 0.01
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=lr * 0.01)

    # ── Output dir ────────────────────────────────────────────────────────
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # Save config for reference
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))
    print(f"Outputs → {output_dir}")

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = None
    if HAS_TB and not args.no_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))
        print(f"TensorBoard → {os.path.join(output_dir, 'tb_logs')}")

    # ── Use VAE mean (deterministic target) ───────────────────────────────
    use_mean = args.use_mean
    print(f"Target: {'VAE mean (deterministic)' if use_mean else 'VAE sample (stochastic)'}")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_interval = max(1, args.epochs // 100)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        ego_encoder.train()
        t0 = time.time()

        epoch_mse = 0.0
        epoch_cos = 0.0
        epoch_grad_norm = 0.0
        n_batches = 0

        loader = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                      leave=False) if HAS_TQDM else train_loader

        for batch in loader:
            ego = batch["ego"].to(device)           # (B, T, 2)
            motion = batch["motion"].to(device)     # (B, T, 263)
            lengths = batch["length"]               # list[int]

            # ── Teacher: VAE encodes motion → latent target ───────────
            with torch.no_grad():
                z_gt, dist_gt = vae.encode(motion, lengths)
                # z_gt shape: (n_token, B, z_dim) = (1, B, 256)
                if use_mean:
                    z_target = dist_gt.loc       # (n_token, B, z_dim) — deterministic
                else:
                    z_target = z_gt              # (n_token, B, z_dim) — sampled
                # Average over latent tokens (handles multi-token VAEs, e.g. latent_dim=[4,256])
                z_target = z_target.mean(dim=0)  # (B, z_dim)

            # ── Student: ego encoder → pooled embedding (includes projection) ─
            ego_emb = ego_encoder(ego)           # (B, 1, z_dim) for pooled
            z_pred = ego_emb.squeeze(1)          # (B, z_dim)

            # ── Loss ──────────────────────────────────────────────────
            mse_loss = F.mse_loss(z_pred, z_target)
            cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean()
            # Combined loss: MSE + (1 - cos_sim) to push directions right too
            loss = mse_loss + args.cos_weight * (1.0 - cos_sim)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_mse += mse_loss.item()
            epoch_cos += cos_sim.item()
            epoch_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            n_batches += 1
            global_step += 1

            if HAS_TQDM and isinstance(loader, tqdm):
                loader.set_postfix(mse=f"{mse_loss.item():.4f}", cos=f"{cos_sim.item():.3f}")

        scheduler.step()
        avg_mse = epoch_mse / max(n_batches, 1)
        avg_cos = epoch_cos / max(n_batches, 1)
        avg_grad = epoch_grad_norm / max(n_batches, 1)
        epoch_time = time.time() - t0

        # ── TensorBoard logging ──────────────────────────────────────
        if writer:
            writer.add_scalar("train/mse", avg_mse, epoch)
            writer.add_scalar("train/cosine_sim", avg_cos, epoch)
            writer.add_scalar("train/grad_norm", avg_grad, epoch)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # ── Validation ────────────────────────────────────────────────
        if epoch % log_interval == 0 or epoch == 1 or epoch == args.epochs:
            val_mse, val_cos = evaluate(ego_encoder, vae, val_loader,
                                        device, use_mean)
            improved = val_mse < best_val_loss
            if improved:
                best_val_loss = val_mse
                save_checkpoint(ego_encoder, optimizer, epoch,
                                val_mse, os.path.join(output_dir, "checkpoints", "best.pt"))

            if writer:
                writer.add_scalar("val/mse", val_mse, epoch)
                writer.add_scalar("val/cosine_sim", val_cos, epoch)

            print(
                f"Epoch {epoch:5d}/{args.epochs} | "
                f"train_mse={avg_mse:.6f} cos={avg_cos:.3f} | "
                f"val_mse={val_mse:.6f} cos={val_cos:.3f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"gnorm={avg_grad:.2f} | "
                f"{epoch_time:.1f}s"
                f"{' *' if improved else ''}"
            )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(ego_encoder, optimizer, epoch,
                            avg_mse,
                            os.path.join(output_dir, "checkpoints", f"epoch={epoch}.pt"))

    # Final save
    save_checkpoint(ego_encoder, optimizer, args.epochs,
                    avg_mse, os.path.join(output_dir, "checkpoints", "last.pt"))

    if writer:
        writer.close()

    print(f"\nDone. Best val MSE: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {os.path.join(output_dir, 'checkpoints')}")
    print(f"\nTo use in diffusion training, set in config:")
    print(f"  TRAIN.PRETRAINED_EGO: '{os.path.join(output_dir, 'checkpoints', 'best.pt')}'")    


@torch.no_grad()
def evaluate(ego_encoder, vae, val_loader, device, use_mean):
    """Returns (mse, cosine_similarity)."""
    ego_encoder.eval()

    total_mse = 0.0
    total_cos = 0.0
    n_batches = 0

    for batch in val_loader:
        ego = batch["ego"].to(device)
        motion = batch["motion"].to(device)
        lengths = batch["length"]

        z_gt, dist_gt = vae.encode(motion, lengths)
        z_target = (dist_gt.loc if use_mean else z_gt).mean(dim=0)

        z_pred = ego_encoder(ego).squeeze(1)  # includes built-in projection

        total_mse += F.mse_loss(z_pred, z_target).item()
        total_cos += F.cosine_similarity(z_pred, z_target, dim=-1).mean().item()
        n_batches += 1

    return total_mse / max(n_batches, 1), total_cos / max(n_batches, 1)


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
        "--epochs", type=int, default=200,
        help="Number of pretraining epochs (200 is usually enough for full dataset)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: use TRAIN.OPTIM.LR from config)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (default: use TRAIN.BATCH_SIZE from config)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./experiments/ego_encoder_pretrain",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--save_every", type=int, default=50,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--use_mean", action="store_true", default=True,
        help="Use VAE distribution mean (mu) as target (default: True). "
             "Use --no_use_mean to disable.",
    )
    parser.add_argument(
        "--no_use_mean", dest="use_mean", action="store_false",
        help="Use sampled z as target instead of mean.",
    )
    parser.add_argument(
        "--overfit", action="store_true", default=None,
        help="Override config to train on single sample (for debugging).",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10,
        help="Number of linear warmup epochs (0 to disable).",
    )
    parser.add_argument(
        "--cos_weight", type=float, default=0.1,
        help="Weight for cosine similarity loss component (0 = MSE only).",
    )
    parser.add_argument(
        "--no_tensorboard", action="store_true",
        help="Disable TensorBoard logging.",
    )
    parser.add_argument(
        "--data_roots", nargs="+", type=str, default=None,
        help="Override data roots, e.g.: --data_roots /path/to/ava /path/to/waymo",
    )

    args = parser.parse_args()
    run_pretraining(args)


if __name__ == "__main__":
    main()
