"""
Baseline 4: VAE Interpolation

Generates motions by interpolating between two randomly chosen training
samples in the VAE latent space and decoding.  This tests whether the
learned latent manifold is smooth and whether generated motions lie on it,
independent of any ego conditioning.

Interpolation
-------------
For each test sample of length L, two distinct training motions are chosen at
random, encoded to get z1 and z2 (shape: ntoken×256), linearly interpolated
with a random α ∈ (0, 1), then decoded with length L.  Linear interpolation
is used (SLERP is not materially better for near-Gaussian latent spaces).

Usage:
    python baselines/eval_vae_interp.py \
        --cfg configs/config_ego_motion_baseline_vae_interp.yaml \
        --cfg_assets configs/assets.yaml
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mld.config import parse_args
from mld.data.EgoMotion import EgoMotionDataModule, ego_motion_collate
from mld.models.architectures.mld_vae import MldVae
from mld.config import instantiate_from_config
from mld.utils.logger import create_logger
from baselines.baseline_utils import (
    compute_metrics,
    compute_motion_embeddings,
    load_t2m_evaluator,
    print_metrics,
    save_metrics,
)

BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------

def load_vae(cfg, device: str) -> MldVae:
    """Instantiate and load pretrained VAE weights (frozen)."""
    vae: MldVae = instantiate_from_config(cfg.model.motion_vae).to(device).eval()

    ckpt = torch.load(cfg.TRAIN.PRETRAINED_VAE, map_location="cpu", weights_only=False)
    # Strip the 'vae.' prefix that PL adds when saving from an MLD model
    sd = ckpt.get("state_dict", ckpt)
    vae_sd = {k[len("vae."):]: v for k, v in sd.items() if k.startswith("vae.")}
    if not vae_sd:  # checkpoint was saved as standalone VAE
        vae_sd = sd
    vae.load_state_dict(vae_sd, strict=True)

    for p in vae.parameters():
        p.requires_grad = False

    return vae


# ---------------------------------------------------------------------------
# Helper: encode a batch of (padded) motions with the VAE
# ---------------------------------------------------------------------------

def vae_encode_batch(vae, motions: torch.Tensor, lengths: list, device: str):
    """
    Returns: z (ntoken, B, latent_dim) — posterior *mean* (no sampling).
    """
    motions = motions.to(device).float()
    with torch.no_grad():
        _, dist = vae.encode(motions, lengths)
    return dist.loc   # use mean to get deterministic encoding


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args(phase="test")
    cfg.FOLDER = cfg.TEST.FOLDER

    logger = create_logger(cfg, phase="test")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Dataset -----------------------------------------------------------
    datamodule = EgoMotionDataModule(cfg, logger=logger)
    datamodule.setup(stage="fit")   # train split
    datamodule.setup(stage="test")  # test split

    train_loader = DataLoader(
        datamodule.train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=ego_motion_collate,
    )
    test_loader = DataLoader(
        datamodule.test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=ego_motion_collate,
    )

    # ---- Collect training motions -----------------------------------------
    logger.info("Collecting training motions …")
    train_motions, train_lengths = [], []
    for batch in tqdm(train_loader, desc="Train"):
        train_motions.append(batch["motion"])   # (B, T, 263)
        train_lengths.extend(batch["length"])
    train_motions = torch.cat(train_motions, dim=0)   # (N_train, T, 263)
    logger.info(f"  {len(train_lengths)} training samples")

    # ---- Collect test ground-truth ----------------------------------------
    logger.info("Collecting test motions (GT) …")
    test_motions_gt, test_lengths = [], []
    for batch in tqdm(test_loader, desc="Test"):
        test_motions_gt.append(batch["motion"])
        test_lengths.extend(batch["length"])
    test_motions_gt = torch.cat(test_motions_gt, dim=0)
    logger.info(f"  {len(test_lengths)} test samples")

    # ---- Load VAE ----------------------------------------------------------
    logger.info(f"Loading VAE from {cfg.TRAIN.PRETRAINED_VAE} …")
    vae = load_vae(cfg, device)

    # ---- Pre-encode all training motions (in mini-batches) ----------------
    logger.info("Encoding training motions …")
    n_train = len(train_lengths)
    all_z: list = []
    for start in tqdm(range(0, n_train, BATCH_SIZE), desc="Encoding train"):
        end = min(start + BATCH_SIZE, n_train)
        z = vae_encode_batch(
            vae, train_motions[start:end], train_lengths[start:end], device
        )                                  # (ntoken, B, latent_dim)
        all_z.append(z.permute(1, 0, 2))  # (B, ntoken, latent_dim)
    all_z = torch.cat(all_z, dim=0).cpu()  # (N_train, ntoken, latent_dim)
    logger.info(f"  Latent shape: {all_z.shape}")

    # ---- Interpolate and decode -------------------------------------------
    logger.info("Generating interpolated motions …")
    rng = np.random.default_rng(42)
    interp_motions: list = []

    n_test = len(test_lengths)
    for start in tqdm(range(0, n_test, BATCH_SIZE), desc="Interpolating"):
        end = min(start + BATCH_SIZE, n_test)
        B = end - start

        # Pick two distinct training samples for each test sample
        idx1 = rng.integers(0, n_train, size=B)
        idx2 = rng.integers(0, n_train - 1, size=B)
        idx2[idx2 >= idx1] += 1   # ensure idx1 != idx2

        z1 = all_z[idx1].to(device)   # (B, ntoken, latent_dim)
        z2 = all_z[idx2].to(device)

        alpha = torch.tensor(
            rng.uniform(0.1, 0.9, size=(B, 1, 1)), dtype=torch.float32, device=device
        )
        z_interp = (1.0 - alpha) * z1 + alpha * z2   # linear interpolation
        z_interp = z_interp.permute(1, 0, 2)          # (ntoken, B, latent_dim)

        batch_lengths = test_lengths[start:end]
        with torch.no_grad():
            decoded = vae.decode(z_interp, batch_lengths)   # (B, T, 263)
        interp_motions.append(decoded.cpu())

    interp_motions = torch.cat(interp_motions, dim=0)   # (N_test, T, 263)

    # ---- T2M embeddings ----------------------------------------------------
    logger.info("Loading t2m evaluator …")
    move_enc, mot_enc = load_t2m_evaluator(cfg, device)

    logger.info("Computing embeddings …")
    pred_emb = compute_motion_embeddings(
        interp_motions, test_lengths, move_enc, mot_enc, device
    )
    gt_emb = compute_motion_embeddings(
        test_motions_gt, test_lengths, move_enc, mot_enc, device
    )

    # ---- Metrics -----------------------------------------------------------
    diversity_times = min(cfg.TEST.DIVERSITY_TIMES, len(test_lengths) - 1)
    metrics = compute_metrics(pred_emb, gt_emb, diversity_times=diversity_times)
    print_metrics(metrics, title="Baseline: VAE Latent Interpolation")

    out_dir = Path(cfg.FOLDER) / cfg.model.model_type / cfg.NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, str(out_dir / f"metrics_{cfg.TIME}.json"))


if __name__ == "__main__":
    main()
