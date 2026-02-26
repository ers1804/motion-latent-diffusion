"""
Baseline 2: Trajectory-Only + Kinematic Fitting

Simulates what a classical "trajectory prediction → body fitting" pipeline
would produce, without a learned diffusion model.

HumanML3D feature layout (263-D per frame)
-------------------------------------------
  [0  ]     root angular velocity
  [1:3]     root linear velocity (x, z in local frame)
  [3  ]     root height
  [4:67]    21 joint positions relative to root   (63-D)
  [67:193]  22 joint 6-D rotations               (126-D)
  [193:259] 22 joint velocities                  (66-D)
  [259:263] foot contact labels                  (4-D)

Two evaluation modes are implemented (choose with --mode):

  oracle   (default)
      "GT-Root + Mean-Body"
      Root features [0:4] are taken from the GT test motion (oracle: perfect
      trajectory).  All body and foot features [4:263] are set to the training
      mean, which in the normalised feature space is exactly 0.
      → Upper bound for a trajectory-only approach given perfect prediction.

  mean
      "Mean-Motion" (all-zeros)
      Every frame of every prediction is the zero vector, corresponding to
      the per-feature training mean.  This is the lower bound: a model that
      ignores everything and predicts "average pedestrian pose" for all frames.

Usage:
    # Oracle mode (uses GT root velocity — upper bound)
    python baselines/eval_traj_kinematic.py \\
        --cfg configs/config_ego_motion_baseline_traj_kinematic.yaml \\
        --cfg_assets configs/assets.yaml

    # Mean-motion mode (lower bound)
    python baselines/eval_traj_kinematic.py \\
        --cfg configs/config_ego_motion_baseline_traj_kinematic.yaml \\
        --cfg_assets configs/assets.yaml \\
        --mode mean
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mld.config import parse_args
from mld.data.EgoMotion import EgoMotionDataModule, ego_motion_collate
from mld.utils.logger import create_logger
from baselines.baseline_utils import (
    compute_metrics,
    compute_motion_embeddings,
    load_t2m_evaluator,
    print_metrics,
    save_metrics,
)

# Feature index constants (HumanML3D 263-D format)
ROOT_START = 0
ROOT_END   = 4    # root velocity: [ang_vel, vel_x, vel_z, height]
BODY_START = 4
NFEATS     = 263

BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Motion synthesis functions
# ---------------------------------------------------------------------------

def oracle_root_mean_body(motions_gt: torch.Tensor, lengths: list) -> torch.Tensor:
    """
    Oracle trajectory + mean body.

    Takes root features (indices 0–3) from the GT motion and replaces all
    body / foot features with 0 (= per-feature training mean in normalised
    space).

    Args:
        motions_gt: (N, T, 263) GT motion tensor (already normalised).
        lengths:    list[int] of actual sequence lengths.

    Returns:
        (N, T, 263) prediction tensor.
    """
    pred = torch.zeros_like(motions_gt)
    pred[:, :, ROOT_START:ROOT_END] = motions_gt[:, :, ROOT_START:ROOT_END]
    # Zero body features outside the valid sequence length (already zero from zeros_like)
    return pred


def mean_motion(motions_gt: torch.Tensor, lengths: list) -> torch.Tensor:
    """
    Mean-motion (all-zeros) baseline — the training mean in normalised space.

    Args:
        motions_gt: (N, T, 263) GT tensor (only used for shape / lengths).
        lengths:    list[int].

    Returns:
        (N, T, 263) zero tensor.
    """
    return torch.zeros_like(motions_gt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # parse_args handles --cfg / --cfg_assets; we add --mode on top
    cfg = parse_args(phase="test")
    cfg.FOLDER = cfg.TEST.FOLDER

    # Manual extra argument: --mode {oracle|mean}
    mode = "oracle"
    raw_argv = sys.argv[1:]
    if "--mode" in raw_argv:
        idx = raw_argv.index("--mode")
        if idx + 1 < len(raw_argv):
            mode = raw_argv[idx + 1]
    if mode not in ("oracle", "mean"):
        raise ValueError(f"--mode must be 'oracle' or 'mean', got '{mode}'")

    logger = create_logger(cfg, phase="test")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Traj+Kinematic baseline — mode: {mode}")

    # ---- Dataset -----------------------------------------------------------
    datamodule = EgoMotionDataModule(cfg, logger=logger)
    datamodule.setup(stage="test")

    test_loader = DataLoader(
        datamodule.test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=ego_motion_collate,
    )

    logger.info("Collecting test motions …")
    gt_motions_list, lengths_all = [], []
    for batch in tqdm(test_loader, desc="Test"):
        gt_motions_list.append(batch["motion"])   # (B, T, 263)
        lengths_all.extend(batch["length"])
    gt_motions = torch.cat(gt_motions_list, dim=0)   # (N, T, 263)
    logger.info(f"  {len(lengths_all)} test samples")

    # ---- Generate predictions ----------------------------------------------
    if mode == "oracle":
        title = "Baseline: GT-Root + Mean-Body (Oracle Trajectory)"
        pred_motions = oracle_root_mean_body(gt_motions, lengths_all)
    else:
        title = "Baseline: Mean Motion (All-Zeros)"
        pred_motions = mean_motion(gt_motions, lengths_all)

    # ---- T2M embeddings ----------------------------------------------------
    logger.info("Loading t2m evaluator …")
    move_enc, mot_enc = load_t2m_evaluator(cfg, device)

    logger.info("Computing prediction embeddings …")
    pred_emb = compute_motion_embeddings(
        pred_motions, lengths_all, move_enc, mot_enc, device
    )

    logger.info("Computing GT embeddings …")
    gt_emb = compute_motion_embeddings(
        gt_motions, lengths_all, move_enc, mot_enc, device
    )

    # ---- Metrics -----------------------------------------------------------
    diversity_times = min(cfg.TEST.DIVERSITY_TIMES, len(lengths_all) - 1)
    metrics = compute_metrics(pred_emb, gt_emb, diversity_times=diversity_times)
    print_metrics(metrics, title=title)

    out_dir = Path(cfg.FOLDER) / cfg.model.model_type / f"{cfg.NAME}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, str(out_dir / f"metrics_{cfg.TIME}.json"))


if __name__ == "__main__":
    main()
