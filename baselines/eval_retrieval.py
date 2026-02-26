"""
Baseline 3: Retrieval (Nearest-Neighbour)

For every test ego trajectory, the closest training ego trajectory is found
(by subsampled L2 distance) and its corresponding motion is used as the
prediction.  This establishes an upper-bound on realism using only training
data: a perfect retrieval system could achieve zero FID if the test
distribution were fully covered by the training set.

Retrieval method
----------------
Each ego trajectory is resampled to N_RESAMPLE=32 equally-spaced 2-D points,
normalised to zero-mean/unit-std over the dataset, then flattened to a 64-D
descriptor.  L2 nearest-neighbour search finds the best match.

Usage:
    python baselines/eval_retrieval.py \
        --cfg configs/config_ego_motion_baseline_retrieval.yaml \
        --cfg_assets configs/assets.yaml
"""

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

N_RESAMPLE = 32   # ego descriptor length (number of equally-spaced points)
BATCH_SIZE = 256  # mini-batch for embedding computation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resample_trajectory(traj: np.ndarray, n: int) -> np.ndarray:
    """
    Linearly resample a (T, 2) trajectory to exactly n points.

    Returns (n, 2) array.
    """
    T = len(traj)
    if T == 1:
        return np.tile(traj, (n, 1))
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, n)
    x = np.interp(new_idx, old_idx, traj[:, 0])
    z = np.interp(new_idx, old_idx, traj[:, 1])
    return np.stack([x, z], axis=1)   # (n, 2)


def build_descriptors(egos: list, n: int = N_RESAMPLE) -> np.ndarray:
    """
    Build L2-comparable descriptors from a list of ego trajectories.

    Args:
        egos: list of (T, 2) numpy arrays (raw, un-padded ego trajectories)
        n:    number of resampled points

    Returns:
        (M, 2*n) float32 array, zero-mean / unit-std normalised column-wise.
    """
    descs = np.stack(
        [resample_trajectory(e, n).flatten() for e in egos], axis=0
    ).astype(np.float32)
    # Column-wise normalisation (global, across all descriptors)
    mu = descs.mean(axis=0, keepdims=True)
    sd = descs.std(axis=0, keepdims=True) + 1e-8
    return (descs - mu) / sd, mu, sd


def nn_retrieve(
    test_descs: np.ndarray,
    train_descs: np.ndarray,
    chunk: int = 512,
) -> np.ndarray:
    """
    Vectorised nearest-neighbour retrieval.

    Args:
        test_descs:  (N_test, D) query descriptors
        train_descs: (N_train, D) database descriptors
        chunk:       number of test samples processed at once

    Returns:
        (N_test,) int array of retrieved training-set indices.
    """
    indices = np.empty(len(test_descs), dtype=np.int64)
    for start in range(0, len(test_descs), chunk):
        end = min(start + chunk, len(test_descs))
        q = test_descs[start:end]                          # (B, D)
        dists = np.sum((q[:, None] - train_descs[None]) ** 2, axis=-1)  # (B, N_train)
        indices[start:end] = np.argmin(dists, axis=1)
    return indices


# ---------------------------------------------------------------------------
# Dataset helpers (load raw ego trajectories and normalised motions)
# ---------------------------------------------------------------------------

def load_split(datamodule: EgoMotionDataModule, split: str):
    """
    Load all samples from a given split and return:
        egos    – list of (T, 2) numpy arrays (real ego length, un-padded)
        motions – (N, T_max, 263) float32 tensor (padded, normalised)
        lengths – list[int]
    """
    ds = datamodule.test_dataset if split == "test" else datamodule.train_dataset

    # Pull raw samples without going through the padded batch interface
    egos = []
    motions = []
    lengths = []

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=ego_motion_collate,
        pin_memory=False,
    )

    for batch in tqdm(loader, desc=f"Loading {split}"):
        ego_batch = batch["ego"]          # (B, T_ego, 2) padded
        mot_batch = batch["motion"]       # (B, T_mot, 263) padded
        len_batch = batch["length"]       # list[int]
        ego_len_batch = batch["ego_length"]

        for b in range(len(len_batch)):
            el = ego_len_batch[b]
            egos.append(ego_batch[b, :el].numpy())   # (el, 2) un-padded
            motions.append(mot_batch[b])              # (T_max, 263)
            lengths.append(len_batch[b])

    motions = torch.stack(motions, dim=0)  # (N, T_max, 263)
    return egos, motions, lengths


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
    datamodule.setup(stage="fit")   # loads train split
    datamodule.setup(stage="test")  # loads test split

    logger.info("Loading train split for retrieval database …")
    train_egos, train_motions, train_lengths = load_split(datamodule, "train")
    logger.info(f"  Train: {len(train_lengths)} samples")

    logger.info("Loading test split for queries …")
    test_egos, test_motions, test_lengths = load_split(datamodule, "test")
    logger.info(f"  Test:  {len(test_lengths)} samples")

    # ---- Build descriptors and retrieve ------------------------------------
    logger.info("Building ego descriptors …")
    all_egos = train_egos + test_egos
    all_descs_raw = np.stack(
        [resample_trajectory(e, N_RESAMPLE).flatten() for e in all_egos],
        axis=0,
    ).astype(np.float32)
    # Global normalisation
    mu = all_descs_raw.mean(axis=0, keepdims=True)
    sd = all_descs_raw.std(axis=0, keepdims=True) + 1e-8
    all_descs = (all_descs_raw - mu) / sd

    n_train = len(train_lengths)
    train_descs = all_descs[:n_train]
    test_descs  = all_descs[n_train:]

    logger.info("Retrieving nearest neighbours …")
    nn_idx = nn_retrieve(test_descs, train_descs)
    logger.info(
        f"  Mean NN distance: {np.mean(np.sum((test_descs - train_descs[nn_idx])**2, axis=1)**.5):.4f}"
    )

    # Gather retrieved motions
    retrieved_motions = train_motions[nn_idx]   # (N_test, T_train_max, 263)
    retrieved_lengths = [train_lengths[i] for i in nn_idx]

    # ---- T2M embeddings ----------------------------------------------------
    logger.info("Loading t2m evaluator …")
    move_enc, mot_enc = load_t2m_evaluator(cfg, device)

    logger.info("Computing embeddings for retrieved motions …")
    pred_emb = compute_motion_embeddings(
        retrieved_motions, retrieved_lengths, move_enc, mot_enc, device
    )

    logger.info("Computing embeddings for GT test motions …")
    gt_emb = compute_motion_embeddings(
        test_motions, test_lengths, move_enc, mot_enc, device
    )

    # ---- Metrics -----------------------------------------------------------
    diversity_times = min(cfg.TEST.DIVERSITY_TIMES, len(test_lengths) - 1)
    metrics = compute_metrics(pred_emb, gt_emb, diversity_times=diversity_times)
    print_metrics(metrics, title="Baseline: Retrieval (Nearest-Neighbour Ego)")

    out_dir = Path(cfg.FOLDER) / cfg.model.model_type / cfg.NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, str(out_dir / f"metrics_{cfg.TIME}.json"))


if __name__ == "__main__":
    main()
