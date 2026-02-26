"""
Shared utilities for all EgoMotion baselines.

All baselines evaluate against the same test split with the same t2m encoder,
giving directly comparable FID / Diversity / gt_Diversity numbers.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from rich import get_console
from rich.table import Table

# Add project root to sys.path so mld imports work when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mld.models.architectures import t2m_motionenc
from mld.models.metrics.utils import (
    calculate_activation_statistics_np,
    calculate_diversity_np,
    calculate_frechet_distance_np,
)


# ---------------------------------------------------------------------------
# T2M evaluator
# ---------------------------------------------------------------------------

def load_t2m_evaluator(cfg, device: str = "cuda"):
    """
    Load the pretrained t2m MovementConvEncoder + MotionEncoderBiGRUCo.

    The checkpoint is read from:
        {cfg.model.t2m_path}/{dataname}/t2m/text_mot_match/model/finest.tar
    """
    move_enc = t2m_motionenc.MovementConvEncoder(
        input_size=cfg.DATASET.NFEATS - 4,
        hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
        output_size=cfg.model.t2m_motionencoder.dim_move_latent,
    ).to(device).eval()

    mot_enc = t2m_motionenc.MotionEncoderBiGRUCo(
        input_size=cfg.model.t2m_motionencoder.dim_move_latent,
        hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
        output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
    ).to(device).eval()

    dataname = cfg.TEST.DATASETS[0]
    ckpt_path = os.path.join(
        cfg.model.t2m_path, dataname, "t2m/text_mot_match/model/finest.tar"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    move_enc.load_state_dict(ckpt["movement_encoder"])
    mot_enc.load_state_dict(ckpt["motion_encoder"])

    for p in move_enc.parameters():
        p.requires_grad = False
    for p in mot_enc.parameters():
        p.requires_grad = False

    return move_enc, mot_enc


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def compute_motion_embeddings(
    motions: torch.Tensor,
    lengths: List[int],
    move_enc,
    mot_enc,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute 512-D t2m motion embeddings for a collection of motions.

    Args:
        motions:    (N, T, 263) float tensor (in the same normalised feature
                    space that the model outputs / the VAE decodes into).
        lengths:    list of N actual sequence lengths (frames).
        move_enc:   MovementConvEncoder (on device, eval mode).
        mot_enc:    MotionEncoderBiGRUCo (on device, eval mode).
        device:     torch device string.
        batch_size: mini-batch size for the encoding pass.

    Returns:
        (N, 512) numpy float32 array.
    """
    if not isinstance(motions, torch.Tensor):
        motions = torch.stack(motions, dim=0)

    N = len(lengths)
    all_emb: List[np.ndarray] = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_mot = motions[start:end].to(device).float()
        batch_len = lengths[start:end]

        # MovementConvEncoder applies two stride-2 convolutions → divide by 4
        m_lens = torch.tensor(batch_len, device=device) // 4
        m_lens = torch.clamp(m_lens, min=1)

        # MotionEncoderBiGRUCo requires descending-sorted lengths
        m_lens_sorted, sort_idx = m_lens.sort(descending=True)
        unsort_idx = sort_idx.argsort()

        with torch.no_grad():
            mov = move_enc(batch_mot[..., :-4][sort_idx])
            lat = mot_enc(mov, m_lens_sorted)[unsort_idx]  # (B, 512)

        all_emb.append(lat.cpu().numpy())

    return np.concatenate(all_emb, axis=0)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    pred_emb: np.ndarray,
    gt_emb: np.ndarray,
    diversity_times: int = 300,
) -> dict:
    """
    Compute FID, Diversity, gt_Diversity from 512-D t2m embeddings.

    Args:
        pred_emb:        (N, 512) embeddings of generated / retrieved motions.
        gt_emb:          (N, 512) embeddings of ground-truth motions.
        diversity_times: number of random pairs for diversity estimation.

    Returns:
        dict with keys 'FID', 'Diversity', 'gt_Diversity'.
    """
    N = pred_emb.shape[0]
    div_times = min(diversity_times, N - 1)

    mu_pred, cov_pred = calculate_activation_statistics_np(pred_emb)
    mu_gt, cov_gt = calculate_activation_statistics_np(gt_emb)
    fid = calculate_frechet_distance_np(mu_gt, cov_gt, mu_pred, cov_pred)
    div = calculate_diversity_np(pred_emb, div_times)
    div_gt = calculate_diversity_np(gt_emb, div_times)

    return {
        "FID": float(fid),
        "Diversity": float(div),
        "gt_Diversity": float(div_gt),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, title: str = "Metrics") -> None:
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}")
    get_console().print(table, justify="center")


def save_metrics(metrics: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_path}")
