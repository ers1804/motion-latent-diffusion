#!/usr/bin/env python3
"""Deterministic seq2seq ego->motion regressor baseline (item 1a).

A standard non-diffusion learned baseline: a transformer encoder over the ego
trajectory (same EgoEncoder backbone as the main models) with a per-timestep
MLP head regressing the 263-D normalized motion features. Trained with masked
MSE under the SAME data recipe as H4 (uniform sampling, random crops).

Scored with the SAME t2m evaluator path as the trivial baselines
(baselines/baseline_utils.py) for FID/Diversity comparability, plus
root-trajectory ADE/FDE (K=1, deterministic).

Usage:
  python research/src/regressor_baseline.py --train          # ~30 min on 4090
  python research/src/regressor_baseline.py --eval           # val_test + full val
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

NAS = "/home/erik/NAS/methods/diffusion_gen"
VT = "research/data/heldout_split/val_test"
FV = f"{NAS}/data/vae/mean_std_txt/ava_nuscenes_waymo"
CKPT = "research/data/regressor_best.pt"
CFG = "configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml"


def build_cfg(split_dir):
    overrides = [
        f"DATASET.EGOMOTION.ROOT=[{NAS}/data/diffusion/ava, {NAS}/data/diffusion/nuscenes, {NAS}/data/diffusion/waymo]",
        f"DATASET.EGOMOTION.MEAN_STD_PATH={split_dir}",
        f"DATASET.EGOMOTION.EGO_MEAN_STD_PATH={split_dir}",
        "DATASET.EGOMOTION.INTERACTION_CROP=False",
        "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=False",
        "NAME=ego_motion_regressor_baseline",
        "TEST.SPLIT=val",
        "TRAIN.BATCH_SIZE=64",
    ]
    sys.argv = ["regressor", "--cfg", CFG, "--nodebug", "--overrides"] + overrides
    from mld.config import parse_args
    cfg = parse_args(phase="test")
    cfg.FOLDER = "results"
    return cfg


class EgoMotionRegressor(nn.Module):
    def __init__(self, nfeats=263, d=256):
        super().__init__()
        from mld.models.architectures.ego_encoder import EgoEncoder
        self.encoder = EgoEncoder(input_dim=2, latent_dim=d, num_layers=4,
                                  num_heads=4, ff_size=1024, dropout=0.1,
                                  activation="gelu")
        self.head = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, nfeats))

    def forward(self, ego):                    # ego: (B, T, 2)
        tokens = self.encoder(ego)             # (B, T, 256)
        return self.head(tokens)               # (B, T, 263)


def masked_mse(pred, target, lengths):
    B, T, _ = pred.shape
    mask = torch.zeros(B, T, 1, device=pred.device)
    for i, L in enumerate(lengths):
        mask[i, : int(L)] = 1.0
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() * pred.shape[-1])


def get_loaders(cfg):
    from mld.data.get_data import get_datasets
    dm = get_datasets(cfg, logger=None, phase="test")[0]
    dm.setup(stage=None)
    return dm


def train(args):
    import pytorch_lightning as pl
    cfg = build_cfg(FV)
    pl.seed_everything(cfg.SEED_VALUE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dm = get_loaders(cfg)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    model = EgoMotionRegressor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best = float("inf")
    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for batch in train_loader:
            ego = batch["ego"].to(device); mot = batch["motion"].to(device)
            loss = masked_mse(model(ego), mot, batch["length"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval(); vtot = vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    ego = batch["ego"].to(device); mot = batch["motion"].to(device)
                    vtot += masked_mse(model(ego), mot, batch["length"]).item(); vn += 1
            vmse = vtot / max(vn, 1)
            flag = ""
            if vmse < best:
                best = vmse
                torch.save({"model": model.state_dict(), "epoch": ep, "val_mse": vmse}, CKPT)
                flag = " *best*"
            print(f"epoch {ep:4d} train_mse={tot/max(n,1):.4f} val_mse={vmse:.4f}{flag}", flush=True)
    print(f"TRAIN_DONE best_val_mse={best:.4f}")


def evaluate(args):
    import pytorch_lightning as pl
    from baselines.baseline_utils import (load_t2m_evaluator, compute_motion_embeddings,
                                          compute_metrics, print_metrics, save_metrics)
    from mld.data.EgoMotion import ego_motion_collate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for split_name, split_dir in [("val_test", VT), ("full_val", FV)]:
        cfg = build_cfg(split_dir)
        pl.seed_everything(cfg.SEED_VALUE)
        dm = get_loaders(cfg)
        dm.setup(stage="test")
        loader = DataLoader(dm.test_dataset, batch_size=64, shuffle=False,
                            num_workers=4, collate_fn=ego_motion_collate)
        model = EgoMotionRegressor().to(device)
        state = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(state["model"]); model.eval()
        preds, gts, lengths = [], [], []
        with torch.no_grad():
            for batch in loader:
                ego = batch["ego"].to(device)
                preds.append(model(ego).cpu())
                gts.append(batch["motion"])
                lengths.extend(batch["length"])
        pred = torch.cat(preds); gt = torch.cat(gts)
        move_enc, mot_enc = load_t2m_evaluator(cfg, device)
        pe = compute_motion_embeddings(pred, lengths, move_enc, mot_enc, device)
        ge = compute_motion_embeddings(gt, lengths, move_enc, mot_enc, device)
        m = compute_metrics(pe, ge, diversity_times=min(cfg.TEST.DIVERSITY_TIMES, len(lengths) - 1))
        print_metrics(m, title=f"Regressor baseline ({split_name}, ep{state['epoch']})")
        save_metrics(m, f"research/data/regressor_metrics_{split_name}.json")
        # ADE/FDE (deterministic, K=1)
        j_p = dm.feats2joints(pred); j_g = dm.feats2joints(gt)
        rp = j_p[:, :, 0, [0, 2]].numpy(); rg = j_g[:, :, 0, [0, 2]].numpy()
        ades, fdes = [], []
        for i, L in enumerate(lengths):
            L = int(L)
            d = np.linalg.norm(rp[i, :L] - rg[i, :L], axis=-1)
            ades.append(d.mean()); fdes.append(d[L - 1])
        print(f"[{split_name}] ADE={np.mean(ades):.4f} FDE={np.mean(fdes):.4f} (K=1, N={len(lengths)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    if args.train:
        train(args)
    if args.eval:
        evaluate(args)
