#!/usr/bin/env python3
"""Behavioral-validity probe (item 9): does p(stopping | ego) of GENERATED
motion match the ground truth?

Uses the root trajectories dumped by eval_ade_fde.py --save-roots (same K=5
samples that produced the ADE/FDE numbers). Stopping label per intention_labels
(speed < 0.2 m/s sustained >= 1 s); crossing is omitted here (base rate ~2%,
and the dumps store no ego trajectory) — documented as a rare-event annex.

Per model:
  marginal    : mean stop-rate over all generated samples vs GT base rate.
  separation  : mean p_model(stop|cond) among GT-stop conditions minus that
                among GT-walk conditions (0 = ego-blind; 1 = perfect).
  entropy     : mean binary entropy of p_model(stop|cond) — collapse indicator.
  brier       : mean (p_model - gt_label)^2 (calibration+discrimination).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intention_labels import is_stopping  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = ["h4", "h2", "h6", "h4_uncond"]


def probe(model):
    d = np.load(DATA / f"ade_fde_{model}_val_test_k5.npz")
    gen, gt, lens = d["roots_gen"], d["roots_gt"], d["lengths"]  # (N,K,T,2),(N,T,2),(N,)
    N, K = gen.shape[:2]
    gt_stop = np.array([is_stopping(gt[i, :lens[i]]) for i in range(N)], dtype=float)
    p = np.zeros(N)
    for i in range(N):
        L = int(lens[i])
        p[i] = np.mean([is_stopping(gen[i, k, :L]) for k in range(K)])
    eps = 1e-9
    ent = -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))
    sep = p[gt_stop == 1].mean() - p[gt_stop == 0].mean()
    return dict(N=N, gt_rate=gt_stop.mean(), gen_rate=p.mean(),
                separation=sep, mean_entropy=ent.mean(),
                brier=((p - gt_stop) ** 2).mean())


if __name__ == "__main__":
    print(f"{'model':10s} {'GT-rate':>8s} {'gen-rate':>9s} {'separation':>11s} {'entropy':>8s} {'brier':>7s}")
    for m in MODELS:
        r = probe(m)
        print(f"{m:10s} {r['gt_rate']:8.3f} {r['gen_rate']:9.3f} {r['separation']:11.3f} "
              f"{r['mean_entropy']:8.3f} {r['brier']:7.3f}")
