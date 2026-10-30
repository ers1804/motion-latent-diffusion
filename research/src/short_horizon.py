#!/usr/bin/env python3
"""Root metrics over the first 1.95 s (40 frames at 20 fps): our models against PedGen no-context.

Every trajectory starts at the origin in the frame of the initial facing. PedGen has no ego input and
was trained on other data, so this compares against a published ego-blind prior; it is no ranking of
architectures. Bootstrap resamples scenes.

Usage: python research/src/short_horizon.py [--split val_test] [--models h4 h2_nopipe ...]
"""
import argparse
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
H = 40


def errors(gen, gt):
    """gen (N,K,H,2), gt (N,H,2) -> per condition ADE, FDE, minADE, minFDE."""
    d = np.linalg.norm(gen - gt[:, None], axis=-1)
    a, f = d.mean(2), d[:, :, -1]
    return dict(ADE=a.mean(1), FDE=f.mean(1), minADE=a.min(1), minFDE=f.min(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_test")
    ap.add_argument("--models", nargs="+", default=["h4", "h4_seedB", "h4_seedC", "h2_nopipe", "ia", "h2", "uncond_trained"])
    ap.add_argument("--reference", default="h2_nopipe", help="dump that supplies GT, row paths and lengths")
    ap.add_argument("--bank", default=str(DATA / "pedgen_no_context_bank.npz"))
    args = ap.parse_args()

    ref = np.load(DATA / f"ade_fde_{args.reference}_{args.split}_k5.npz")
    gt = ref["roots_gt"][:, :H]
    gt = gt - gt[:, :1]
    keep = ref["lengths"] >= H  # full horizon only, so no model is scored on padding
    scenes = np.array(["/".join((p.split("/")[0], Path(p).stem.lstrip("M").split("_")[0])) for p in ref["paths"]])
    print(f"split={args.split} N={keep.sum()}/{len(gt)} GT speed m/s median "
          f"{np.median(np.linalg.norm(np.diff(gt, axis=1), axis=-1).mean(1) * 20):.2f}, "
          f"GT end z median {np.median(gt[:, -1, 1]):.2f}, end x median {np.median(gt[:, -1, 0]):.2f}")

    results = {}
    for m in args.models:
        gen = np.load(DATA / f"ade_fde_{m}_{args.split}_k5.npz")["roots_gen"][:, :, :H]
        results[m] = errors(gen - gen[:, :, :1], gt)
    bank = np.load(args.bank)["roots"]
    rng = np.random.default_rng(0)
    results["pedgen"] = errors(bank[rng.permutation(len(bank))[:len(gt) * 5]].reshape(len(gt), 5, H, 2), gt)

    print(f"{'model':16s}" + "".join(f"{k:>9s}" for k in ("ADE", "FDE", "minADE", "minFDE")))
    for m, r in results.items():
        print(f"{m:16s}" + "".join(f"{r[k][keep].mean():9.3f}" for k in ("ADE", "FDE", "minADE", "minFDE")))

    idx = {s: np.where((scenes == s) & keep)[0] for s in np.unique(scenes[keep])}
    keys = sorted(idx)
    print("\nPaired against PedGen, positive = model better [95% CI, scene bootstrap]")
    for m, r in results.items():
        if m == "pedgen":
            continue
        out = []
        for k in ("ADE", "FDE", "minADE", "minFDE"):
            diff = results["pedgen"][k] - r[k]
            boot = [diff[np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])].mean() for _ in range(2000)]
            out.append(f"{k} {diff[keep].mean():+.3f} [{np.percentile(boot, 2.5):+.3f},{np.percentile(boot, 97.5):+.3f}]")
        print(f"{m:16s}" + "  ".join(out))


if __name__ == "__main__":
    main()
