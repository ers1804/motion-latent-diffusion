#!/usr/bin/env python3
"""Pick the checkpoint, then the guidance, of each retrained run from validation FID (plan step 4).

Rules, fixed before any result is seen; the test partition plays no part.
- Checkpoint: lowest training-time validation FID among saved checkpoints. Training-time validation feeds
  unconditional runs the real ego, so for them the FIDs come from slurm/resplit/select_uncond.sh instead.
- Guidance (--guidance): one value per cell, the one with the lowest validation FID averaged over its seeds,
  from slurm/resplit/eval.sh runs at the selected checkpoints. Unconditional runs keep 1.

Usage, from the repository root on the cluster:
  python research/src/select_resplit.py --out research/data/resplit_epochs.json
  python research/src/select_resplit.py --guidance research/data/resplit_epochs.json --out research/data/resplit_selection.json
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

LINE = re.compile(r"Epoch (\d+): .*?R_TOP_1 ([\d.e+-]+).*?\bFID ([\d.e+-]+).*?gt_Diversity ([\d.e+-]+)")
DATA = Path(__file__).resolve().parent.parent / "data"


def read_run(path):
    """{epoch: (FID, R@1, gt_Diversity)} for epochs with validation; logs repeat values in between."""
    rows = {}
    for log in sorted(glob.glob(f"{path}/log_*_train.log")):
        with open(log, errors="ignore") as f:
            for line in f:
                m = LINE.search(line)
                if m and (int(m.group(1)) + 1) % 100 == 0:
                    rows[int(m.group(1))] = (float(m.group(3)), float(m.group(2)), float(m.group(4)))
    return rows


def read_metrics(pattern):
    """The newest metrics json under results/mld/<pattern>, or None."""
    files = sorted(glob.glob(f"{pattern}/metrics_*.json"))
    return json.load(open(files[-1])) if files else None


def read_uncond(results, name):
    """{epoch: (FID, nan, gt_Diversity)} from the zero-ego validation of select_uncond.sh."""
    rows = {}
    for d in glob.glob(f"{results}/selu_val_{name}_ep*"):
        m = read_metrics(d)
        if m:
            rows[int(d.rsplit("_ep", 1)[1])] = (m["Metrics/FID/mean"], float("nan"), m["Metrics/gt_Diversity/mean"])
    return rows


def cell_of(run):
    return re.sub(r"_s\d+$", "", run.removeprefix("rs1_"))


def epochs(args):
    selected = {}
    for path in sorted(glob.glob(f"{args.root}/{args.pattern}")):
        name = os.path.basename(path)
        if name == "rs1_vae_latent4":
            continue
        rows = read_uncond(args.results, name) if "uncond" in name else read_run(path)
        saved = {int(x[6:-5]) for x in os.listdir(f"{path}/checkpoints")} if os.path.isdir(f"{path}/checkpoints") else set()
        usable = {e: v for e, v in rows.items() if e in saved}
        if not usable:
            print(f"{name:28s} no validated checkpoint yet")
            continue
        best = min(usable, key=lambda e: usable[e][0])
        fid, r1, gt = usable[best]
        selected[name] = dict(epoch=best, val_fid=fid, val_r1=r1, gt_diversity=gt, last_epoch=max(saved))
        print(f"{name:28s} best ep {best:5d}  FID {fid:7.3f}  R@1 {r1:.3f}  gt_Div {gt:.3f}  "
              f"last saved {max(saved)}  validated {len(usable)}")
    return selected


def guidance(args):
    chosen = json.load(open(args.guidance))
    table = {}  # cell -> guidance -> [FID per seed]
    print(f"{'run':26s}{'ep':>6s}{'cfg':>5s}{'val FID':>9s}{'val ADE':>9s}")
    for run, s in sorted(chosen.items()):
        scales = [1] if "uncond" in run else args.scales
        for g in scales:
            tag = f"{run}_ep{s['epoch']}_cfg{g}"
            m = read_metrics(f"{args.results}/eval_val_{tag}")
            dump = DATA / f"ade_fde_{tag}_scene_val_k5.npz"
            ade = np.load(dump)["ades"].mean() if dump.exists() else float("nan")
            fid = m["Metrics/FID/mean"] if m else float("nan")
            table.setdefault(cell_of(run), {}).setdefault(g, []).append(fid)
            print(f"{run:26s}{s['epoch']:6d}{g:5g}{fid:9.3f}{ade:9.3f}")
    best = {}
    for cell, by_g in sorted(table.items()):
        complete = {g: v for g, v in by_g.items() if not np.isnan(v).any()}
        if len(complete) < len(by_g):
            print(f"{cell}: missing sweep results, no guidance chosen")
            continue
        best[cell] = min(complete, key=lambda g: np.mean(complete[g]))
        print(f"{cell:12s} guidance {best[cell]:g}  mean val FID " +
              "  ".join(f"cfg{g:g} {np.mean(v):.3f}" for g, v in sorted(complete.items())))
    return {run: dict(epoch=s["epoch"], guidance=best[cell_of(run)]) for run, s in chosen.items() if cell_of(run) in best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hnvme/workspace/v103fe12-ped_gen/models_resplit/mld")
    ap.add_argument("--pattern", default="rs1_*")
    ap.add_argument("--results", default="results/mld")
    ap.add_argument("--guidance", help="epochs json from the first step; switches to guidance selection")
    ap.add_argument("--scales", nargs="+", type=float, default=[5, 10, 15])
    ap.add_argument("--out")
    args = ap.parse_args()
    args.scales = [int(g) if g == int(g) else g for g in args.scales]

    selected = guidance(args) if args.guidance else epochs(args)
    if args.out:
        json.dump(selected, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
