#!/usr/bin/env python3
"""Pick the checkpoint of each retrained run from its validation FID (plan step 4).

Rule, fixed before any result is seen: lowest training-time validation FID among saved checkpoints,
from the scene-split validation partition. The test partition plays no part. Unconditional runs are
listed but not selected here: training-time validation feeds them the real ego, so they are selected
offline with baselines/eval_uncond.py.

Usage: python research/src/select_resplit.py [--root <models>/mld] [--pattern 'rs1_*'] [--out <json>]
"""
import argparse
import glob
import json
import os
import re

LINE = re.compile(r"Epoch (\d+): .*?R_TOP_1 ([\d.e+-]+).*?\bFID ([\d.e+-]+).*?gt_Diversity ([\d.e+-]+)")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hnvme/workspace/v103fe12-ped_gen/models_resplit/mld")
    ap.add_argument("--pattern", default="rs1_*")
    ap.add_argument("--out")
    args = ap.parse_args()

    selected = {}
    for path in sorted(glob.glob(f"{args.root}/{args.pattern}")):
        name = os.path.basename(path)
        rows = read_run(path)
        saved = {int(x[6:-5]) for x in os.listdir(f"{path}/checkpoints")} if os.path.isdir(f"{path}/checkpoints") else set()
        usable = {e: v for e, v in rows.items() if e in saved}
        if not usable:
            print(f"{name:28s} no validated checkpoint yet")
            continue
        last = max(rows)
        if "uncond" in name:
            print(f"{name:28s} last epoch {last}; select offline")
            continue
        best = min(usable, key=lambda e: usable[e][0])
        fid, r1, gt = usable[best]
        selected[name] = dict(epoch=best, val_fid=fid, val_r1=r1, gt_diversity=gt, last_epoch=last)
        print(f"{name:28s} best ep {best:5d}  FID {fid:7.3f}  R@1 {r1:.3f}  gt_Div {gt:.3f}  last ep {last}")
    if args.out:
        json.dump(selected, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
