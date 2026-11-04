#!/usr/bin/env python3
"""Per-source breakdown of the scene-split cells, and the guidance trade-off on validation, from existing dumps.

ECS needs stopping and walking conditions, so it is undefined for a source with only a handful of rows.
Usage: python research/src/resplit_breakdown.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_resplit import cell_of, load, metrics  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    sel = json.load(open(DATA / "resplit_selection.json"))
    epochs = json.load(open(DATA / "resplit_epochs.json"))
    for split in ("scene_val", "scene_test"):
        print(f"== {split}: per source, cell mean over seeds at the selected guidance")
        dumps = {r: load(DATA / f"ade_fde_{r}_ep{s['epoch']}_cfg{s['guidance']}_{split}_k5.npz") for r, s in sel.items()}
        source = np.array([str(p).split("/")[0] for p in next(iter(dumps.values()))["paths"]])
        for name in ("ava", "nuscenes", "waymo"):
            idx = np.where(source == name)[0]
            line = f"  {name:9s} N={len(idx):4d} "
            for cell in ("pooled", "egoped", "uncond"):
                r = [metrics(dumps[k], idx) for k in dumps if cell_of(k) == cell]
                line += f" {cell} ADE {np.mean([x['ADE'] for x in r]):.2f} ECS {np.mean([x['ECS'] for x in r]):+.2f} |"
            print(line)
    print("== guidance on validation: cell means at each run's selected checkpoint")
    for cell in ("pooled", "egoped", "pooled_ia", "egoped_ia"):
        for g in (5, 10, 15):
            r = [metrics(load(DATA / f"ade_fde_{run}_ep{v['epoch']}_cfg{g}_scene_val_k5.npz"), slice(None))
                 for run, v in epochs.items() if cell_of(run) == cell]
            print(f"  {cell:10s} cfg {g:2d}: " + "  ".join(f"{k} {np.mean([x[k] for x in r]):.3f}" for k in ("ADE", "minADE", "ECS")))


if __name__ == "__main__":
    main()
