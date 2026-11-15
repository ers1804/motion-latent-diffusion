#!/usr/bin/env python3
"""Secondary check of the rs2 decision (research/review_action_plan_2026-11-13.md): does feeding the ego in the
frame of the window start narrow the gap in coupling between windows that start after the track's first frame
and the other conditions?

For one cell, both generations at their own selected checkpoints and guidance, validation only: ECS, mean ADE
and stop probability on the two subsets, the rs2 minus rs1 contrast on each, and the contrast of the gap. On the
other conditions both generations receive the same ego input at evaluation, so a difference there comes from
training (the changed inputs of the windowed training tracks, seeds, checkpoints), not from the evaluated input.

Usage: python research/src/rs2_frame_check.py --selection <json> [<json> ...] [--cell pooled]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_resplit import cell_of, generation_of, load  # noqa: E402
from coupling_breakdown import Bootstrap, ecs, fmt, load_geometry  # noqa: E402

SPLIT = "scene_val"  # the test partition plays no part in this decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True, nargs="+")
    ap.add_argument("--cell", default="pooled")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()

    selection = {}
    for f in args.selection:
        selection.update(json.load(open(f)))
    runs = {"rs1": [], "rs2": []}
    for run, s in sorted(selection.items()):
        if cell_of(run) == args.cell and generation_of(run) in runs:
            runs[generation_of(run)].append(
                load(Path(args.data) / f"ade_fde_{run}_ep{s['epoch']}_cfg{s['guidance']}_{SPLIT}_k5.npz"))
    assert runs["rs1"] and runs["rs2"], {g: len(v) for g, v in runs.items()}
    ref = runs["rs1"][0]
    for group in runs.values():
        for m in group:
            assert (m["paths"] == ref["paths"]).all() and (m["lens"] == ref["lens"]).all()
    gt, lens, paths = ref["gt"], ref["lens"], ref["paths"]
    stop = ref["stop"].astype(bool)
    walk = ~stop
    _, offset, _, _ = load_geometry(SPLIT, paths, lens, gt)
    later = offset > 0
    boot = Bootstrap(paths)

    def subset_ecs(r, idx, mask):
        return ecs(r["p"][idx], (stop & mask)[idx], (walk & mask)[idx])

    print(f"cell={args.cell} split={SPLIT} N={len(gt)}; windows that start later {later.sum()}, others {(~later).sum()}; "
          f"seeds rs1 {len(runs['rs1'])}, rs2 {len(runs['rs2'])}")
    print("* = 95% scene-bootstrap interval excludes 0 and all seed pairs agree in sign")
    for g, group in runs.items():
        p, ade = np.mean([r["p"] for r in group], axis=0), np.mean([r["ade"] for r in group], axis=0)
        print(f"{g}  ECS later {ecs(p, stop & later, walk & later):+.3f}, others {ecs(p, stop & ~later, walk & ~later):+.3f}   "
              f"ADE later {ade[later].mean():.3f}, others {ade[~later].mean():.3f}   "
              f"stop probability later {p[later].mean():.3f}, others {p[~later].mean():.3f}")
    for label, mask in (("later", later), ("others", ~later)):
        e = boot.contrast(lambda r, idx: subset_ecs(r, idx, mask), runs["rs2"], runs["rs1"])
        a = boot.contrast(lambda r, idx: -r["ade"][idx][mask[idx]].mean(), runs["rs2"], runs["rs1"])
        print(f"rs2 minus rs1 on {label}: ECS {fmt(e)}   ADE gain (positive = rs2 lower) {fmt(a)}")
    gap = boot.contrast(lambda r, idx: subset_ecs(r, idx, ~later) - subset_ecs(r, idx, later), runs["rs2"], runs["rs1"])
    print(f"gap in ECS (others minus later), rs2 minus rs1: {fmt(gap)}   (negative = the fix narrows the gap)")


if __name__ == "__main__":
    main()
