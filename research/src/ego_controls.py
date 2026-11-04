#!/usr/bin/env python3
"""How much the method uses the ego trajectory: full input against the controls of eval_controls.sh.

Rows are cell means over the seeds of one model at its selected checkpoint and guidance. Deltas are full input
minus control, scene bootstrap with the seed-pair range, as in compare_resplit (delta > 0: full input better).

Usage: python research/src/ego_controls.py --split scene_test [--cell pooled] [--prior uncond]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_resplit import beats, cell_of, compare, load, metrics, scene_key  # noqa: E402

CONTROLS = [("hor4", "ego known for 4 s"), ("hor2", "ego known for 2 s"), ("hor1", "ego known for 1 s"),
            ("hor0", "vehicle frozen at start"), ("swap", "wrong vehicle (other scene)"), ("zero", "ego zeroed")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["scene_val", "scene_test"])
    ap.add_argument("--selection", default="research/data/resplit_selection.json")
    ap.add_argument("--cell", default="pooled")
    ap.add_argument("--prior", default="uncond")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()

    selection = json.load(open(args.selection))
    tags = {run: f"{run}_ep{s['epoch']}_cfg{s['guidance']}" for run, s in selection.items()}
    runs = sorted(r for r in tags if cell_of(r) == args.cell)
    dump = lambda tag: load(Path(args.data) / f"ade_fde_{tag}_{args.split}_k5.npz")
    full = [dump(tags[r]) for r in runs]
    prior = [dump(tags[r]) for r in tags if cell_of(r) == args.prior]

    clusters = {}
    for i, p in enumerate(full[0]["paths"]):
        clusters.setdefault(scene_key(str(p)), []).append(i)
    clusters = {k: np.array(v) for k, v in clusters.items()}
    every = np.arange(len(full[0]["paths"]))

    def row(name, models):
        r = [metrics(m, every) for m in models]
        print(f"{name:30s}" + "".join(f"{np.mean([x[k] for x in r]):8.3f}" for k in ("ADE", "minADE", "ECS", "Brier")))

    print(f"{args.split}: {args.cell} seeds {runs}; N={len(every)} scenes={len(clusters)}")
    print(f"{'input':30s}{'ADE':>8s}{'minADE':>8s}{'ECS':>8s}{'Brier':>8s}")
    row("full ego trajectory", full)
    controls = {}
    for suffix, name in CONTROLS:
        controls[suffix] = [dump(f"{tags[r]}_{suffix}") for r in runs]
        row(name, controls[suffix])
    row(f"trained prior ({args.prior})", prior)

    print("\nfull input minus control; delta>0 = full input better [95% CI] (seed pair range); * = rule met")
    for suffix, name in CONTROLS:
        r = compare(full, controls[suffix], clusters)
        print(f"{name:30s}" + "  ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]{'*' if beats(v) else ' '}"
                                       for k, v in r.items() if k in ("ADE", "ECS")))


if __name__ == "__main__":
    main()
