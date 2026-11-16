#!/usr/bin/env python3
"""Frozen models on the Munich external test set (munich_eval.py): per-condition metrics from K=5 root dumps.

Recorded stops come from the labelled attribute (manifest), not from the speed rule, and rows can be restricted to
a subset fixed in the manifest. Statistics and the decision rule are those of compare_resplit: seeds of a model are
averaged, scenes are resampled, a model beats another only if the 95% interval excludes 0 and all seed pairs agree.

Primary, fixed before the run: EgoPed (pooled) against the prior (uncond), mean ADE and ECS, on all sequences and on
the interactive subset. Everything else printed here is secondary.

Usage: python research/src/munich_report.py --selection research/data/resplit_selection.json [--variant motion]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_resplit import SIGN, beats, cell_metrics, cell_of, compare, load, metrics, scene_key  # noqa: E402
from munich_eval import OUT, load_sequences  # noqa: E402

SUBSETS = {
    "all": lambda r: True,
    "interactive": lambda r: r["kind"] != "none",
    "within 20 m": lambda r: r["dmin"] < 20,
}
STOPS = {"label": "stop", "speed rule": "stop_speed"}
NAMES = dict(pooled="EgoPed", egoped="Full-seq", uncond="prior", pooled_ia="EgoPed + IA", egoped_ia="Full-seq + IA",
             uncond_ia="prior + IA", regressor="regressor")
CONTRASTS = [("pooled", "uncond"), ("egoped", "uncond"), ("pooled", "egoped"), ("pooled_ia", "uncond_ia"),
             ("egoped_ia", "uncond_ia"), ("regressor", "pooled"), ("regressor", "uncond")]


def name_of(path):
    return "U" + Path(str(path)).stem


def check_rows(m, names, ped, length, tol=0.02):
    """The dump must hold the sequences of the manifest, in its order, with the recorded root decoded from the file."""
    index = {n: i for i, n in enumerate(names)}
    worst = 0.0
    for row, p in enumerate(m["paths"]):
        i = index[name_of(p)]
        assert m["lens"][row] == length[i], (p, m["lens"][row], length[i])
        worst = max(worst, float(np.abs(m["gt"][row, :length[i]] - ped[i, :length[i]]).max()))
    assert worst < tol, f"recorded root of the dump differs from the manifest by {worst:.3f} m"
    return worst


def restrict(m, keep, stop):
    out = dict(m, stop=stop[keep])
    for k in ("ade", "minade", "p", "gt", "lens", "paths"):
        out[k] = m[k][keep]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True, nargs="+")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--folder", default=str(OUT), help="sequence files and manifest")
    ap.add_argument("--variant", default="label", choices=["label", "motion"],
                    help="pedestrian heading: labelled (primary) or from its motion (sensitivity)")
    args = ap.parse_args()
    split = "munich" if args.variant == "label" else "munich_motion"
    names, ped, _, length, manifest = load_sequences(args.variant, args.folder)
    rows = {r["name"]: r for r in manifest["sequences"]}

    selection = {}
    for f in args.selection:
        selection.update(json.load(open(f)))
    files = {run: Path(args.data) / f"ade_fde_{run}_ep{s['epoch']}_cfg{s['guidance']}_{split}_k5.npz"
             for run, s in sorted(selection.items()) if cell_of(run) in NAMES}
    files.update({f.name.split(f"_{split}_")[0].removeprefix("ade_fde_"): f
                  for f in sorted(Path(args.data).glob(f"ade_fde_rs1_regressor_s*_{split}_k1.npz"))})
    full, ref, worst = {}, None, 0.0
    for run, f in files.items():
        if not f.exists():
            print(f"missing {f.name}")
            continue
        m = load(f)
        worst = max(worst, check_rows(m, names, ped, length))
        ref = ref or m
        assert (m["paths"] == ref["paths"]).all() and (m["lens"] == ref["lens"]).all(), run
        full.setdefault(cell_of(run), []).append(dict(m, run=run))
    if ref is None:
        sys.exit(f"no dumps of split {split} in {args.data}")
    order = [name_of(p) for p in ref["paths"]]
    print(f"Munich external test, heading variant {args.variant}: {len(order)} sequences, "
          f"{len({rows[n]['scene'] for n in order})} scenes; "
          f"recorded roots of the dumps match the manifest to {worst * 100:.1f} cm")
    print("seeds: " + ", ".join(f"{NAMES[c]} {len(full[c])}" for c in NAMES if c in full))

    for stop_name, key in STOPS.items():
        stop = np.array([float(rows[n][key]) for n in order])
        for subset, rule in SUBSETS.items():
            if stop_name != "label" and subset == "within 20 m":
                continue
            keep = np.array([rule(rows[n]) for n in order])
            cells = {c: [restrict(m, keep, stop) for m in runs] for c, runs in full.items()}
            n = int(keep.sum())
            clusters = {}
            for i, p in enumerate(np.asarray(ref["paths"])[keep]):
                clusters.setdefault(scene_key(str(p)), []).append(i)
            clusters = {k: np.array(v) for k, v in clusters.items()}
            every = np.arange(n)
            s = stop[keep]
            print(f"\n=== {subset}, recorded stop from the {stop_name}: N={n}, scenes={len(clusters)}, "
                  f"stops {int(s.sum())} ({s.mean():.3f})")
            print(f"{'model':15s}{'ADE':>18s}{'minADE':>9s}{'ECS':>18s}{'BrierC':>9s}{'gen stop':>10s}")
            for c in NAMES:
                if c not in cells:
                    continue
                per = [metrics(m, every) for m in cells[c]]
                mean = cell_metrics(cells[c], every)
                rng = lambda k: f"[{min(x[k] for x in per):.2f},{max(x[k] for x in per):.2f}]"  # noqa: E731
                print(f"{NAMES[c]:15s}{mean['ADE']:7.3f} {rng('ADE'):>10s}{mean['minADE']:9.3f}{mean['ECS']:+7.3f} {rng('ECS'):>10s}"
                      f"{mean['BrierC']:9.3f}{np.mean([m['p'].mean() for m in cells[c]]):10.3f}")
            print("delta>0 = first model better [95% CI, scene bootstrap] (seed pair range); * = rule met")
            for a, b in CONTRASTS:
                if a not in cells or b not in cells:
                    continue
                r = compare(cells[a], cells[b], clusters)
                print(f"  {NAMES[a]:13s} vs {NAMES[b]:11s} " + "  ".join(
                    f"{k} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] ({v[4]:+.3f},{v[5]:+.3f}){'*' if beats(v) else ' '}"
                    for k, v in r.items() if k in ("ADE", "ECS", "BrierC")))
    assert set(SIGN) >= {"ADE", "ECS", "BrierC"}


if __name__ == "__main__":
    main()
