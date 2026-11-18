#!/usr/bin/env python3
"""Energy score of the K root paths of a condition, and the secondary analysis of 2026-11-18.

Mean ADE rewards a model for concentrating its samples (the deterministic regressor has the lowest), best-of-K
rewards it for spreading them (the ego-blind prior has the lowest). The energy score is a proper score of the
whole set: with d the mean root distance over the true length (the distance behind ADE),

    ES = mean_k d(x_k, y) - 1/2 * mean_{k != l} d(x_k, x_l)        (lower is better)

The second term is the average over pairs of different samples, which estimates E d(X, X') without bias. A
deterministic model (K=1, or K equal samples) has no spread, so its energy score is its ADE.

Protocol, fixed on validation and committed before any test number (research/review_action_plan_2026-11-13.md,
section "Energy score and retrieval baseline"):
  1. The score is computed from the K=5 dumps that exist, at the checkpoints and guidance selected by validation
     FID, which stay the reported setting.
  2. Secondary setting, for EgoPed and Full-seq: the guidance in {1, 2, 3, 5, 10, 15} with the lowest validation
     energy score, averaged over the seeds, at the same checkpoints (--select writes the file once every validation
     dump exists). Its cells are named <cell>@E. The scales under 5 were added because the score falls
     monotonically down to 5, the lower edge of the first sweep.
  3. Contrasts use the rule of compare_resplit (95% scene-bootstrap interval excludes 0 and all seed pairs agree).

Usage:
  python research/src/energy_score.py --select --out research/data/resplit_selection_energy.json
  python research/src/energy_score.py --split scene_val
  python research/src/energy_score.py --split scene_test        # once, after the commit
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_resplit import SIGN, beats, cell_of, compare, load, metrics, scene_key  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
SELECTIONS = ["resplit_selection.json", "resplit_selection_priors2.json", "resplit_selection_priors_ia2.json"]
MAIN = ["pooled", "egoped", "pooled_ia", "egoped_ia"]  # pooled is EgoPed, egoped is Full-seq (run names)
SECONDARY = ["pooled", "egoped"]  # the two models whose guidance selected by FID is 15
SCALES = [1, 2, 3, 5, 10, 15]
EXPECTED = dict({c: 3 for c in MAIN + ["uncond", "uncond_ia", "regressor"]}, retrieval=1)  # dumps per reported cell
SIGN_E = dict(SIGN, Energy=-1)
CONTRASTS = [  # the first cell is better where delta > 0
    ("pooled", "uncond"), ("pooled@E", "uncond"),  # the question: does EgoPed beat the ego-blind prior
    ("egoped", "uncond"), ("egoped@E", "uncond"),
    ("pooled_ia", "uncond_ia"), ("egoped_ia", "uncond_ia"),
    ("pooled@E", "pooled"), ("egoped@E", "egoped"),  # what the guidance does
    ("pooled", "egoped"), ("pooled@E", "egoped@E"),  # the encoders
    ("pooled", "regressor"), ("pooled@E", "regressor"), ("uncond", "regressor"),
    ("pooled", "retrieval"), ("pooled@E", "retrieval"), ("retrieval", "uncond"), ("retrieval", "regressor"),
]


def path_distance(a, b, lens):
    """Mean distance of two root paths over the true length. a, b: (N, T, 2), lens: (N,). Returns (N,)."""
    mask = np.arange(a.shape[1])[None, :] < np.asarray(lens)[:, None]
    dist = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=-1)
    return np.where(mask, dist, 0.0).sum(1) / np.asarray(lens)


def energy_terms(gen, gt, lens):
    """(error, spread) per condition. gen: (N, K, T, 2), gt: (N, T, 2).

    error is the mean distance of a sample to the recording (the mean ADE of the condition), spread the mean
    distance between two different samples, 0 for K=1.
    """
    k = gen.shape[1]
    error = np.mean([path_distance(gen[:, i], gt, lens) for i in range(k)], axis=0)
    pairs = [path_distance(gen[:, i], gen[:, j], lens) for i in range(k) for j in range(i + 1, k)]
    spread = np.mean(pairs, axis=0) if pairs else np.zeros(len(gt))
    return error, spread


def load_e(path):
    """compare_resplit.load plus the energy terms of every condition."""
    m = load(path)
    d = np.load(path)
    m["err"], m["spread"] = energy_terms(d["roots_gen"], d["roots_gt"], d["lengths"])
    assert np.allclose(m["err"], m["ade"], atol=1e-3), Path(path).name  # the same distance as the stored ADE
    m["energy"] = m["err"] - 0.5 * m["spread"]
    m["run"] = Path(path).name
    return m


def metrics_e(m, idx):
    return dict(metrics(m, idx), Energy=float(m["energy"][idx].mean()))


def read_selection(data):
    selection = {}
    for f in SELECTIONS:
        if (data / f).exists():
            selection.update(json.load(open(data / f)))
    return {r: s for r, s in selection.items() if r.startswith("rs1_")}


def dump(data, run, epoch, guidance, split):
    return data / f"ade_fde_{run}_ep{epoch}_cfg{guidance}_{split}_k5.npz"


def select(data, out):
    """Guidance with the lowest validation energy score for EgoPed and Full-seq, seeds averaged; checkpoints unchanged."""
    selection = read_selection(data)
    chosen = {}
    print("validation energy score by guidance, seed mean (per seed); the guidance selected by FID in brackets")
    for cell in SECONDARY:
        runs = sorted(r for r in selection if cell_of(r) == cell)
        score = {}
        for g in SCALES:  # every scale of the grid must be there, or the choice would depend on what happened to finish
            files = [dump(data, r, selection[r]["epoch"], g, "scene_val") for r in runs]
            assert len(files) == 3 and all(f.exists() for f in files), [f.name for f in files if not f.exists()]
            score[g] = [float(load_e(f)["energy"].mean()) for f in files]
        best = min(SCALES, key=lambda g: np.mean(score[g]))
        print(f"  {cell:10s} [{selection[runs[0]]['guidance']:2d}] "
              + "  ".join(f"w={g}: {np.mean(score[g]):.3f} ({', '.join(f'{v:.3f}' for v in score[g])})" for g in SCALES)
              + f"  -> {best}")
        for r in runs:
            chosen[r] = {"epoch": selection[r]["epoch"], "guidance": best}
    json.dump(chosen, open(out, "w"), indent=1, sort_keys=True)
    print("wrote", out)


def load_cells(data, split, energy_selection):
    selection = read_selection(data)
    cells, missing = {}, []

    def add(cell, f):
        if f.exists():
            cells.setdefault(cell, []).append(load_e(f))
        else:
            missing.append(f.name)

    for run, s in sorted(selection.items()):
        if cell_of(run) in MAIN + ["uncond", "uncond_ia"]:
            add(cell_of(run), dump(data, run, s["epoch"], s["guidance"], split))
    pending = [c + "@E" for c in SECONDARY]
    if Path(energy_selection).exists():
        for run, s in sorted(json.load(open(energy_selection)).items()):
            assert s["epoch"] == selection[run]["epoch"], run  # the secondary setting changes the guidance only
            add(cell_of(run) + "@E", dump(data, run, s["epoch"], s["guidance"], split))
        pending = [c for c in pending if c not in cells]
    for f in sorted(data.glob(f"ade_fde_rs1_regressor_s*_{split}_k1.npz")):
        add("regressor", f)
    add("retrieval", data / f"ade_fde_retrieval_{split}_k5.npz")
    # A cell with a dump missing would be compared on fewer seeds without a word, so nothing is computed then.
    # The secondary cells may be absent as a whole (not generated yet) but never in part.
    expected = dict(EXPECTED, **{c + "@E": 3 for c in SECONDARY if c + "@E" not in pending})
    sizes = {c: len(cells.get(c, [])) for c in expected}
    assert sizes == expected, (sizes, missing)
    return cells, pending


def report(data, split, energy_selection):
    cells, pending = load_cells(data, split, energy_selection)
    for name in pending:
        print("pending, not generated yet:", name)
    ref = cells["pooled"][0]
    for runs in cells.values():  # every dump holds the same conditions in the same order
        for m in runs:
            assert (m["lens"] == ref["lens"]).all() and (m["paths"] == ref["paths"]).all(), m["run"]
            assert max(np.abs(m["gt"][i, :n] - ref["gt"][i, :n]).max() for i, n in enumerate(ref["lens"])) < 1e-3, m["run"]
    clusters = {}
    for i, p in enumerate(ref["paths"]):
        clusters.setdefault(scene_key(str(p)), []).append(i)
    clusters = {k: np.array(v) for k, v in clusters.items()}
    every = np.arange(len(ref["paths"]))
    print(f"split={split} N={len(every)} scenes={len(clusters)}; lower energy is better; @E = guidance selected by validation energy")
    print(f"\n{'cell':14s}{'runs':>5s}{'ADE':>8s}{'minADE':>8s}{'spread':>8s}{'Energy':>8s}{'ECS':>7s}{'BrierC':>8s}   energy per run")
    for cell in MAIN + [c + "@E" for c in SECONDARY] + ["uncond", "uncond_ia", "regressor", "retrieval"]:
        if cell not in cells:
            continue
        rows = [metrics_e(m, every) for m in cells[cell]]
        mean = lambda k: np.mean([r[k] for r in rows])  # noqa: E731
        print(f"{cell:14s}{len(rows):5d}{mean('ADE'):8.3f}{mean('minADE'):8.3f}{np.mean([m['spread'].mean() for m in cells[cell]]):8.3f}"
              f"{mean('Energy'):8.3f}{mean('ECS'):7.3f}{mean('BrierC'):8.3f}   " + ", ".join(f"{r['Energy']:.3f}" for r in rows))
    print("\ndelta>0 = first cell better; [95% CI, scene bootstrap]; (seed pair range); * = rule met")
    for a, b in CONTRASTS:
        if a not in cells or b not in cells:
            continue
        r = compare(cells[a], cells[b], clusters, metrics_fn=metrics_e, sign=SIGN_E)
        print(f"{a} vs {b}")
        print("  " + "   ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] ({v[4]:+.3f},{v[5]:+.3f}){'*' if beats(v) else ''}"
                               for k, v in r.items() if k in ("ADE", "minADE", "ECS", "BrierC", "Energy")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["scene_val", "scene_test"])
    ap.add_argument("--select", action="store_true", help="choose the guidance by validation energy and write --out")
    ap.add_argument("--out", default=str(DATA / "resplit_selection_energy.json"))
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args()
    if args.select:
        select(Path(args.data), args.out)
    if args.split:
        report(Path(args.data), args.split, args.out)


if __name__ == "__main__":
    main()
