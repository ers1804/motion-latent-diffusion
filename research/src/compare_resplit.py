#!/usr/bin/env python3
"""The 2x2 of conditioning and recipe on the scene split, from K=5 dumps (plan steps 4-6).

Seeds of one cell are averaged, then the bootstrap resamples scenes of that average. It says nothing about
seed variance, so every contrast also prints the range over all seed pairs. Rule, fixed before any test
result is seen: a model beats another only if the CI excludes 0 and all seed pairs agree in sign. Primary
contrast: egoped vs pooled, mean ADE. Delta > 0 means the first cell is better.

Selection file: {"rs1_egoped_s1234": {"epoch": 2999, "guidance": 10}, ...}
Usage: python research/src/compare_resplit.py --split scene_val --selection <json> [<json> ...] [--data research/data]
Several selection files are merged, e.g. the 2x2 and the encoder ablations.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intention_labels import is_stopping  # noqa: E402
from make_scene_split import scene_of  # noqa: E402

LETTER = {"ava": "A", "nuscenes": "N", "waymo": "W"}
# Brier: of the stop frequency over K samples, as in the first test pass. BrierC: the same with the sampling noise
# of K samples removed, the score reported since 2026-11-13. E[(p_hat - y)^2] = (pi - y)^2 + pi (1 - pi) / K, and
# p_hat (1 - p_hat) / (K - 1) estimates the last term without bias. The raw score penalises a model for spreading
# its samples, the corrected one compares with a stated probability.
SIGN = dict(ADE=-1, minADE=-1, ECS=1, Brier=-1, BrierC=-1)
CONTRASTS = [("egoped", "pooled"), ("egoped_ia", "pooled_ia"),  # encoder at matched recipe
             ("egoped", "egoped_ia"), ("pooled", "pooled_ia"),  # recipe per encoder
             ("egoped", "uncond"), ("pooled", "uncond"),  # against the ego-blind prior of that recipe
             ("egoped_ia", "uncond_ia"), ("pooled_ia", "uncond_ia"),
             ("egoped", "unfrozen"),
             ("pooled", "pooled_unfrozen"),  # freezing the pretrained encoder
             ("pooled_unfrozen", "pooled_scratch")]  # pretraining, with the encoder trainable in both


def cell_of(run):
    return re.sub(r"_s\d+$", "", re.sub(r"^rs\d+_", "", run))  # rs1_pooled_s1234 and rs2_pooled_s1234 -> pooled


def scene_key(path):
    source, _, stem = path.split("/")
    return scene_of(LETTER[source] + Path(stem).stem)


def load(path):
    d = np.load(path)
    gen, gt, lens = d["roots_gen"], d["roots_gt"], d["lengths"]
    stop = np.array([is_stopping(gt[i, :lens[i]]) for i in range(len(gt))], dtype=float)
    p = np.array([np.mean([is_stopping(gen[i, k, :lens[i]]) for k in range(gen.shape[1])]) for i in range(len(gt))])
    return dict(ade=d["ades"].mean(1), minade=d["ades"].min(1), p=p, stop=stop, gt=gt, lens=lens, paths=d["paths"],
                k=gen.shape[1])


def metrics(m, idx):
    p, s = m["p"][idx], m["stop"][idx]
    brier = ((p - s) ** 2).mean()
    noise = (p * (1 - p)).mean() / (m["k"] - 1) if m["k"] > 1 else 0.0  # K=1: a deterministic model states 0 or 1
    return dict(ADE=m["ade"][idx].mean(), minADE=m["minade"][idx].mean(),
                ECS=p[s == 1].mean() - p[s == 0].mean(), Brier=brier, BrierC=brier - noise)


def cell_metrics(runs, idx):
    rows = [metrics(r, idx) for r in runs]
    return {k: float(np.mean([r[k] for r in rows])) for k in SIGN}


def compare(a, b, clusters, n=2000, seed=0):
    """{metric: (delta, CI low, CI high, p, lowest seed-pair delta, highest seed-pair delta)}."""
    rng = np.random.default_rng(seed)
    keys = sorted(clusters)
    every = np.concatenate([clusters[k] for k in keys])
    obs_a, obs_b = cell_metrics(a, every), cell_metrics(b, every)
    pairs = [(metrics(x, every), metrics(y, every)) for x in a for y in b]
    boot = {k: [] for k in SIGN}
    for _ in range(n):
        idx = np.concatenate([clusters[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        ma, mb = cell_metrics(a, idx), cell_metrics(b, idx)
        for k in SIGN:
            boot[k].append(SIGN[k] * (ma[k] - mb[k]))
    out = {}
    for k in SIGN:
        v = np.array(boot[k])
        per_pair = [SIGN[k] * (x[k] - y[k]) for x, y in pairs]
        out[k] = (SIGN[k] * (obs_a[k] - obs_b[k]), *np.percentile(v, [2.5, 97.5]),
                  min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())), min(per_pair), max(per_pair))
    return out


def beats(result):
    """The fixed rule: CI excludes 0 and all seed pairs agree in sign."""
    delta, low, high, _, lo, hi = result
    return (low > 0 and lo > 0) if delta > 0 else (high < 0 and hi < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["scene_val", "scene_test"])
    ap.add_argument("--selection", required=True, nargs="+")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()

    selection = {}
    for f in args.selection:
        selection.update(json.load(open(f)))
    cells, ref = {}, None
    for run, s in sorted(selection.items()):
        f = Path(args.data) / f"ade_fde_{run}_ep{s['epoch']}_cfg{s['guidance']}_{args.split}_k5.npz"
        if not f.exists():
            print(f"missing {f.name}")
            continue
        m = load(f)
        ref = ref or m
        err = max(np.abs(m["gt"][i, :ref["lens"][i]] - ref["gt"][i, :ref["lens"][i]]).max() for i in range(len(ref["gt"])))
        assert err < 1e-3 and (m["lens"] == ref["lens"]).all() and (m["paths"] == ref["paths"]).all(), (run, err)
        m["run"] = run
        cells.setdefault(cell_of(run), []).append(m)
    for cell, runs in cells.items():  # rs1 and rs2 runs of one kind must not be averaged into one cell
        assert len({m["run"].split("_")[0] for m in runs}) == 1, f"{cell}: {[m['run'] for m in runs]}"

    clusters = {}
    for i, p in enumerate(ref["paths"]):
        clusters.setdefault(scene_key(str(p)), []).append(i)
    clusters = {k: np.array(v) for k, v in clusters.items()}
    every = np.arange(len(ref["paths"]))
    print(f"split={args.split} N={len(every)} scenes={len(clusters)} stopping rate={ref['stop'].mean():.3f}")

    print(f"\n{'run':26s}{'ADE':>7s}{'minADE':>8s}{'ECS':>7s}{'Brier':>7s}{'BrierC':>8s}{'gen stop':>10s}")
    for cell in sorted(cells):
        for m in cells[cell]:
            r = metrics(m, every)
            print(f"{m['run']:26s}{r['ADE']:7.3f}{r['minADE']:8.3f}{r['ECS']:7.3f}{r['Brier']:7.3f}{r['BrierC']:8.3f}{m['p'].mean():10.3f}")

    print("\ndelta>0 = first cell better; [95% CI, scene bootstrap]; p; (seed pair range); * = rule met")
    for a, b in CONTRASTS:
        if a not in cells or b not in cells:
            continue
        r = compare(cells[a], cells[b], clusters)
        print(f"{a} vs {b}")
        for k, v in r.items():
            print(f"  {k:7s}{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] p={v[3]:.3f} ({v[4]:+.3f},{v[5]:+.3f}){' *' if beats(v) else ''}")


if __name__ == "__main__":
    main()
