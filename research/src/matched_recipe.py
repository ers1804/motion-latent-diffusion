#!/usr/bin/env python3
"""Pooled vs full-sequence conditioning at matched training recipes, old split (plan step 3).

Bootstrap resamples scenes, because mirrored twins and pedestrians of one scene are not independent.
Delta > 0 means the first model is better: lower ADE/minADE/Brier, higher separation.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intention_labels import is_stopping  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
NAS = os.environ.get("EGOPED_NAS", "/home/erik/NAS/methods/diffusion_gen")
LETTER = {"ava": "A", "nuscenes": "N", "waymo": "W"}
MODELS = ["h4", "h4_seedB", "h4_seedC", "h2_nopipe", "ia", "ia_ep3399", "ia_seed2", "ia_seed2_ep1299",
          "h2", "h2_seedB", "h2_seedC", "h6", "uncond_trained", "uncond_pipeline"]
PAIRS = [  # no interaction recipe: EgoPed seeds against the single Pooled run
    ("h4", "h2_nopipe"), ("h4_seedB", "h2_nopipe"), ("h4_seedC", "h2_nopipe"),
    # interaction recipe: EgoPed-IA against Pooled-IA
    ("ia", "h2"), ("ia_ep3399", "h2"), ("ia_seed2", "h2"), ("ia_seed2_ep1299", "h2"),
    # the recipe effect inside each architecture
    ("h2_nopipe", "h2"), ("h4", "ia"),
    # both against the trained ego-blind prior
    ("h2_nopipe", "uncond_trained"), ("h4", "uncond_trained")]


def load(model):
    d = np.load(DATA / f"ade_fde_{model}_val_test_k5.npz")
    gen, gt, lens = d["roots_gen"], d["roots_gt"], d["lengths"]
    gt_stop = np.array([is_stopping(gt[i, :lens[i]]) for i in range(len(gt))], dtype=float)
    p = np.array([np.mean([is_stopping(gen[i, k, :lens[i]]) for k in range(gen.shape[1])]) for i in range(len(gt))])
    return dict(ade=d["ades"].mean(1), minade=d["ades"].min(1), p=p, stop=gt_stop, gt=gt, lens=lens)


def metrics(m, idx):
    p, s = m["p"][idx], m["stop"][idx]
    return dict(ADE=m["ade"][idx].mean(), minADE=m["minade"][idx].mean(),
                sep=p[s == 1].mean() - p[s == 0].mean(), Brier=((p - s) ** 2).mean())


def compare(a, b, clusters, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(clusters)
    obs_a, obs_b = metrics(a, np.concatenate(list(clusters.values()))), metrics(b, np.concatenate(list(clusters.values())))
    sign = dict(ADE=-1, minADE=-1, sep=1, Brier=-1)
    deltas = {k: sign[k] * (obs_a[k] - obs_b[k]) for k in sign}
    boot = {k: [] for k in sign}
    for _ in range(n):
        idx = np.concatenate([clusters[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        ma, mb = metrics(a, idx), metrics(b, idx)
        for k in sign:
            boot[k].append(sign[k] * (ma[k] - mb[k]))
    out = {}
    for k in sign:
        v = np.array(boot[k])
        out[k] = (deltas[k], *np.percentile(v, [2.5, 97.5]), min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))
    return out


def main():
    paths = np.load(DATA / "ade_fde_h2_nopipe_val_test_k5.npz")["paths"]
    names = [(p.split("/")[0], Path(p).stem) for p in paths]
    scenes = [(s, n.lstrip("M").split("_")[0]) for s, n in names]
    data = {m: load(m) for m in MODELS if (DATA / f"ade_fde_{m}_val_test_k5.npz").exists()}

    # Same rows and same windows in every dump, including those written on another machine.
    ref = data["h2_nopipe"]
    for m, d in data.items():
        err = max(np.abs(d["gt"][i, :ref["lens"][i]] - ref["gt"][i, :ref["lens"][i]]).max() for i in range(len(ref["gt"])))
        assert err < 1e-3 and (d["lens"] == ref["lens"]).all(), (m, err)
    print(f"alignment OK across {len(data)} dumps; N={len(paths)}; scenes={len(set(scenes))}")

    all_idx = np.arange(len(paths))
    print(f"\n{'model':16s}{'ADE':>7s}{'minADE':>8s}{'sep':>7s}{'Brier':>7s}")
    for m, d in data.items():
        r = metrics(d, all_idx)
        print(f"{m:16s}{r['ADE']:7.3f}{r['minADE']:8.3f}{r['sep']:7.3f}{r['Brier']:7.3f}")

    clusters = {}
    for i, s in enumerate(scenes):
        clusters.setdefault(s, []).append(i)
    clusters = {k: np.array(v) for k, v in clusters.items()}

    def twin_in_train(s, n):
        twin = n[1:] if n.startswith("M") else "M" + n
        return os.path.exists(f"{NAS}/data/diffusion/{s}/train/{twin}.json")
    free = [i for i, (s, n) in enumerate(names) if not twin_in_train(s, n)]
    free_clusters = {}
    for i in free:
        free_clusters.setdefault(scenes[i], []).append(i)
    free_clusters = {k: np.array(v) for k, v in free_clusters.items()}

    for title, cl in (("ALL held-out", clusters), (f"TWIN-FREE N={len(free)}", free_clusters)):
        print(f"\n== {title}, {len(cl)} scene clusters; delta>0 = first model better [95% CI] p")
        for a, b in PAIRS:
            if a not in data or b not in data:
                continue
            r = compare(data[a], data[b], cl)
            txt = "  ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] p={v[3]:.3f}" for k, v in r.items())
            print(f"{a:16s}vs {b:16s}{txt}")


if __name__ == "__main__":
    main()
