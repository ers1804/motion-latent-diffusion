#!/usr/bin/env python3
"""What the Ego-Coupling Score rests on (review of 2026-11-13, research/review_action_plan_2026-11-13.md).

Three questions, answered from the K=5 dumps of the selected checkpoints, CPU only:
  1. Which kind of stopping carries the score: pedestrians who stand throughout, or who move and stop?
  2. Does the score survive on conditions where the vehicle cannot have reacted to the pedestrian?
  3. How much does the initial geometry explain, i.e. the angle between the vehicle's direction of travel and
     the axis the pedestrian faces in the first frame?

Every constant below was fixed on the validation partition on 2026-11-13, before this script read the test
partition.

Frames. The dataset crops the ego trajectory by index, so for a windowed sequence the model sees the vehicle in
the frame of the track's first pose, while the motion window decodes to a path that starts at the origin facing
+z. All geometry here is computed in that window frame (load_geometry), and section 0 reports how far the two
frames are apart.

Usage: python research/src/coupling_breakdown.py --split scene_val
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_resplit import cell_of, load, metrics, scene_key  # noqa: E402

FPS = 20
STILL_RADIUS = 1.0  # m; a stopping pedestrian who never leaves this radius stands throughout
PATH_MARGINS = (2.5, 1.5, 4.0)  # m; the first is primary, the others show sensitivity
MIN_TRAVEL = 1.0  # m; below this the vehicle has no direction of travel
ANGLE_EDGES = (0, 10, 20, 40, 60, 90)  # degrees between direction of travel and facing axis
RULE_ANGLE = 10.0  # the reference rule: stops unless the vehicle travels within this angle of the facing axis
CELLS = {"EgoPed": "pooled", "Full-seq": "egoped", "prior": "uncond"}
CFG = "configs/config_ego_motion_new_vae_stoch.yaml"


def load_cells(data, selection, split):
    cells, ref = {}, None
    for run, s in sorted(selection.items()):
        if cell_of(run) not in CELLS.values():
            continue
        m = load(data / f"ade_fde_{run}_ep{s['epoch']}_cfg{s['guidance']}_{split}_k5.npz")
        ref = ref or m
        assert (m["paths"] == ref["paths"]).all() and (m["lens"] == ref["lens"]).all(), run
        assert max(np.abs(m["gt"][i, :n] - ref["gt"][i, :n]).max() for i, n in enumerate(ref["lens"])) < 1e-3, run
        cells.setdefault(cell_of(run), []).append(m)
    return cells, ref


def load_geometry(split, paths, lens, gt):
    """Per condition the vehicle path in the window frame of the dumps, plus the offset of the frame the model sees.

    Mirrors EgoMotionDataset.__getitem__: a track of at least 196 frames is cut at a start frame hashed from its
    path, a track shorter than the minimum length is replaced by the next sample. The pedestrian's root pose at the
    window start, decoded from the full track's features, takes the ego into the window frame. The recorded root
    path taken the same way must reproduce the dump, which checks start frame and conventions together.
    Returns (list of (m, 2) ego paths with m <= length, offset in metres, yaw offset in degrees, mask of replaced rows).
    """
    from mld.data.EgoMotion import _canonical_crop_key
    from mld.data.humanml.common.quaternion import qrot
    from mld.data.humanml.scripts.motion_process import recover_root_rot_pos
    from teaser_generate import build

    _, datamodule, _ = build(dict(cfg=CFG, ckpt="unused", guidance=15), split)
    ds = datamodule.test_dataset
    index = {"/".join(Path(q).parts[-3:]): i for i, q in enumerate(ds.sample_paths)}
    egos, offset, yaw, replaced = [], np.zeros(len(paths)), np.zeros(len(paths)), np.zeros(len(paths), dtype=bool)
    for row, (listed, n) in enumerate(zip(paths, lens)):
        idx = index[str(listed)]
        while True:
            sample = ds._load_sample(ds.sample_paths[idx])
            total = len(sample["motion"])
            if total >= ds.min_motion_length:
                break
            idx = (idx + 1) % len(ds)
        replaced[row] = idx != index[str(listed)]
        start = 0
        if total >= ds.max_motion_length:
            h = int(hashlib.sha1(_canonical_crop_key(ds.sample_paths[idx]).encode()).hexdigest()[:8], 16)
            start = int(np.random.default_rng(h).integers(0, total - ds.max_motion_length + 1))
        assert min(total, ds.max_motion_length) == n, (listed, total, n)
        quat, pos = recover_root_rot_pos(torch.as_tensor(sample["motion"])[None])
        quat, pos = quat[0], pos[0]

        def to_window(xz):  # (k, 2) in the track frame -> window frame
            p3 = torch.zeros(len(xz), 3)
            p3[:, [0, 2]] = torch.as_tensor(xz, dtype=torch.float32) - pos[start, [0, 2]]
            return qrot(quat[start].expand(len(xz), 4), p3)[:, [0, 2]].numpy()

        err = np.abs(to_window(pos[start:start + n, [0, 2]]) - gt[row, :n]).max()
        assert err < 0.02, (listed, start, err)
        egos.append(to_window(sample["ego"][start:start + n]))  # shorter than n if the ego record ends early
        offset[row] = float(np.linalg.norm(pos[start, [0, 2]]))
        facing = qrot(quat[start][None], torch.tensor([[0.0, 0.0, 1.0]]))[0]
        yaw[row] = float(np.degrees(np.arctan2(abs(facing[0]), facing[2])))
    return egos, offset, yaw, replaced


def travel_angle(ego, n):
    """Degrees between the vehicle's direction of travel and the pedestrian's facing axis (+z); NaN if it hardly moves."""
    first = min(n - 1, FPS)
    d = ego[first] - ego[0]
    if np.linalg.norm(d) < MIN_TRAVEL / 2:
        d = ego[n - 1] - ego[0]
    if np.linalg.norm(d) < MIN_TRAVEL:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(abs(d[1]) / np.linalg.norm(d), 0, 1))))


def path_clearance(ego, gt, n):
    """Closest the pedestrian comes to any position the vehicle occupies at that moment or later."""
    d = np.linalg.norm(gt[:n, None] - ego[None, :n], axis=2)  # [pedestrian time, vehicle time]
    return float(d[np.triu_indices(n)].min())


def ecs(p, pos, neg):
    return p[pos].mean() - p[neg].mean() if pos.any() and neg.any() else np.nan


def auc(score, pos, neg):
    return mannwhitneyu(score[pos], score[neg]).statistic / (pos.sum() * neg.sum())


class Bootstrap:
    """Scene-clustered bootstrap, as in compare_resplit.compare (2,000 draws)."""

    def __init__(self, paths, n=2000, seed=0):
        clusters = {}
        for i, p in enumerate(paths):
            clusters.setdefault(str(scene_key(str(p))), []).append(i)
        self.clusters = [np.array(clusters[k]) for k in sorted(clusters)]
        rng = np.random.default_rng(seed)
        self.draws = [np.concatenate([self.clusters[j] for j in rng.integers(0, len(self.clusters), len(self.clusters))])
                      for _ in range(n)]
        self.fold = np.empty(len(paths), dtype=int)  # five folds of whole scenes
        for k, j in enumerate(np.random.default_rng(seed).permutation(len(self.clusters))):
            self.fold[self.clusters[j]] = k % 5

    def ci(self, fn):
        v = np.array([fn(idx) for idx in self.draws])
        assert np.isfinite(v).mean() > 0.99, f"{(~np.isfinite(v)).sum()} undefined draws"
        return np.nanpercentile(v, [2.5, 97.5])

    def contrast(self, fn, runs_a, runs_b):
        """Difference of seed means of fn(run, idx): value, 95% interval, and whether all seed pairs agree in sign."""
        every = np.arange(len(self.fold))
        mean = lambda runs, idx: np.mean([fn(r, idx) for r in runs])
        delta = mean(runs_a, every) - mean(runs_b, every)
        low, high = self.ci(lambda idx: mean(runs_a, idx) - mean(runs_b, idx))
        pairs = [fn(a, every) - fn(b, every) for a in runs_a for b in runs_b]
        agree = all(x > 0 for x in pairs) or all(x < 0 for x in pairs)
        return delta, low, high, agree and (low > 0 or high < 0)


def fmt(result):
    delta, low, high, rule = result
    return f"{delta:+.3f} [{low:+.3f}, {high:+.3f}]{'*' if rule else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["scene_val", "scene_test"])
    ap.add_argument("--selection", default="research/data/resplit_selection.json")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()

    cells, ref = load_cells(Path(args.data), json.load(open(args.selection)), args.split)
    gt, lens, paths = ref["gt"], ref["lens"], ref["paths"]
    stop, n = ref["stop"].astype(bool), len(ref["gt"])
    walk, every = ~stop, np.arange(len(ref["gt"]))
    boot = Bootstrap(paths)
    runs = {name: cells[cell] for name, cell in CELLS.items()}
    p = {name: np.mean([r["p"] for r in rs], axis=0) for name, rs in runs.items()}
    for name, rs in runs.items():  # the dumps reproduce the reported numbers
        ref_ecs = np.mean([metrics(r, slice(None))["ECS"] for r in rs])
        assert abs(ecs(p[name], stop, walk) - ref_ecs) < 1e-9, name
    print(f"split={args.split} N={n} scenes={len(boot.clusters)} stopping rate={stop.mean():.3f}; "
          f"seeds: " + ", ".join(f"{k} {len(v)}" for k, v in runs.items()))
    print("* = 95% scene-bootstrap interval excludes 0 and all seed pairs agree in sign")

    # ---- 1. kinds of stopping
    reach = np.array([np.linalg.norm(gt[i, :lens[i]] - gt[i, 0], axis=1).max() for i in range(n)])
    still, moving = stop & (reach < STILL_RADIUS), stop & (reach >= STILL_RADIUS)
    print(f"\n1. Kinds of stopping (stands throughout = never leaves a {STILL_RADIUS} m radius)")
    print(f"   stands throughout {still.sum()} of {stop.sum()} stopping conditions ({still.sum() / stop.sum():.0%}), "
          f"moves and stops {moving.sum()}, walks {walk.sum()}")
    print(f"   {'model':9s}{'p|stands':>10s}{'p|moves+stops':>15s}{'p|walks':>9s}{'ECS':>8s}   ECS, moves and stops only")
    for name in CELLS:
        q = p[name]
        low, high = boot.ci(lambda idx: ecs(q[idx], moving[idx], walk[idx]))
        print(f"   {name:9s}{q[still].mean():10.3f}{q[moving].mean():15.3f}{q[walk].mean():9.3f}{ecs(q, stop, walk):8.3f}"
              f"   {ecs(q, moving, walk):+.3f} [{low:+.3f}, {high:+.3f}]")
    for name in ("EgoPed", "Full-seq"):
        result = boot.contrast(lambda r, idx: ecs(r["p"][idx], moving[idx], walk[idx]), runs[name], runs["prior"])
        print(f"   {name} minus prior, ECS on moves and stops: {fmt(result)}")

    # ---- 0. frame of the ego input
    ego, offset, yaw, replaced = load_geometry(args.split, paths, lens, gt)
    seen = np.array([len(e) for e in ego])  # frames with a vehicle position, at most the sequence length
    shifted = offset > 0
    print(f"\n0. Frame of the ego input: {shifted.sum()} of {n} conditions ({shifted.mean():.0%}) are windows that start "
          f"after the track's first frame.")
    print(f"   There the model sees the vehicle in the frame of the track's first pose: offset to the window frame "
          f"median {np.median(offset[shifted]):.1f} m (90th percentile {np.percentile(offset[shifted], 90):.1f} m), "
          f"yaw median {np.median(yaw[shifted]):.0f} deg (90th percentile {np.percentile(yaw[shifted], 90):.0f} deg).")
    print(f"   Among them recorded stop rate {stop[shifted].mean():.3f}, elsewhere {stop[~shifted].mean():.3f}. "
          f"Rows that load the next sample because their own is too short: {replaced.sum()}. "
          f"Vehicle record shorter than the sequence: {(seen < lens).sum()}.")
    for name in CELLS:
        q = p[name]
        print(f"   {name:9s} ECS on windows that start later {ecs(q, stop & shifted, walk & shifted):+.3f}, "
              f"on the others {ecs(q, stop & ~shifted, walk & ~shifted):+.3f}")

    print(f"   Without the {replaced.sum()} replaced rows (added after the first test run, no constant involved):")
    for name in CELLS:
        rows = [metrics(r, ~replaced) for r in runs[name]]
        full = [metrics(r, slice(None)) for r in runs[name]]
        print(f"   {name:9s} " + "  ".join(f"{k} {np.mean([m[k] for m in rows]):.3f} (all rows {np.mean([m[k] for m in full]):.3f})"
                                         for k in ("ADE", "minADE", "ECS", "Brier")))

    # ---- 2. conditions the vehicle cannot have reacted to
    clearance = np.array([path_clearance(ego[i], gt[i], seen[i]) for i in range(n)])
    approach = np.array([np.linalg.norm(gt[i, :seen[i]] - ego[i], axis=1).min() for i in range(n)])
    print(f"\n2. Pedestrian never within a margin of where the vehicle is or later drives "
          f"(median closest approach {np.median(approach):.1f} m)")
    for margin in PATH_MARGINS:
        clear = clearance >= margin
        print(f"   margin {margin} m: {clear.sum()} of {n} conditions ({clear.mean():.1%}), stopping rate {stop[clear].mean():.3f}")
        for name in CELLS:
            q = p[name]
            ade = np.mean([r["ade"] for r in runs[name]], axis=0)
            low, high = boot.ci(lambda idx: ecs(q[idx], (stop & clear)[idx], (walk & clear)[idx]))
            print(f"     {name:9s} ECS {ecs(q, stop & clear, walk & clear):+.3f} [{low:+.3f}, {high:+.3f}]  ADE {ade[clear].mean():.3f}"
                  f"   | rest ({(~clear).sum()}): ECS {ecs(q, stop & ~clear, walk & ~clear):+.3f}  ADE {ade[~clear].mean():.3f}")
        if margin == PATH_MARGINS[0]:
            for name in ("EgoPed", "Full-seq"):
                e = boot.contrast(lambda r, idx: ecs(r["p"][idx], (stop & clear)[idx], (walk & clear)[idx]), runs[name], runs["prior"])
                a = boot.contrast(lambda r, idx: -r["ade"][idx][clear[idx]].mean(), runs[name], runs["prior"])
                print(f"     {name} minus prior on these conditions: ECS {fmt(e)}, ADE gain {fmt(a)} m")

    # ---- 3. initial geometry
    angle = np.array([travel_angle(ego[i], seen[i]) for i in range(n)])
    moves = np.isfinite(angle)
    bins = np.where(moves, np.digitize(angle, ANGLE_EDGES[1:-1]), len(ANGLE_EDGES) - 1)
    names = [f"{a}-{b} deg" for a, b in zip(ANGLE_EDGES[:-1], ANGLE_EDGES[1:])] + ["hardly moves"]
    print("\n3. Angle between the vehicle's direction of travel and the pedestrian's facing axis")
    print(f"   {'vehicle':13s}{'n':>6s}{'recorded stop':>15s}{'EgoPed p':>10s}{'prior p':>9s}{'EgoPed ECS in bin':>19s}")
    for b, label in enumerate(names):
        m = bins == b
        print(f"   {label:13s}{m.sum():6d}{stop[m].mean():15.3f}{p['EgoPed'][m].mean():10.3f}{p['prior'][m].mean():9.3f}"
              f"{ecs(p['EgoPed'], stop & m, walk & m):+19.3f}")
    skipped = []

    def within(idx):  # bins without a stopping or a walking condition carry no score and are left out
        parts = [(ecs(p["EgoPed"][idx], (stop & (bins == b))[idx], (walk & (bins == b))[idx]), (bins[idx] == b).sum())
                 for b in range(len(names))]
        skipped.append(sum(np.isnan(e) for e, _ in parts))
        return sum(e * w for e, w in parts if not np.isnan(e)) / sum(w for e, w in parts if not np.isnan(e))

    value = within(every)
    low, high = boot.ci(within)
    print(f"   EgoPed ECS within bins, weighted by bin size: {value:+.3f} [{low:+.3f}, {high:+.3f}] (overall "
          f"{ecs(p['EgoPed'], stop, walk):+.3f}); bins left out: {skipped[0]} here, in {np.mean(np.array(skipped[1:]) > 0):.1%} of draws")

    rate = np.zeros(n)  # stopping rate of the angle bin, estimated on the other four folds of scenes
    for k in range(5):
        for b in range(len(names)):
            train = (boot.fold != k) & (bins == b)
            rate[(boot.fold == k) & (bins == b)] = stop[train].mean() if train.any() else stop[boot.fold != k].mean()
    rule = (~moves | (angle >= RULE_ANGLE)).astype(float)
    brier = lambda q: ((q - stop) ** 2).mean()
    print(f"\n   {'predictor of stopping':42s}{'ECS':>8s}{'AUC':>7s}{'Brier':>8s}{'stop rate':>11s}")
    for name in CELLS:
        rs = runs[name]
        print(f"   {name + ' samples (mean of seeds)':42s}{np.mean([ecs(r['p'], stop, walk) for r in rs]):+8.3f}"
              f"{np.mean([auc(r['p'], stop, walk) for r in rs]):7.3f}{np.mean([brier(r['p']) for r in rs]):8.3f}{p[name].mean():11.3f}")
    for label, q in ((f"rule: stops unless within {RULE_ANGLE:.0f} deg", rule), ("angle-bin stopping rate (out of fold)", rate),
                     ("constant stopping rate", np.full(n, stop.mean()))):
        a = auc(q, stop, walk) if q.std() > 0 else 0.5
        print(f"   {label:42s}{ecs(q, stop, walk):+8.3f}{a:7.3f}{brier(q):8.3f}{q.mean():11.3f}")
    print(f"   recorded stopping rate {stop.mean():.3f}")
    # Added after the first test run, no constant involved. A frequency over K samples carries binomial noise that a
    # stated probability does not: E[(p_hat - y)^2] = (pi - y)^2 + pi (1 - pi) / K, and p_hat (1 - p_hat) / (K - 1)
    # estimates the last term without bias. Only the corrected score compares with the rows that state a probability.
    k = runs["EgoPed"][0]["p"]
    K = int(round(1 / np.min(k[k > 0])))
    print(f"   Brier score of the samples with the sampling noise of K={K} removed: " + ", ".join(
        f"{name} {np.mean([brier(r['p']) - (r['p'] * (1 - r['p'])).mean() / (K - 1) for r in runs[name]]):.3f}" for name in CELLS))


if __name__ == "__main__":
    main()
