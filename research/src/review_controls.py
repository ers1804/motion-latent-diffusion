#!/usr/bin/env python3
"""Three controls asked for by the pre-submission review (research/review_action_plan_2026-11-13.md, C2 to C4).

1. Braking. Both arms show the model the first second of the recorded ego trajectory and continue in the same
   direction. "Keeps going" holds the speed (hor1), "brakes" decelerates at 2 m/s^2 until the vehicle stands
   (hor1brake2). Reported is the shift in the generated stop probability and in the distance the generated root
   covers, for all conditions, for vehicles faster than 2 m/s at the end of the first second, and for the slower
   rest, where the two inputs barely differ and the shift should vanish.
2. Body statistics of generated motion against the labels of the same windows (foot-skate ratio, planted
   fraction, articulation, written by eval_ade_fde.py --save-skate).
3. The deterministic ego-to-motion regressor against EgoPed and the prior, with the per-condition metrics. It
   gives one motion per condition, so its stop prediction is 0 or 1 and its Brier score has no sampling noise.

Rule as everywhere (compare_resplit): a difference counts only if the 95% scene bootstrap interval excludes 0
and all seed pairs agree in sign, marked with *. Thresholds and outputs were fixed on validation and committed
before the test dumps were opened.

Usage: python research/src/review_controls.py --split scene_val [--skip braking body regressor]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from compare_resplit import beats, cell_of, compare, load, metrics, scene_key  # noqa: E402
from coupling_breakdown import Bootstrap, auc, fmt, load_geometry  # noqa: E402

FPS = 20
HORIZON = 1.0     # s of recorded ego trajectory both braking arms see (eval_review.sh)
FAST = 2.0        # m/s, "the vehicle moves" at the end of the first second
BODY = ("foot-skate ratio", "planted fraction", "articulation (m)")


def load_run(path):
    """compare_resplit.load plus the net distance the generated root covers, averaged over the samples."""
    m = load(path)
    gen = np.load(path)["roots_gen"]
    m["travel"] = np.array([np.linalg.norm(gen[i, :, n - 1] - gen[i, :, 0], axis=-1).mean() for i, n in enumerate(m["lens"])])
    return m


def same_conditions(m, ref, name):
    assert (m["paths"] == ref["paths"]).all() and (m["lens"] == ref["lens"]).all(), name
    assert max(np.abs(m["gt"][i, :n] - ref["gt"][i, :n]).max() for i, n in enumerate(ref["lens"])) < 1e-3, name


def speed_at_horizon(egos):
    """Vehicle speed (m/s) the editor continues with: mean velocity of the last 0.5 s before the horizon.

    Mirrors eval_ade_fde.ego_editor. A rigid change of frame leaves it unchanged, so the window-frame ego of
    load_geometry serves. Rows whose ego record ends within the horizon are not edited and count as standing.
    """
    cut = int(round(HORIZON * FPS))
    first = max(cut - 1 - FPS // 2, 0)
    return np.array([np.linalg.norm(e[cut - 1] - e[first]) / (cut - 1 - first) * FPS if len(e) > cut else 0.0 for e in egos])


def braking(args, tags, runs, full, boot):
    keep = [load_run(args.data / f"ade_fde_{tags[r]}_hor1_{args.split}_k5.npz") for r in runs]
    brake = [load_run(args.data / f"ade_fde_{tags[r]}_hor1brake2_{args.split}_k5.npz") for r in runs]
    for m, r in zip(keep + brake, runs + runs):
        same_conditions(m, full[0], r)
    egos = load_geometry(args.split, full[0]["paths"], full[0]["lens"], full[0]["gt"])[0]
    speed = speed_at_horizon(egos)
    subsets = [("all conditions", np.ones(len(speed), dtype=bool)),
               (f"vehicle faster than {FAST:g} m/s", speed > FAST),
               (f"vehicle at most {FAST:g} m/s", speed <= FAST)]
    print(f"\n1. Braking after the first second, {HORIZON:g} s shown, then 2 m/s^2 against constant speed")
    print(f"   vehicle speed at {HORIZON:g} s: median {np.median(speed):.1f} m/s, faster than {FAST:g} m/s in "
          f"{(speed > FAST).mean():.1%} of the conditions (median {np.median(speed[speed > FAST]):.1f} m/s there)")
    print(f"   recorded stop rate {full[0]['stop'].mean():.3f}, generated with the full input "
          f"{np.mean([m['p'].mean() for m in full]):.3f}")
    for key, unit in (("p", "stop probability"), ("travel", "distance covered by the root (m)")):
        print(f"\n   {unit}")
        print(f"   {'subset':30s}{'n':>6s}{'keeps going':>13s}{'brakes':>9s}   brakes minus keeps going [95% CI]")
        for name, mask in subsets:
            fn = lambda m, idx, mask=mask, key=key: m[key][idx[mask[idx]]].mean()
            every = np.arange(len(mask))
            print(f"   {name:30s}{mask.sum():6d}{np.mean([fn(m, every) for m in keep]):13.3f}"
                  f"{np.mean([fn(m, every) for m in brake]):9.3f}   {fmt(boot.contrast(fn, brake, keep))}")
    fast = speed > FAST
    for name, pick in (("recorded pedestrian stops", full[0]["stop"] == 1), ("recorded pedestrian walks", full[0]["stop"] == 0)):
        mask = fast & pick
        fn = lambda m, idx, mask=mask: m["p"][idx[mask[idx]]].mean()
        print(f"   stop probability, vehicle faster than {FAST:g} m/s and {name} (n={mask.sum()}): "
              f"{fmt(boot.contrast(fn, brake, keep))}")


def body(args, tags, runs, full, boot):
    dumps = [np.load(args.data / f"ade_fde_{tags[r]}_body_{args.split}_k5.npz") for r in runs]
    for d, r in zip(dumps, runs):
        assert (d["paths"] == full[0]["paths"]).all(), r
        assert np.allclose(d["body_gt"], dumps[0]["body_gt"], atol=1e-4, equal_nan=True), r
    label = dumps[0]["body_gt"]                                        # (N, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                # conditions without a walking sample
        gen = [np.nanmean(d["body_gen"], axis=1) for d in dumps]      # (N, 3) per seed
    print("\n2. Body statistics, generated motion against the labels of the same windows")
    print(f"   windows with at least 10 walking frames: labels {np.isfinite(label[:, 0]).mean():.1%}, generated samples "
          f"{np.mean([np.isfinite(d['body_gen'][..., 0]).mean() for d in dumps]):.1%}")
    print(f"   {'statistic':20s}{'labels':>9s}{'generated':>11s}{'per seed':>24s}   generated minus labels, same windows [95% CI]")
    for c, name in enumerate(BODY):
        per_seed = [np.nanmean(d["body_gen"][..., c]) for d in dumps]
        both = [np.isfinite(g[:, c]) & np.isfinite(label[:, c]) for g in gen]
        diff = lambda k, idx: (gen[k][idx, c] - label[idx, c])[both[k][idx]].mean()
        every = np.arange(len(label))
        deltas = [diff(k, every) for k in range(len(gen))]
        low, high = boot.ci(lambda idx: np.mean([diff(k, idx) for k in range(len(gen))]))
        rule = (all(x > 0 for x in deltas) or all(x < 0 for x in deltas)) and (low > 0 or high < 0)
        print(f"   {name:20s}{np.nanmean(label[:, c]):9.3f}{np.mean(per_seed):11.3f}"
              f"{'(' + ', '.join(f'{x:.3f}' for x in per_seed) + ')':>24s}   {fmt((np.mean(deltas), low, high, rule))}")
    print(f"   articulation, generated over labels: {np.mean([np.nanmean(d['body_gen'][..., 2]) for d in dumps]) / np.nanmean(label[:, 2]):.2f}")
    # Added after the test read, to quote the generated motion in the terms of the label audit (pose_quality_audit.py
    # and Section IV-C report the median over sequences). The means and the paired differences above were fixed before.
    print("   medians over windows, labels / generated samples: " + ", ".join(
        f"{name} {np.nanmedian(label[:, c]):.3f} / {np.mean([np.nanmedian(d['body_gen'][..., c]) for d in dumps]):.3f}"
        for c, name in enumerate(BODY)))


def regressor(args, tags, full, cells, boot):
    files = sorted(args.data.glob(f"ade_fde_rs1_regressor_s*_{args.split}_k1.npz"))
    print(f"\n3. Deterministic regressor ({len(files)} seeds) against the diffusion models")
    if not files:
        print("   no regressor dumps found")
        return
    reg = [load(f) for f in files]
    for m, f in zip(reg, files):
        same_conditions(m, full[0], f.name)
        assert m["k"] == 1, f.name
    cells = dict(cells, regressor=reg)
    every = np.arange(len(full[0]["paths"]))
    stop, walk = full[0]["stop"] == 1, full[0]["stop"] == 0
    print(f"   {'model':12s}{'seeds':>6s}{'ADE':>8s}{'minADE':>8s}{'ECS':>8s}{'AUC':>7s}{'Brier':>8s}{'BrierC':>8s}{'gen stop':>10s}")
    for name in ("regressor", "pooled", "uncond"):
        rows = [metrics(m, every) for m in cells[name]]
        mean = lambda k: np.mean([r[k] for r in rows])
        print(f"   {name:12s}{len(rows):6d}{mean('ADE'):8.3f}{mean('minADE'):8.3f}{mean('ECS'):8.3f}"
              f"{np.mean([auc(m['p'], stop, walk) for m in cells[name]]):7.3f}{mean('Brier'):8.3f}{mean('BrierC'):8.3f}"
              f"{np.mean([m['p'].mean() for m in cells[name]]):10.3f}")
    for f in files:
        j = args.data / f"{f.name.removeprefix('ade_fde_').split('_scene_')[0]}_metrics_{args.split}.json"
        if j.exists():
            m = json.load(open(j))
            print(f"   {j.name}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, float))[:150])
    print("\n   delta>0 = regressor better [95% CI] (seed pair range); * = rule met")
    for other in ("pooled", "uncond"):
        r = compare(reg, cells[other], boot_clusters(boot))
        print(f"   regressor vs {other:8s}" + "  ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] ({v[4]:+.3f},{v[5]:+.3f})"
                                                 f"{'*' if beats(v) else ' '}" for k, v in r.items() if k in ("ADE", "minADE", "ECS", "BrierC")))


def boot_clusters(boot):
    return {i: c for i, c in enumerate(boot.clusters)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["scene_val", "scene_test"])
    ap.add_argument("--selection", default="research/data/resplit_selection.json")
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--skip", nargs="*", default=[], choices=["braking", "body", "regressor"])
    args = ap.parse_args()

    selection = json.load(open(args.selection))
    tags = {run: f"{run}_ep{s['epoch']}_cfg{s['guidance']}" for run, s in selection.items()}
    runs = sorted(r for r in tags if cell_of(r) == "pooled")
    cells = {c: [load(args.data / f"ade_fde_{tags[r]}_{args.split}_k5.npz") for r in sorted(tags) if cell_of(r) == c]
             for c in ("pooled", "uncond")}
    full = cells["pooled"]
    for m in cells["pooled"] + cells["uncond"]:
        same_conditions(m, full[0], "full input")
    boot = Bootstrap(full[0]["paths"])
    print(f"{args.split}: EgoPed seeds {runs}; N={len(full[0]['paths'])} scenes={len(boot.clusters)}")
    if "braking" not in args.skip:
        braking(args, tags, runs, full, boot)
    if "body" not in args.skip:
        body(args, tags, runs, full, boot)
    if "regressor" not in args.skip:
        regressor(args, tags, full, cells, boot)


if __name__ == "__main__":
    main()
