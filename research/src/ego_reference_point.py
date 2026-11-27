#!/usr/bin/env python3
"""Does it matter which point of the vehicle the ego trajectory tracks? A check with frozen models on validation.

The processed nuScenes scenes store the pose of the roof lidar, 0.94 m ahead of the rear axle. The Waymo scenes
store the Waymo vehicle frame, whose origin lies near the rear axle (research/paper_ral/CHANGES_2026-11-26.md).
The check moves the stored ego path of one source along the vehicle's heading at inference and compares the
generated motion with the unedited run under the same noise.

  headings  The vehicle's heading in the frame of every stored track of a split. The stored path holds positions
            only, and a standing vehicle has no direction of travel, so the heading comes from the recorded pose.
            The scene's world frame is fitted to the stored track for that (fit_track).
  report    The shifted runs of eval_ade_fde.py --ego-shift against the unedited run of the same seed.

Usage:
  python research/src/ego_reference_point.py headings --split scene_val
  python research/src/eval_ade_fde.py --model <tag>_rp_nus_back094 ... --ego-shift -0.94 --ego-shift-source nuscenes
  python research/src/ego_reference_point.py report --split scene_val --runs nus_back094 nus_fwd094
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teaser_map import DRIVESTUDIO, FAR, MOTION_HZ, SCENE_HZ, resample, rigid  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
FORWARD = {"nuscenes": 1, "waymo": 0}  # column of the stored pose that points forward (nuScenes: the lidar frame)
LIDAR_AHEAD = 0.94  # metres from the rear axle to the nuScenes roof lidar
ANCHOR_S = 1.0  # seconds of the pedestrian's path that enter the fit, as in teaser_map.fit_window
MAX_RMS = 0.5  # metres, a worse fit means the wrong scene or the wrong part of it
MAX_SPREAD = 20.0  # degrees between the headings of all starts that fit about as well as the best one
MOVING = 0.5  # metres within one second, below that the path gives no direction of travel
SHIFT_PREFIX = "rp_"  # eval_ade_fde tags: <run tag>_rp_base, <run tag>_rp_<name>
SHIFTED = {"nus": ["nuscenes"], "way": ["waymo"], "all": ["nuscenes", "waymo"]}  # first part of a run name


# ---------------------------------------------------------------- heading of the vehicle in the stored track
def world_tracks(dataset, scene, obj):
    """Position and forward direction of the vehicle and position of one pedestrian in the scene's world frame."""
    root = Path(DRIVESTUDIO) / dataset / "drivestudio_output" / scene
    n = len(list((root / "ego_pose").glob("*.txt")))
    pose = np.array([np.loadtxt(root / "ego_pose" / f"{i:03d}.txt") for i in range(n)])
    forward = pose[:, :2, FORWARD[dataset]]
    ann = json.load(open(root / "instances" / "instances_info.json"))[obj]["frame_annotations"]
    ped = np.full((n, 2), np.nan)
    ped[ann["frame_idx"]] = np.array(ann["obj_to_world"])[:, :2, 3]
    return pose[:, :2, 3], forward / np.linalg.norm(forward, axis=1, keepdims=True), ped


def angle_between(a, b):
    """Degrees between unit vectors, row by row."""
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def fit_track(ego_world, ped_world, ego_track, ped_track, reflect, step=0.05):
    """Where a stored track starts in its scene and how the scene's world frame maps into the track's frame.

    The vehicle path enters whole, the pedestrian's path with its first ANCHOR_S seconds, because the motion
    representation stretches and turns the rest (teaser_map.fit_window). The handedness is given: a standing
    vehicle next to a standing pedestrian is two points, which fit mirrored as well as not.
    Returns start_s, R, rms, the rms of the best start over FAR s away and spread, the largest angle in degrees
    between the rotation of the best start and that of any start that fits about as well.
    """
    times = np.arange(len(ego_world)) / SCENE_HZ
    length = len(ego_track)
    anchor = min(length, int(round(ANCHOR_S * MOTION_HZ)))
    target = np.concatenate([ego_track, ped_track[:anchor]])
    fits = []
    for start in np.arange(0, times[-1] - (length - 1) / MOTION_HZ + 1e-9, step):
        t = start + np.arange(length) / MOTION_HZ
        ped = resample(t[:anchor], times, ped_world)
        if np.isnan(ped).any():  # the pedestrian must be tracked while it anchors the fit
            continue
        source = np.concatenate([resample(t, times, ego_world), ped])
        r, shift = rigid(source, target, reflect)
        rms = float(np.sqrt((np.linalg.norm(source @ r.T + shift - target, axis=1) ** 2).mean()))
        fits.append((float(start), rms, r))
    if not fits:
        raise ValueError("the pedestrian is not tracked at any start of a window of this length")
    start, rms, r = min(fits, key=lambda f: f[1])
    elsewhere = [f[1] for f in fits if abs(f[0] - start) > FAR]
    close = [f[2] for f in fits if f[1] <= 2 * max(rms, 0.05)]
    spread = max(float(angle_between(r[:, 0], other[:, 0])) for other in close)
    return dict(start_s=start, R=r, rms=rms, rms_elsewhere=min(elsewhere) if elsewhere else float("nan"), spread=spread)


def pose_heading(forward_world, fit, length):
    """(length, 2) unit vectors: where the vehicle points at every frame of the stored track, in the track's frame."""
    times = np.arange(len(forward_world)) / SCENE_HZ
    angle = np.unwrap(np.arctan2(forward_world[:, 1], forward_world[:, 0]))
    a = np.interp(fit["start_s"] + np.arange(length) / MOTION_HZ, times, angle)
    return np.stack([np.cos(a), np.sin(a)], 1) @ fit["R"].T


def travel_heading(path, half=MOTION_HZ // 2):
    """(T, 2) unit direction of travel over the second around every frame, NaN where the vehicle moved under MOVING."""
    path = np.asarray(path, dtype=float)
    i = np.arange(len(path))
    v = path[np.minimum(i + half, len(path) - 1)] - path[np.maximum(i - half, 0)]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n >= MOVING, v / np.maximum(n, 1e-9), np.nan)


def hold(heading):
    """Fill the NaN rows of a heading with the nearest row that has one. None if no row has one."""
    known = np.flatnonzero(~np.isnan(heading).any(1))
    if not len(known):
        return None
    nearest = known[np.abs(np.arange(len(heading))[:, None] - known[None]).argmin(1)]
    return heading[nearest]


def track_heading(path, reflect=None, cache=None):
    """Heading of the vehicle over one stored track and how it was obtained.

    Returns (heading or None, method, info). Methods: 'pose' from the recorded pose, 'path' from the direction of
    travel with standing phases held (the fit failed or disagrees with the travelled path), 'none' for a vehicle
    that never moves and has no trustworthy fit. reflect=None tries both handednesses and returns both fits in
    info, which is how the handedness of a source is found (handedness_of).
    """
    source, _, name = Path(path).parts[-3:]
    mirrored = name.startswith("M")
    scene, obj = Path(name).stem[1 if mirrored else 0:].split("_")
    data = json.load(open(path))
    ego = np.array(data["ego_in_ped_frame"], dtype=float)[:, [0, 2]]
    ped = np.array(data["ped_in_ped_frame"], dtype=float)[:, 0][:, [0, 2]]
    length = min(len(ego), len(ped))
    travel = travel_heading(ego)
    key = (source, scene, obj)
    if cache is None or key not in cache:
        tracks = world_tracks(source, scene, obj)
        if cache is not None:
            cache[key] = tracks
    ego_world, forward_world, ped_world = cache[key] if cache is not None else tracks
    if reflect is None:
        return None, "both", {r: fit_track(ego_world, ped_world, ego[:length], ped[:length], r) for r in (False, True)}

    info = dict(length=len(ego), moving=float((~np.isnan(travel).any(1)).mean()), rms=np.nan, spread=np.nan, off=np.nan)
    try:
        fit = fit_track(ego_world, ped_world, ego[:length], ped[:length], reflect != mirrored)
    except ValueError:
        fit = None
    if fit is not None:
        heading = pose_heading(forward_world, fit, len(ego))
        seen = ~np.isnan(travel).any(1)
        info.update(rms=fit["rms"], spread=fit["spread"], start_s=fit["start_s"],
                    off=float(np.median(angle_between(heading[seen], travel[seen]))) if seen.any() else np.nan)
        agrees = np.isnan(info["off"]) or info["off"] < MAX_SPREAD
        if fit["rms"] <= MAX_RMS and fit["spread"] <= MAX_SPREAD and agrees:
            return heading, "pose", info
    held = hold(travel)
    return held, "none" if held is None else "path", info


def handedness_of(paths, cache):
    """True if the world frame maps into the stored frame of an unmirrored track with a reflection.

    Decided by the tracks whose vehicle path bends enough to tell, which must all agree.
    """
    votes = []
    for path in paths:
        try:
            _, _, fits = track_heading(path, None, cache)
        except ValueError:
            continue
        best, other = sorted(fits, key=lambda r: fits[r]["rms"])
        if fits[best]["rms"] < 0.3 and fits[other]["rms"] > 3 * max(fits[best]["rms"], 0.05):
            votes.append(best)
    if not votes or len(set(votes)) != 1:
        raise ValueError(f"handedness not unanimous: {sum(votes)} of {len(votes)} tracks fit mirrored")
    return votes[0], len(votes)


def save_headings(out, rows):
    """rows: [(key, heading or None, method)]. One flat array with offsets, because the tracks differ in length."""
    lengths = [0 if h is None else len(h) for _, h, _ in rows]
    np.savez(out, paths=np.array([k for k, _, _ in rows]), method=np.array([m for _, _, m in rows]),
             offsets=np.concatenate([[0], np.cumsum(lengths)]),
             heading=np.concatenate([h for _, h, _ in rows if h is not None]).astype("float32"))


def load_headings(table):
    """{'source/folder/name.json': (T, 2) unit vectors, or None for a row without a heading}."""
    d = np.load(table)
    o = d["offsets"]
    return {str(p): (d["heading"][o[i]:o[i + 1]] if o[i + 1] > o[i] else None) for i, p in enumerate(d["paths"])}


def headings(args):
    from teaser_generate import build

    _, datamodule, _ = build(dict(cfg=args.cfg, ckpt="unused", guidance=1), args.split)
    paths = datamodule.test_dataset.sample_paths
    rows, summary = [], {}
    for source in FORWARD:
        own = [p for p in paths if Path(p).parts[-3] == source]
        cache = {}
        reflect, votes = handedness_of([p for p in own if not Path(p).name.startswith("M")], cache)
        print(f"{source}: {len(own)} tracks, unmirrored tracks map {'with' if reflect else 'without'} a reflection "
              f"({votes} tracks decide, all agree)")
        infos = []
        for p in own:
            heading, method, info = track_heading(p, reflect, cache)
            rows.append(("/".join(Path(p).parts[-3:]), heading, method))
            infos.append(dict(info, method=method))
        summary[source] = infos
        count = lambda m, cond=lambda i: True: sum(i["method"] == m and cond(i) for i in infos)
        stands = lambda i: i["moving"] == 0
        print(f"  heading from the pose {count('pose')} (vehicle never moves in {count('pose', stands)}), "
              f"from the path {count('path')}, none {count('none')}")
        for name, values in (("fit rms [m]", [i["rms"] for i in infos if i["method"] == "pose"]),
                             ("pose against travel [deg]", [i["off"] for i in infos if i["method"] == "pose"]),
                             ("ambiguity of the fit [deg]", [i["spread"] for i in infos if i["method"] == "pose"])):
            v = np.array(values, dtype=float)
            v = v[~np.isnan(v)]
            print(f"    {name:28s} median {np.median(v):.2f}  95% {np.percentile(v, 95):.2f}  max {v.max():.2f}  (n={len(v)})")
        for i, p in zip(infos, own):
            if i["method"] != "pose":
                print(f"    {i['method']}: {'/'.join(Path(p).parts[-3:])} rms {i['rms']:.2f} spread {i['spread']:.1f} "
                      f"off {i['off']:.1f} moving {i['moving']:.2f}")
    out = Path(args.out or DATA / f"ego_heading_{args.split}.npz")
    save_headings(out, rows)
    print("saved", out)


# ---------------------------------------------------------------- shifted runs against the unedited run
def shifted_sources(name):
    """'nus_back094' -> ['nuscenes']: the sources a shifted run moves, from the first part of its name."""
    return SHIFTED[name.split("_")[0]]


def report(args):
    from compare_resplit import cell_of, load, metrics, pick, scene_key
    from intention_labels import is_stopping

    selection = pick(json.load(open(args.selection)), "rs1")
    tags = {r: f"{r}_ep{s['epoch']}_cfg{s['guidance']}" for r, s in selection.items() if cell_of(r) == args.cell}
    runs = sorted(tags)
    dump = lambda run, name: Path(args.data) / f"ade_fde_{tags[run]}_{SHIFT_PREFIX}{name}_{args.split}_k5.npz"

    def read(run, name):
        m = load(dump(run, name))
        m["gen"] = np.load(dump(run, name))["roots_gen"]
        m["stops"] = np.array([[is_stopping(m["gen"][i, k, :m["lens"][i]]) for k in range(m["k"])] for i in range(len(m["lens"]))])
        return m

    base = {r: read(r, "base") for r in runs}
    paths = [str(p) for p in base[runs[0]]["paths"]]
    lens = base[runs[0]]["lens"]
    source = np.array([p.split("/")[0] for p in paths])
    heading = load_headings(args.headings)
    every = np.arange(len(paths))
    for m in base.values():
        assert list(m["paths"]) == paths and (m["lens"] == lens).all()

    print(f"{args.split}: {args.cell} seeds {runs}, N={len(paths)} "
          f"(nuscenes {(source == 'nuscenes').sum()}, waymo {(source == 'waymo').sum()}, ava {(source == 'ava').sum()})")
    print("\nunedited run on this machine against the stored dump of the cluster (same checkpoint, other noise)")
    for r in runs:
        f = Path(args.data) / f"ade_fde_{tags[r]}_{args.split}_k5.npz"
        if f.exists():
            a, b = metrics(load(f), every), metrics(base[r], every)
            print(f"  {r}: ADE {a['ADE']:.3f} / {b['ADE']:.3f}   ECS {a['ECS']:.3f} / {b['ECS']:.3f}   (stored / here)")
    noise = np.mean([np.mean([np.linalg.norm(m["gen"][i, x, :lens[i]] - m["gen"][i, y, :lens[i]], axis=-1).mean()
                              for i in every for x in range(m["k"]) for y in range(x + 1, m["k"])]) for m in base.values()])
    print(f"two samples of one row that differ in the noise only lie {noise:.2f} m apart on average")

    for name in args.runs:
        shifted = {r: read(r, name) for r in runs}
        for m in shifted.values():
            assert list(m["paths"]) == paths and (m["lens"] == lens).all(), name
        moves = np.array([s in shifted_sources(name) and heading.get(p) is not None for s, p in zip(source, paths)])
        rows, rest = np.flatnonzero(moves), np.flatnonzero(~moves)
        # rows of the other sources share batch and noise with the shifted rows: they must come out the same
        leak = max(np.abs(shifted[r]["gen"][rest] - base[r]["gen"][rest]).max() for r in runs) if len(rest) else 0.0
        print(f"\n=== {name}: {len(rows)} rows shifted, {len(rest)} rows not (their paths differ from the unedited run by at most {leak:.4f} m)")
        for label, idx in (("all rows", every), ("shifted rows", rows)):
            scenes = {}
            for i in idx:
                scenes.setdefault(scene_key(paths[i]), []).append(i)
            keys = sorted(scenes)
            rng = np.random.default_rng(0)
            draws = [np.concatenate([scenes[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(args.boot)]
            print(f"  {label} (N={len(idx)}, scenes={len(keys)})")
            for k in ("ADE", "minADE", "ECS", "BrierC"):
                b = [metrics(base[r], idx)[k] for r in runs]
                s = [metrics(shifted[r], idx)[k] for r in runs]
                per_seed = [y - x for x, y in zip(b, s)]
                boot = [np.mean([metrics(shifted[r], d)[k] - metrics(base[r], d)[k] for r in runs]) for d in draws]
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                print(f"    {k:7s} unedited {np.mean(b):.3f}  shifted {np.mean(s):.3f}  change {np.mean(per_seed):+.3f} "
                      f"[{lo:+.3f},{hi:+.3f}]  per seed {' '.join(f'{d:+.3f}' for d in per_seed)}  "
                      f"spread of the unedited seeds {max(b) - min(b):.3f}")
        far = np.concatenate([[np.linalg.norm(shifted[r]["gen"][i, :, :lens[i]] - base[r]["gen"][i, :, :lens[i]], axis=-1).mean()
                               for i in rows] for r in runs])
        flips = np.mean([(shifted[r]["stops"][rows] != base[r]["stops"][rows]).mean() for r in runs])
        print(f"  generated path under the same noise: {far.mean():.2f} m from the unedited one on average (median "
              f"{np.median(far):.2f}, 95% {np.percentile(far, 95):.2f}), stop or walk flips in {100 * flips:.1f}% of the samples")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("headings")
    h.add_argument("--split", default="scene_val", choices=["scene_val"])  # the test partition is not for this check
    h.add_argument("--cfg", default="configs/config_ego_motion_new_vae_stoch.yaml")
    h.add_argument("--out")
    r = sub.add_parser("report")
    r.add_argument("--split", default="scene_val", choices=["scene_val"])
    r.add_argument("--runs", nargs="+", required=True,
                   help="names of the shifted runs, e.g. nus_back094; the first part names the shifted sources (SHIFTED)")
    r.add_argument("--selection", default=str(DATA / "resplit_selection.json"))
    r.add_argument("--cell", default="pooled")
    r.add_argument("--data", default=str(DATA))
    r.add_argument("--headings", default=str(DATA / "ego_heading_scene_val.npz"))
    r.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    {"headings": headings, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
