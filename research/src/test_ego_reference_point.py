#!/usr/bin/env python3
"""Checks of the reference point shift: the heading fit on made-up tracks, the edited input on the validation data.

The second part needs the dataset and research/data/ego_heading_scene_val.npz (CPU, no model).
Usage: python research/src/test_ego_reference_point.py [--no-data]
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ego_reference_point import (DATA, MOTION_HZ, SCENE_HZ, angle_between, fit_track, hold, load_headings,  # noqa: E402
                                 pose_heading, save_headings, travel_heading)
from teaser_map import resample  # noqa: E402


def frame(angle_deg, reflect):
    a = np.radians(angle_deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return r @ np.diag([1, -1]) if reflect else r


def stored(ego_world, ped_world, start_s, length, r):
    """The track a scene leaves in the frame that has the pedestrian's first position at the origin."""
    times = np.arange(len(ego_world)) / SCENE_HZ
    t = start_s + np.arange(length) / MOTION_HZ
    ego, ped = resample(t, times, ego_world) @ r.T, resample(t, times, ped_world) @ r.T
    origin = ped[0].copy()
    ped = ped - origin
    ped[MOTION_HZ:] = ped[MOTION_HZ - 1] + 1.05 * (ped[MOTION_HZ:] - ped[MOTION_HZ - 1])  # the stretch of the root path
    return ego - origin, ped


def test_fit():
    t = np.arange(200) / SCENE_HZ
    # a vehicle on a bend, a pedestrian who stands for 4 s and then walks
    ego_world = np.stack([6 * t, 12 * np.sin(t / 5)], 1) + [300, -80]
    tangent = np.stack([np.full_like(t, 6), 12 / 5 * np.cos(t / 5)], 1)
    forward = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
    ped_world = np.stack([340 + 0 * t, -70 + 1.4 * np.clip(t - 4, 0, None)], 1)
    for reflect in (False, True):
        r = frame(40, reflect)
        ego, ped = stored(ego_world, ped_world, 3.2, 150, r)
        fit = fit_track(ego_world, ped_world, ego, ped, reflect)
        assert abs(fit["start_s"] - 3.2) < 0.051 and fit["rms"] < 0.05, fit
        assert np.allclose(fit["R"], r, atol=5e-3) and fit["spread"] < 2, fit
        want = resample(3.2 + np.arange(150) / MOTION_HZ, t, forward) @ r.T
        assert angle_between(pose_heading(forward, fit, 150), want / np.linalg.norm(want, axis=1, keepdims=True)).max() < 1
        other = fit_track(ego_world, ped_world, ego, ped, not reflect)
        assert other["rms"] > 10 * max(fit["rms"], 0.05)  # a bend tells the handedness

    # both stand: two points fit in either handedness, and only the right one gives the heading
    ego_world = np.tile([10.0, 5.0], (200, 1))
    ped_world = np.tile([16.0, 7.0], (200, 1))
    forward = np.tile([0.0, 1.0], (200, 1))
    r = frame(-110, True)
    ego, ped = stored(ego_world, ped_world, 2.0, 120, r)
    right, wrong = (fit_track(ego_world, ped_world, ego, ped, h) for h in (True, False))
    assert right["rms"] < 1e-6 and wrong["rms"] < 1e-6
    want = np.tile(r @ [0.0, 1.0], (120, 1))
    assert angle_between(pose_heading(forward, right, 120), want).max() < 0.1
    assert angle_between(pose_heading(forward, wrong, 120), want).min() > 20

    # a pedestrian who is tracked only after 5 s leaves no earlier start
    late = ped_world.copy()
    late[:50] = np.nan
    ego, ped = stored(ego_world, ped_world, 6.0, 120, r)
    assert fit_track(ego_world, late, ego, ped, True)["start_s"] >= 5.0
    late[:] = np.nan
    try:
        fit_track(ego_world, late, ego, ped, True)
        raise AssertionError("an untracked pedestrian must not be fitted")
    except ValueError:
        pass
    print("fit: start, rotation and heading recovered in both handednesses, standing pair needs the handedness")


def test_travel():
    step = 2.0 / MOTION_HZ  # 2 m/s
    path = np.concatenate([np.zeros((40, 2)), np.stack([np.zeros(60), step * np.arange(1, 61)], 1)])
    h = travel_heading(path)
    assert np.isnan(h[:20]).all() and np.allclose(h[60:], [0, 1])
    held = hold(h)
    assert not np.isnan(held).any() and np.allclose(held, [0, 1])
    assert hold(travel_heading(np.zeros((50, 2)))) is None  # a vehicle that never moves has no direction of travel
    back = hold(travel_heading(path[::-1]))  # standing at the end: the heading it arrived with
    assert np.allclose(back, [0, -1])
    creep = np.stack([np.zeros(100), 0.2 / MOTION_HZ * np.arange(100)], 1)  # 0.2 m/s is under the threshold
    assert np.isnan(travel_heading(creep)).all()

    rows = [("a/b/1.json", np.tile([1.0, 0.0], (5, 1)), "pose"), ("a/b/2.json", None, "none"),
            ("a/b/3.json", np.tile([0.0, 1.0], (3, 1)), "path")]
    with tempfile.TemporaryDirectory() as tmp:
        save_headings(Path(tmp) / "h.npz", rows)
        table = load_headings(Path(tmp) / "h.npz")
    assert table["a/b/2.json"] is None and table["a/b/1.json"].shape == (5, 2) and np.allclose(table["a/b/3.json"], [0, 1])
    print("travel: direction of travel, standing phases held, table written and read")


def test_shift():
    from eval_ade_fde import shifted_egos
    from teaser_generate import build

    _, datamodule, _ = build(dict(cfg="configs/config_ego_motion_new_vae_stoch.yaml", ckpt="unused", guidance=15), "scene_val")
    ds = datamodule.test_dataset
    table = DATA / "ego_heading_scene_val.npz"
    heading = load_headings(table)
    raw = lambda e: e * (ds.ego_std + 1e-8) + ds.ego_mean
    n = len(ds)
    items = [ds[i] for i in range(n)]
    plain = np.stack([it["ego"] for it in items]).astype("float32")
    lens = [int(it["length"]) for it in items]
    assert np.array_equal(shifted_egos(ds, table, 0.0, ["nuscenes"]), plain)  # no shift: the dataset's own input
    assert "_load_sample" not in vars(ds)  # the loader of the class is back

    # the track a row shows: one under the minimum length is answered with the next one
    track_len = [len(ds._load_sample(p)["motion"]) for p in ds.sample_paths]

    def shown(i):
        while track_len[i] < ds.min_motion_length:
            i = (i + 1) % n
        return ds.sample_paths[i]

    key = lambda p: "/".join(Path(p).parts[-3:])
    for metres, sources in ((-0.94, ["nuscenes"]), (0.94, ["waymo"]), (-4.0, ["nuscenes", "waymo"])):
        out = shifted_egos(ds, table, metres, sources)
        moved = 0
        for i in range(n):
            L, path = lens[i], shown(i)
            assert np.array_equal(out[i, L:], plain[i, L:])  # padding untouched
            step = raw(out[i, :L]) - raw(plain[i, :L])
            if Path(path).parts[-3] in sources and heading[key(path)] is not None:
                assert np.allclose(np.linalg.norm(step, axis=1), abs(metres), atol=2e-3), (key(path), metres)
                v = raw(plain[i, :L])
                travel = travel_heading(v)
                seen = ~np.isnan(travel).any(1)
                if seen.sum() > 20:  # a moving vehicle is shifted along its path, ahead for a positive shift
                    cos = (step[seen] / abs(metres) * travel[seen]).sum(1) * np.sign(metres)
                    assert np.median(cos) > 0.95, (key(path), float(np.median(cos)))
                moved += 1
            else:
                assert np.array_equal(out[i, :L], plain[i, :L]), key(path)
        assert moved > 0.3 * n
        print(f"shift {metres:+.2f} m of {sources}: {moved} rows move by exactly that, {n - moved} are untouched")

    # a mirrored copy is shifted the mirrored way (the twin lies in the same folder, scene numbers are per folder)
    out = shifted_egos(ds, table, -0.94, ["nuscenes", "waymo"])
    index = {key(shown(i)): i for i in range(n)}
    gaps = []
    for k, i in index.items():
        source, folder, name = k.split("/")
        twin = index.get(f"{source}/{folder}/M{name}")
        if twin is None or source not in ("nuscenes", "waymo"):
            continue
        same_window = lens[i] == lens[twin] and np.abs(raw(plain[i, :lens[i]]) * [-1, 1] - raw(plain[twin, :lens[i]])).max() < 0.02
        if not same_window:  # a track longer than the window is cropped at another place for the copy
            continue
        gaps.append(np.abs(raw(out[i, :lens[i]]) * [-1, 1] - raw(out[twin, :lens[i]])).max())
    gaps = np.array(gaps)
    assert len(gaps) > 200 and np.median(gaps) < 0.005 and gaps.max() < 0.05, (len(gaps), np.median(gaps), gaps.max())
    print(f"mirror: {len(gaps)} pairs, shifted inputs mirror each other to {np.median(gaps):.3f} m (median), {gaps.max():.3f} m at most")


def test_standing():
    """The heading of a vehicle that never moves rests on the fitted rotation alone. Held-out check of that
    rotation: the pedestrian's path after its first second is not in the fit, and its displacement in the world,
    turned into the frame of the track, must point the way the stored path does (which is turned by a few degrees)."""
    import json

    from ego_reference_point import FORWARD, handedness_of, world_tracks
    from teaser_generate import build

    _, datamodule, _ = build(dict(cfg="configs/config_ego_motion_new_vae_stoch.yaml", ckpt="unused", guidance=15), "scene_val")
    table = load_headings(DATA / "ego_heading_scene_val.npz")
    for source in FORWARD:
        own = [p for p in datamodule.test_dataset.sample_paths if Path(p).parts[-3] == source]
        reflect, _ = handedness_of([p for p in own if not Path(p).name.startswith("M")][:40], {})
        off, kept = [], 0
        for path in own:
            data = json.load(open(path))
            ego = np.array(data["ego_in_ped_frame"], dtype=float)[:, [0, 2]]
            if not np.isnan(travel_heading(ego)).all():
                continue
            ped = np.array(data["ped_in_ped_frame"], dtype=float)[:, 0][:, [0, 2]]
            name = Path(path).name
            mirrored = name.startswith("M")
            scene, obj = Path(name).stem[1 if mirrored else 0:].split("_")
            ego_world, forward, ped_world = world_tracks(source, scene, obj)
            length = min(len(ego), len(ped))
            fit = fit_track(ego_world, ped_world, ego[:length], ped[:length], reflect != mirrored)
            heading = table["/".join(Path(path).parts[-3:])]
            assert angle_between(pose_heading(forward, fit, len(ego)), heading).max() < 0.5  # the table holds this fit
            assert np.allclose(np.linalg.norm(heading, axis=1), 1, atol=1e-4)
            kept += 1
            t = fit["start_s"] + np.arange(length) / MOTION_HZ
            walked = resample(t, np.arange(len(ego_world)) / SCENE_HZ, ped_world) @ fit["R"].T
            seen = np.flatnonzero(~np.isnan(walked).any(1))
            if len(seen) < 40 or np.linalg.norm(ped[seen[-1]] - ped[seen[0]]) < 3:
                continue
            a, b = walked[seen[-1]] - walked[seen[0]], ped[seen[-1]] - ped[seen[0]]
            off.append(float(angle_between(a / np.linalg.norm(a), b / np.linalg.norm(b))))
        assert kept >= 20 and len(off) >= 15 and np.median(off) < 8 and max(off) < 25, (source, kept, len(off), np.median(off), max(off))
        print(f"standing vehicles, {source}: {kept} tracks, rotation checked on {len(off)} walking pedestrians, "
              f"off by {np.median(off):.1f} deg (median), {max(off):.1f} deg at most")


if __name__ == "__main__":
    test_fit()
    test_travel()
    if "--no-data" not in sys.argv:
        test_shift()
        test_standing()
    print("all checks passed")
