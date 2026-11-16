#!/usr/bin/env python3
"""Known answers for munich_eval: no GPU, no labels. Usage: python research/src/test_munich_eval.py"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import munich_eval as me  # noqa: E402
from compare_resplit import scene_key  # noqa: E402
from mld.data.humanml.scripts.motion_process import recover_root_rot_pos  # noqa: E402


def line(start, velocity, n):
    return np.asarray(start, dtype=float) + np.outer(np.arange(n) / me.FPS, velocity)


def main():
    # Heading: a box stored at Rz(-90)(p - t) + t comes back to p, and its yaw gains 90 degrees.
    rng = np.random.default_rng(0)
    true, pivot = rng.normal(0, 20, (50, 2)), rng.normal(0, 20, (50, 2))
    d = true - pivot
    stored = pivot + np.stack([d[:, 1], -d[:, 0]], axis=1)
    back, yaw = me.correct_heading(stored, np.zeros(50), pivot)
    assert np.allclose(back, true) and np.allclose(yaw, np.pi / 2)

    # Frame: a pedestrian at the origin heading north (+y) has east on its right, so x (left) is -east, z is north.
    f = me.to_ped_frame(np.array([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]]), np.zeros(2), np.pi / 2)
    assert np.allclose(f, [[-1, 0], [0, 2], [3, 0]]), f
    # heading east: north is to the left
    assert np.allclose(me.to_ped_frame(np.array([[0.0, 1.0], [5.0, 0.0]]), np.zeros(2), 0.0), [[1, 0], [0, 5]])

    # Window: short tracks whole, long tracks centred on the closest approach and clamped to the track.
    ped = line([0, 0], [0, 0], 400)
    ego = line([-30, 2], [3, 0], 400)  # passes the pedestrian at t = 10 s, frame 200
    assert me.window_of(ped[:150], ego[:150]) == (0, 150)
    assert me.window_of(ped, ego) == (200 - 98, 196)
    assert me.window_of(ped, line([1, 2], [3, 0], 400)) == (0, 196)          # closest at the start
    assert me.window_of(ped, line([-200, 2], [3, 0], 400)) == (204, 196)     # closest at the end

    # Recorded stop: three consecutive keyframes inside the window, a gap or the window edge breaks the run.
    inside = np.ones(6, bool)
    assert me.standing_stop([0, 1, 2, 3, 4, 5], [0, 1, 1, 1, 0, 0], inside)
    assert not me.standing_stop([0, 1, 2, 3, 4, 5], [1, 1, 0, 1, 1, 0], inside)
    assert not me.standing_stop([0, 1, 3, 4, 6, 7], [1, 1, 1, 1, 1, 1], inside)  # never three in a row
    assert not me.standing_stop([0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 0, 0], np.array([0, 1, 1, 1, 1, 1], bool))

    # Heading from motion: a walker gets its walking direction, someone who hardly moves keeps the label.
    assert abs(me.motion_heading(line([0, 0], [0, 1.2], 100), 0.3) - np.pi / 2) < 1e-9
    assert me.motion_heading(line([0, 0], [0, 0.5], 100), 0.3) == 0.3   # 1.0 m in 2 s, under MOTION_MIN
    assert me.motion_heading(line([0, 0], [0, 1.2], 20), 0.3) == 0.3    # track shorter than it takes to cover 1.5 m

    # Kinds. The vehicle drives north along x = 0 at 5 m/s from y = -20, n frames.
    n = 160
    ego = line([0, -20], [0, 5], n)
    kind = lambda p, e=ego: me.kind_of(me.measures(p, e))  # noqa: E731
    # crosses the path at y = 0 at t = 2 s, the vehicle is there at t = 4 s
    assert kind(line([-3, 0], [1.5, 0], n)) == "front_moving"
    # crosses at y = 0 at t = 6 s, two seconds after the vehicle
    assert kind(line([-9, 0], [1.5, 0], n)) == "behind"
    # walks along the kerb 2 m beside the path
    assert kind(line([2, -20], [0, 1.2], n)) == "close_pass"
    # 10 m to the side
    assert kind(line([10, 0], [0, 1.2], n)) == "none"
    # the vehicle waits 4 s at y = -5 while the pedestrian crosses at y = 0, then drives on
    wait = np.concatenate([np.tile([0.0, -5.0], (80, 1)), line([0, -5], [0, 4], 80)])
    assert kind(line([-3, 0], [1.5, 0], n), wait) == "front_waiting"
    # the vehicle waits 7 s after the pedestrian has crossed: no conflict in time, still a spot it drives over
    long_wait = np.concatenate([np.tile([0.0, -5.0], (180, 1)), line([0, -5], [0, 4], 60)])
    m = me.measures(line([-3, 0], [1.5, 0], 240), long_wait)
    assert m["pet_front"] > me.PET_MAX and m["ego_speed_front"] < me.MOVING and me.kind_of(m) == "close_pass", m
    # a vehicle that does not drive has no path
    assert kind(line([-3, 0], [1.5, 0], n), np.tile([0.0, 0.0], (n, 1))) == "none"
    # leaves the path 6.5 s before the vehicle arrives: no conflict in time, still a spot the vehicle drives over
    m = me.measures(line([0, -25], [3, 0], n), line([0, -60], [0, 5], n))
    assert m["pet_front"] > me.PET_MAX and m["pet_behind"] is None and me.kind_of(m) == "close_pass", m

    # Features: the root channels decode to the path, through normalisation with the training statistics.
    path = np.cumsum(rng.normal(0, 0.05, (120, 2)), axis=0)
    path -= path[0]
    feats = me.encode_root(path)
    mean, std = np.load(me.STATS / "Mean.npy"), np.load(me.STATS / "Std.npy")
    normalised = (feats - mean) / (std + 1e-8)
    restored = torch.as_tensor(normalised * std + mean, dtype=torch.float32)[None]
    _, pos = recover_root_rot_pos(restored)
    assert np.abs(pos[0][:, [0, 2]].numpy() - path).max() < 2e-3, np.abs(pos[0][:, [0, 2]].numpy() - path).max()

    # Files: what materialize writes is what the loader reads, and a changed set is refused.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        names = ["U003_1", "U003_7", "U010_0"]
        ped, ego2 = np.zeros((3, me.MAX_LEN, 2), np.float32), np.zeros((3, me.MAX_LEN, 2), np.float32)
        length = np.array([120, 196, 40])
        for i, L in enumerate(length):
            ped[i, :L], ego2[i, :L] = line([0, 0], [0.1 * i, 1.0], L), line([5, -10], [0, 4], L)
        for v in me.VARIANTS:
            np.savez(tmp / f"sequences_{v}.npz", names=np.array(names), ped=ped, ego=ego2, length=length)
        h = me.content_hash(names, ped, ego2, length)
        (tmp / "manifest.json").write_text(json.dumps(dict(content_sha256={v: h for v in me.VARIANTS})))
        me.materialize(tmp / "out", folder=tmp)
        j = json.loads((tmp / "out/motion/data/diffusion/munich/val/003_7.json").read_text())
        assert np.array(j["vectors_263"]).shape == (196, 263) and np.array(j["ped_in_ped_frame"]).shape == (196, 22, 3)
        assert np.allclose(np.array(j["ego_in_ped_frame"])[:, [0, 2]], ego2[1], atol=1e-5)
        assert np.allclose(np.array(j["ped_in_ped_frame"])[:, 0][:, [0, 2]], ped[1], atol=1e-5)
        assert (tmp / "out/label/split/val.txt").read_text().split() == names
        assert all((tmp / "out/label/split" / f).read_bytes() == (me.STATS / f).read_bytes() for f in me.STAT_FILES)
        ped[0, 0, 0] += 1
        np.savez(tmp / "sequences_label.npz", names=np.array(names), ped=ped, ego=ego2, length=length)
        try:
            me.load_sequences("label", tmp)
            raise AssertionError("a changed set must be refused")
        except ValueError:
            pass

    # Scene clustering: Munich rows cluster by their scene index, apart from the other sources.
    assert scene_key("munich/val/017_5.json") == scene_key("munich/val/017_12.json") == ("U", "017")
    assert scene_key("munich/val/017_5.json") != scene_key("nuscenes/val/017_5.json")
    print("ok")


if __name__ == "__main__":
    main()
