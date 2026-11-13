#!/usr/bin/env python3
"""Checks DATASET.EGOMOTION.EGO_WINDOW_FRAME on the real scene-split validation data (CPU, no model).

With the flag, the ego of a window cut from a longer track must sit in the frame of the decoded motion window.
Three checks: nothing but the ego of such windows changes, the new ego equals the independent computation of
coupling_breakdown.load_geometry (itself checked against the root dumps), and for random start frames the
distance between vehicle and pedestrian is the one of the recording.

Usage: python research/src/test_window_frame.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coupling_breakdown import CFG, load_geometry  # noqa: E402
from mld.data.humanml.scripts.motion_process import recover_root_rot_pos  # noqa: E402
from teaser_generate import build  # noqa: E402


def dataset(window_frame):
    _, datamodule, _ = build(dict(cfg=CFG, ckpt="unused", guidance=15, window_frame=window_frame), "scene_val")
    return datamodule.test_dataset, datamodule


def root_xz(raw_motion):
    return recover_root_rot_pos(torch.as_tensor(raw_motion[None], dtype=torch.float32))[1][0][:, [0, 2]].numpy()


def main():
    old, _ = dataset(False)
    new, datamodule = dataset(True)
    assert not old.ego_window_frame and new.ego_window_frame
    n = len(old)
    raw = lambda e: e * (old.ego_std + 1e-8) + old.ego_mean
    a, b = [old[i] for i in range(n)], [new[i] for i in range(n)]
    lens = np.array([int(x["length"]) for x in a])
    assert all(np.array_equal(x["motion"], y["motion"]) and x["length"] == y["length"] and x["ego_length"] == y["ego_length"]
               for x, y in zip(a, b))
    print(f"{n} validation rows: motion and lengths unchanged by the flag")

    paths = np.array(["/".join(Path(p).parts[-3:]) for p in old.sample_paths])
    motion = torch.as_tensor(np.stack([x["motion"] for x in a]))
    gt = datamodule.feats2joints(motion)[:, :, 0, [0, 2]].numpy()
    egos, offset, yaw, _ = load_geometry("scene_val", paths, lens, gt)
    moved = (offset > 0) | (yaw > 0)  # a pedestrian who stood still up to the window start can still have turned
    worst = 0.0
    for i in range(n):
        L = lens[i]
        if not moved[i]:  # window at the track start, or an identity transform up to rounding
            assert np.allclose(a[i]["ego"], b[i]["ego"], atol=1e-5), paths[i]
        assert np.array_equal(a[i]["ego"][L:], b[i]["ego"][L:]), paths[i]  # padding untouched
        worst = max(worst, np.abs(raw(b[i]["ego"][:L]) - egos[i]).max())
    assert worst < 2e-3, worst
    changed = sum(not np.allclose(x["ego"], y["ego"], atol=1e-5) for x, y in zip(a, b))
    print(f"ego changed in {changed} rows, all among the {moved.sum()} windows whose frame is shifted or turned; "
          f"largest difference to the window-frame ego of load_geometry {worst * 1000:.2f} mm")

    rng = np.random.default_rng(0)
    long = [p for p in new.sample_paths if len(new._load_sample(p)["motion"]) >= new.max_motion_length + 20][:40]
    for p in long:
        sample = new._load_sample(p)
        total, L = len(sample["motion"]), new.max_motion_length
        start = int(rng.integers(1, total - L + 1))
        cropped = new._normalize_ego(sample["ego"])[start:start + L]
        ego_w = raw(new._ego_in_window_frame(cropped, sample["ego"], sample["motion"], start, L))
        valid = min(L, len(sample["ego"]) - start)
        there = np.linalg.norm(sample["ego"][start:start + valid] - root_xz(sample["motion"])[start:start + valid], axis=1)
        here = np.linalg.norm(ego_w[:valid] - root_xz(sample["motion"][start:start + L])[:valid], axis=1)
        assert np.abs(there - here).max() < 5e-3, (p, start, np.abs(there - here).max())
    print(f"{len(long)} long tracks at random start frames: vehicle-pedestrian distance preserved")
    print("all checks passed")


if __name__ == "__main__":
    main()
