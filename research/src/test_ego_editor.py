#!/usr/bin/env python3
"""Checks the ego controls and the body statistics of eval_ade_fde on the real scene-split validation data (CPU, no model).

Usage: python research/src/test_ego_editor.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ade_fde import FPS, body_stats, ego_editor  # noqa: E402
from make_scene_split import scene_of  # noqa: E402
from teaser_generate import build  # noqa: E402

LETTER = {"ava": "A", "nuscenes": "N", "waymo": "W"}


def main():
    spec = dict(cfg="configs/config_ego_motion_new_vae_stoch.yaml", ckpt="unused", guidance=15)
    _, datamodule, _ = build(spec, "scene_val")
    ds = datamodule.test_dataset
    n = 64
    items = [ds[i] for i in range(n)]
    ego = torch.as_tensor(np.stack([it["ego"] for it in items]))
    lens = [int(it["length"]) for it in items]
    scenes = [scene_of(LETTER[Path(p).parts[-3]] + Path(p).stem) for p in ds.sample_paths]

    swap = ego_editor(SimpleNamespace(ego_swap=True, ego_horizon=None), ds)
    out = swap(ego, 0, lens).numpy()
    for i in range(n):
        j = [k for k in range(len(ds)) if np.array_equal(out[i], np.asarray(ds[k]["ego"]))] if i < 4 else None
        assert not np.array_equal(out[i], ego[i].numpy())
        if j:
            assert all(scenes[k] != scenes[i] for k in j), (i, j)
    print("swap: every row gets another scene's ego")

    raw = lambda e: e * (ds.ego_std + 1e-8) + ds.ego_mean
    for horizon in (0, 1, 2, 4, 20):
        cut = int(horizon * FPS)
        out = ego_editor(SimpleNamespace(ego_swap=False, ego_horizon=horizon), ds)(ego, 0, lens).numpy()
        for i in range(n):
            L = lens[i]
            r0, r1 = raw(ego[i, :L].numpy()), raw(out[i, :L])
            assert np.allclose(out[i, L:], ego[i, L:].numpy())  # padding untouched
            if cut >= L:
                assert np.allclose(r1, r0, atol=1e-4)
                continue
            assert np.allclose(r1[:max(cut, 1)], r0[:max(cut, 1)] if cut else r0[:1], atol=1e-4)
            step = np.diff(r1[max(cut - 1, 0):], axis=0)
            assert np.allclose(step, step[0], atol=1e-4)  # constant velocity after the cut
        print(f"horizon {horizon} s: known part kept, constant velocity after")

    # braking after the first second: same direction as the constant-velocity control, speed falls to zero
    A, cut = 2.0, FPS
    keep = ego_editor(SimpleNamespace(ego_swap=False, ego_horizon=1, ego_brake=None), ds)(ego, 0, lens).numpy()
    slow = ego_editor(SimpleNamespace(ego_swap=False, ego_horizon=1, ego_brake=A), ds)(ego, 0, lens).numpy()
    checked = 0
    for i in range(n):
        L = lens[i]
        assert np.allclose(slow[i, L:], ego[i, L:].numpy())
        if cut >= L:
            assert np.allclose(slow[i, :L], ego[i, :L].numpy(), atol=1e-5)
            continue
        k, b = raw(keep[i, :L]), raw(slow[i, :L])
        assert np.allclose(b[:cut], k[:cut], atol=1e-4)  # the shown second is identical
        v0 = np.linalg.norm(k[cut] - k[cut - 1]) * FPS
        gone_keep, gone_slow = np.linalg.norm(k[cut:] - k[cut - 1], axis=1), np.linalg.norm(b[cut:] - b[cut - 1], axis=1)
        t = np.arange(1, L - cut + 1) / FPS
        assert np.allclose(gone_keep, v0 * t, atol=1e-3)
        assert np.allclose(gone_slow, np.where(t < v0 / A, v0 * t - A * t ** 2 / 2, v0 ** 2 / (2 * A)), atol=1e-3)
        assert (gone_slow <= gone_keep + 1e-4).all() and (np.diff(gone_slow) >= -1e-4).all()
        if v0 > 0.5:  # same straight line
            d_keep, d_slow = k[-1] - k[cut - 1], b[-1] - b[cut - 1]
            assert abs(d_keep[0] * d_slow[1] - d_keep[1] * d_slow[0]) < 1e-2 * np.linalg.norm(d_keep) * max(np.linalg.norm(d_slow), 1e-6) + 1e-3
            if t[-1] > v0 / A:
                assert np.allclose(b[-1], b[-2], atol=1e-4)  # stands at the end
            checked += 1
    print(f"braking at {A} m/s^2 after 1 s: shown second kept, same line, stops ({checked} moving vehicles checked)")

    # body statistics on synthetic joints: a rigidly sliding body and a body with one planted foot
    T = 60
    slide = np.zeros((T, 22, 3)); slide[:, :, 2] += np.linspace(0, 3, T)[:, None]  # 1 m/s, all joints together
    assert np.allclose(body_stats(slide)[:2], (1.0, 0.0), atol=1e-6) and body_stats(slide)[2] < 1e-9
    plant = slide.copy(); plant[:, 10] = 0.0  # left toe never moves
    ratio, planted, articulation = body_stats(plant)
    assert ratio < 1e-6 and planted == 1.0 and articulation > 0
    assert np.isnan(body_stats(np.zeros((T, 22, 3)))[0])  # standing: no walking frames to judge
    print("body statistics: sliding body 1.0, planted foot 0.0, standing undefined")
    print("all checks passed")


if __name__ == "__main__":
    main()
