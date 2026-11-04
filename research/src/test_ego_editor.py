#!/usr/bin/env python3
"""Checks the two ego controls of eval_ade_fde on the real scene-split validation data (CPU, no model).

Usage: python research/src/test_ego_editor.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ade_fde import FPS, ego_editor  # noqa: E402
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
    print("all checks passed")


if __name__ == "__main__":
    main()
