#!/usr/bin/env python3
"""Loads the scene split through the real dataset class and checks what training will receive."""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "src"))
from make_scene_split import scene_of, twin_of  # noqa: E402
from mld.data.EgoMotion import EgoMotionDataset, _canonical_crop_key  # noqa: E402

NAS = os.environ.get("EGOPED_NAS", "/home/erik/NAS/methods/diffusion_gen")
SPLIT = ROOT / "research" / "data" / "scene_split_v1"
EXPECTED = {"train": 9410, "val": 1194, "test": 1182}
LETTER = {"ava": "A", "nuscenes": "N", "waymo": "W"}


def listed_name(path):
    parts = Path(path).parts
    return LETTER[parts[-3]] + Path(path).stem


def main():
    roots = [f"{NAS}/data/diffusion/{s}" for s in ("ava", "nuscenes", "waymo")]
    seen = {}
    for split, n in EXPECTED.items():
        ds = EgoMotionDataset(data_root=roots, split=split, split_list_root=str(SPLIT),
                              mean=np.load(SPLIT / "Mean.npy"), std=np.load(SPLIT / "Std.npy"),
                              ego_mean=np.load(SPLIT / "Ego_Mean.npy"), ego_std=np.load(SPLIT / "Ego_Std.npy"),
                              scene_split=True, deterministic_crop=True, interaction_crop=True)
        assert len(ds) == n, (split, len(ds))
        assert len(set(ds.sample_paths)) == n, split
        assert ds.deterministic_crop == (split != "train"), split
        assert ds.interaction_crop == (split == "train"), split
        names = {listed_name(p) for p in ds.sample_paths}
        assert names == {l.strip() for l in open(SPLIT / f"{split}.txt") if l.strip()}, split
        assert all(twin_of(x) in names for x in names), split
        seen[split] = {scene_of(x) for x in names}
    assert not seen["train"] & seen["val"] and not seen["train"] & seen["test"] and not seen["val"] & seen["test"]

    a = _canonical_crop_key("/tmp/job1/data/diffusion/nuscenes/val/1_2.json")
    b = _canonical_crop_key(f"{NAS}/data/diffusion/nuscenes/val/1_2.json")
    assert a == b == "/home/erik/NAS/methods/diffusion_gen/data/diffusion/nuscenes/val/1_2.json"
    print("scene split OK", {k: len(v) for k, v in seen.items()})


if __name__ == "__main__":
    main()
