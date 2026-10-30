#!/usr/bin/env python3
"""Build the scene-level split used for all retrained models.

The previous split was made per file: mirrored twins (M*.json) and other pedestrians of one scene
landed on both sides. Here a scene and everything derived from it stays in one partition.

Candidate pool: the 11,786 names of the existing list, resolved by source prefix only.
Output (research/data/scene_split_v1):
  train.txt, val.txt, test.txt   val selects checkpoints, test is touched only for reporting
  Mean.npy, Std.npy, Ego_Mean.npy, Ego_Std.npy   computed by --stats on the training partition
  split_manifest.json

Usage: python research/src/make_scene_split.py [--stats]
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

NAS = os.environ.get("EGOPED_NAS", "/home/erik/NAS/methods/diffusion_gen")
DATA = f"{NAS}/data/diffusion"
OLD_LIST = f"{NAS}/data/vae/mean_std_txt/ava_nuscenes_waymo/train.txt"
OUT = Path(__file__).resolve().parent.parent / "data" / "scene_split_v1"
SOURCES = {"A": "ava", "N": "nuscenes", "W": "waymo"}
SEED = 20261030
FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
AVA_SCENES = {"train": 11, "val": 3, "test": 3}  # 17 staged scenes; both held-out sets need some


def scene_of(name):
    """N/W/A + optional M + scene + _ + object -> (source letter, scene id)."""
    return name[0], name[1:].lstrip("M").split("_")[0]


def twin_of(name):
    body = name[1:]
    return name[0] + (body[1:] if body.startswith("M") else "M" + body)


def assign(scenes, rng, targets):
    """Greedy: shuffled scenes go to the partition furthest below its share of files."""
    keys = sorted(scenes)
    rng.shuffle(keys)
    total = sum(len(scenes[k]) for k in keys)
    filled = Counter()
    out = {}
    for k in keys:
        part = max(targets, key=lambda p: targets[p] * total - filled[p])
        out[k] = part
        filled[part] += len(scenes[k])
    return out


def build():
    names = [l.strip() for l in open(OLD_LIST) if l.strip()]
    assert len(names) == len(set(names))
    by_scene = defaultdict(lambda: defaultdict(list))
    for n in names:
        by_scene[n[0]][scene_of(n)].append(n)
    rng = random.Random(SEED)
    parts = defaultdict(list)
    for src in sorted(by_scene):
        scenes = by_scene[src]
        if src == "A":
            keys = sorted(scenes)
            rng.shuffle(keys)
            cut = [("train", AVA_SCENES["train"]), ("val", AVA_SCENES["val"]), ("test", AVA_SCENES["test"])]
            i = 0
            for part, c in cut:
                for k in keys[i:i + c]:
                    parts[part].extend(scenes[k])
                i += c
            assert i == len(keys)
        else:
            for k, part in assign(scenes, rng, FRACTIONS).items():
                parts[part].extend(scenes[k])
    OUT.mkdir(parents=True, exist_ok=True)
    for p, ns in parts.items():
        (OUT / f"{p}.txt").write_text("\n".join(sorted(ns)) + "\n")
    return {p: sorted(ns) for p, ns in parts.items()}


def check(parts):
    """Hard guarantees; any failure aborts."""
    all_names = [n for ns in parts.values() for n in ns]
    assert len(all_names) == len(set(all_names)), "name in two partitions"
    scenes = {p: {scene_of(n) for n in ns} for p, ns in parts.items()}
    for a in parts:
        for b in parts:
            if a < b:
                assert not scenes[a] & scenes[b], f"scene shared: {a}/{b}"
    for p, ns in parts.items():
        s = set(ns)
        assert all(twin_of(n) in s for n in ns), f"twin left {p}"
    for n in all_names:
        src = SOURCES[n[0]]
        assert any(os.path.exists(f"{DATA}/{src}/{d}/{n[1:]}.json") for d in ("train", "val")), n
    manifest = {
        "seed": SEED,
        "files": {p: len(ns) for p, ns in parts.items()},
        "scenes": {p: len(v) for p, v in scenes.items()},
        "files_by_source": {p: dict(Counter(SOURCES[n[0]] for n in ns)) for p, ns in parts.items()},
        "scenes_by_source": {p: dict(Counter(SOURCES[k[0]] for k in v)) for p, v in scenes.items()},
    }
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


def _load(path):
    d = json.load(open(path))
    return (np.asarray(d["vectors_263"], dtype=np.float64),
            np.asarray(d["ego_in_ped_frame"], dtype=np.float64)[:, [0, 2]])


def _path(name):
    src = SOURCES[name[0]]
    for d in ("train", "val"):
        p = f"{DATA}/{src}/{d}/{name[1:]}.json"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


# HumanML3D feature groups share one std: root rotation, root linear, root height, joint positions,
# joint rotations, joint velocities, foot contacts.
GROUPS = [(0, 1), (1, 3), (3, 4), (4, 67), (67, 193), (193, 259), (259, 263)]


def statistics(names):
    """Mean over frames; group-averaged std as in HumanML3D. Ego: plain mean and std over frames."""
    with Pool(12) as pool:
        loaded = pool.map(_load, [_path(n) for n in names], chunksize=64)
    X = np.concatenate([a for a, _ in loaded])
    E = np.concatenate([b for _, b in loaded])
    mean, std = X.mean(0), X.std(0)
    for a, b in GROUPS:
        std[a:b] = std[a:b].mean()
    return mean, std, E.mean(0), E.std(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="also compute statistics on the training partition")
    args = ap.parse_args()
    parts = build()
    check(parts)
    if args.stats:
        # The formula must reproduce the old file on the old training set before it is trusted.
        old = f"{NAS}/data/vae/mean_std_txt/ava_nuscenes_waymo"
        old_names = [n for n in (l.strip() for l in open(OLD_LIST)) if n and
                     os.path.exists(f"{DATA}/{SOURCES[n[0]]}/train/{n[1:]}.json")]
        m, s, _, _ = statistics(old_names)
        print("old-train check: max |mean-old|", np.abs(m - np.load(f"{old}/Mean.npy")).max(),
              "max rel |std-old|", (np.abs(s - np.load(f"{old}/Std.npy")) / np.load(f"{old}/Std.npy")).max())
        mean, std, em, es = statistics(parts["train"])
        for f, v in (("Mean", mean), ("Std", std), ("Ego_Mean", em), ("Ego_Std", es)):
            np.save(OUT / f"{f}.npy", v.astype(np.float32))
        print("new: ego mean", em, "ego std", es, "| old ego", np.load(f"{old}/Ego_Mean.npy"), np.load(f"{old}/Ego_Std.npy"))
