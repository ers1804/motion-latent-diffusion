#!/usr/bin/env python3
"""Proxy behavioral-intention labels from trajectories (item 9, review_tracker.md).

Heuristic, automatically derived categories — presented in the paper as such.
All geometry in the shared pedestrian-centric ground-plane (XZ) frame at FPS=20.

Labels (non-exclusive flags):
  crossing : the pedestrian's root path intersects the ego vehicle's path
             (2-D segment-intersection test between the two polylines).
  stopping : root speed < STOP_SPEED (m/s) sustained for >= STOP_FRAMES frames.
  walking  : neither flag (residual category when reporting exclusive classes).

Usage (GT distributions):
  python research/src/intention_labels.py            # val + train-sample stats
Importable:
  from research.src.intention_labels import label_from_roots
"""
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np

FPS = 20.0
STOP_SPEED = 0.2      # m/s
STOP_FRAMES = 20      # 1 s
BASE = "/home/erik/NAS/methods/diffusion_gen/data/diffusion"


def _segments_intersect(p1, p2, q1, q2):
    """2-D proper/improper segment intersection via orientation tests."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    def on_seg(a, b, c):
        return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and
                min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)

    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    for a, b, c, o in ((p1, p2, q1, o1), (p1, p2, q2, o2), (q1, q2, p1, o3), (q1, q2, p2, o4)):
        if o == 0 and on_seg(a, b, c):
            return True
    return False


def paths_cross(ped_xz, ego_xz, stride=2):
    """True if the ped root polyline intersects the ego polyline (subsampled)."""
    P = np.asarray(ped_xz)[::stride]
    E = np.asarray(ego_xz)[::stride]
    for i in range(len(P) - 1):
        for j in range(len(E) - 1):
            if _segments_intersect(P[i], P[i + 1], E[j], E[j + 1]):
                return True
    return False


def is_stopping(ped_xz, fps=FPS, speed_thr=STOP_SPEED, min_frames=STOP_FRAMES):
    v = np.linalg.norm(np.diff(np.asarray(ped_xz), axis=0), axis=1) * fps
    slow = v < speed_thr
    run = best = 0
    for s in slow:
        run = run + 1 if s else 0
        best = max(best, run)
    return best >= min_frames


def label_from_roots(ped_xz, ego_xz, fps=FPS):
    """Return dict of flags for one sequence (any length >= 2)."""
    return {
        "crossing": paths_cross(ped_xz, ego_xz),
        "stopping": is_stopping(ped_xz, fps=fps),
    }


def label_json(path):
    d = json.load(open(path))
    ped = np.asarray(d["ped_in_ped_frame"], dtype=np.float32)[:, 0, :][:, [0, 2]]
    ego = np.asarray(d["ego_in_ped_frame"], dtype=np.float32)[:, [0, 2]]
    T = min(len(ped), len(ego))
    return label_from_roots(ped[:T], ego[:T])


def summarize(files, name):
    n = cross = stop = both = neither = 0
    for f in files:
        try:
            lab = label_json(f)
        except Exception:
            continue
        n += 1
        c, s = lab["crossing"], lab["stopping"]
        cross += c; stop += s; both += (c and s); neither += (not c and not s)
    print(f"[{name}] n={n}  crossing={cross} ({100*cross/n:.1f}%)  "
          f"stopping={stop} ({100*stop/n:.1f}%)  both={both} ({100*both/n:.1f}%)  "
          f"walking-only={neither} ({100*neither/n:.1f}%)")


if __name__ == "__main__":
    val = sorted(sum((glob(f"{BASE}/{s}/val/*.json") for s in ("ava", "nuscenes", "waymo")), []))
    summarize(val, "val (all)")
    vt_names = {l.strip() for l in open("research/data/heldout_split/val_test/val.txt")}
    vt = [f for f in val if any(Path(f).stem == n[1:] or Path(f).stem == n for n in vt_names
                                if n[:1].upper() in "ANW")]
    # simpler: match by stripped stem set
    stems = {n[1:] if n[:1].upper() in "ANW" else n for n in vt_names}
    vt = [f for f in val if Path(f).stem in stems]
    summarize(vt, "val_test (held-out)")
    import random
    train = sorted(sum((glob(f"{BASE}/{s}/train/*.json") for s in ("ava", "nuscenes", "waymo")), []))
    random.seed(0); random.shuffle(train)
    summarize(train[:1500], "train (1500 sample)")
