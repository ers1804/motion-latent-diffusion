#!/usr/bin/env python3
"""Do pedestrians orient toward the vehicle? Raw SMPL estimates, per dataset (plan step 7).

Training labels carry no head yaw, so this reads the per-frame SMPL fits behind them. Attention
should fade with distance. A camera-facing bias of the monocular estimator should not, and it looks
the same here because the camera sits on the vehicle; near/far is the only handle that separates them.

Usage: python research/src/orientation_test.py
"""
import json
import os
import pickle

import numpy as np

ROOT = "/home/erik/NAS/methods/diffusion_gen/data/drivestudio/"
SCENES = {"nuscenes": 200, "waymo": 200}
# Waymo camera axes (x forward, y left, z up) from OpenCV axes used by the SMPL fits.
CV_TO_WAYMO = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])


def ang(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cam_to_world(ds, path, t, cam, ego):
    if ds == "waymo":
        return ego[t, :3, :3] @ np.loadtxt(f"{path}/extrinsics/{cam}.txt")[:3, :3] @ CV_TO_WAYMO
    f = f"{path}/extrinsics/{t:03d}_{cam}.txt"
    if not os.path.exists(f):
        f = f"{path}/extrinsics/{cam}.txt"
    return np.loadtxt(f)[:3, :3] if os.path.exists(f) else None


def frames(ds):
    rows = []
    base = f"{ROOT}{ds}/drivestudio_output/"
    for scene in sorted(os.listdir(base))[:SCENES[ds]]:
        p = base + scene
        try:
            fits = pickle.load(open(p + "/humanpose/smpl.pkl", "rb"))
            info = json.load(open(p + "/instances/instances_info.json"))
        except OSError:
            continue
        nf = len(os.listdir(p + "/ego_pose"))
        ego = np.stack([np.loadtxt(f"{p}/ego_pose/{i:03d}.txt") for i in range(nf)])
        for oid, o in fits.items():
            if str(oid) not in info:
                continue
            ann = info[str(oid)]["frame_annotations"]
            pos = np.full((nf, 3), np.nan)
            pos[np.array(ann["frame_idx"])] = np.array(ann["obj_to_world"])[:, :3, 3]
            mask = o["valid_mask"].numpy()[:nf] & o["matched_mask"].numpy()[:nf]
            if mask.sum() < 20:
                continue
            cam = o["selected_cam_idx"].numpy()[:nf]
            orient = o["smpl"]["global_orient"].numpy()[:nf, 0]
            pose = o["smpl"]["body_pose"].numpy()[:nf]
            for t in np.where(mask)[0]:
                W = cam_to_world(ds, p, t, int(cam[t]), ego)
                if W is None or not np.isfinite(pos[t, 0]):
                    continue
                root = orient[t]
                torso = root @ pose[t, 2] @ pose[t, 5] @ pose[t, 8]
                head = torso @ pose[t, 11] @ pose[t, 14]
                up, fw, hd = W @ root[:, 1], W @ torso[:, 2], W @ head[:, 2]
                walk_yaw, speed = np.nan, np.nan
                if 0 < t < nf - 1 and np.isfinite(pos[t - 1, 0]) and np.isfinite(pos[t + 1, 0]):
                    v = pos[t + 1, :2] - pos[t - 1, :2]
                    walk_yaw, speed = np.arctan2(v[1], v[0]), np.linalg.norm(v) / 0.2
                b = ego[t, :2, 3] - pos[t, :2]
                rows.append((np.arctan2(fw[1], fw[0]), np.arctan2(hd[1], hd[0]), np.arctan2(b[1], b[0]),
                             np.linalg.norm(b), speed, walk_yaw, up[2]))
    return np.array(rows)


def report(ds, A):
    torso, head, to_veh, dist, speed, walk, up = A.T
    moving = speed > 0.7
    print(f"== {ds} frames {len(A)} | sanity: up {np.nanmean(up):+.3f}, "
          f"torso-vs-walking {np.nanmean(np.cos(ang(torso - walk))[moving]):+.3f}")
    ok = up > 0.8
    turn = ang(head - torso)
    print("   head turn deg median %.1f p95 %.1f" % tuple(np.percentile(np.degrees(np.abs(turn[ok])), [50, 95])))
    rng = np.random.default_rng(0)
    for name, m in (("near<15m", dist < 15), ("near standing", (dist < 15) & (speed < 0.3)),
                    ("near walking", (dist < 15) & (speed > 0.7)), ("far>30m", dist > 30),
                    ("far standing", (dist > 30) & (speed < 0.3))):
        m = m & ok
        side = np.sin(ang(to_veh - torso))[m]
        r = np.corrcoef(side, np.sin(turn)[m])[0, 1]
        shuffled = to_veh[m][rng.permutation(m.sum())]
        print(f"   {name:14s} n={m.sum():6d} r(head turn, vehicle side)={r:+.3f} | "
              f"cos(head,vehicle) {np.cos(ang(head[m] - to_veh[m])).mean():+.3f} "
              f"shuffled {np.cos(ang(head[m] - shuffled)).mean():+.3f} | "
              f"cos(torso,vehicle) {np.cos(ang(torso[m] - to_veh[m])).mean():+.3f}")


if __name__ == "__main__":
    for ds in SCENES:
        report(ds, frames(ds))
