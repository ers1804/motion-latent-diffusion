#!/usr/bin/env python3
"""Reference-free quality audit of the pseudo-ground-truth pedestrian poses.

The dataset's poses are OmniRe estimates; no reference (mocap / manual labels) exists, so we
audit what can be checked WITHOUT one -- internal physical consistency -- plus one calibrated
comparison against real mocap statistics (HumanML3D Mean/Std, which ship with the repo).

Checks
  1. Body-shape template.  Bone lengths should differ between people. If every sequence has the
     same skeleton, body shape was never estimated and bone-length / symmetry checks carry no
     quality signal (rigidity is enforced, not earned).
  2. Foot skate (the load-bearing check; no confound).  During locomotion at least one foot should
     be stationary in world coordinates. We take the slowest of the 4 foot joints per frame and
     divide by root speed: ~0 means proper footfalls, ~1 means the body slides.
  3. Articulation vs real mocap.  Ratio of our per-dimension std to HumanML3D's, by 263-D block.
     CAVEAT: HumanML3D covers every kind of motion and ours covers walking, so part of any gap is
     content, not quality. Reported, not over-claimed.
  4. Ground penetration and jerk spikes (estimator-failure signatures).

Usage
  python research/src/pose_quality_audit.py                 # full dataset
  python research/src/pose_quality_audit.py --limit 200     # quick pilot
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.getcwd())
from mld.data.humanml.utils.paramUtil import t2m_kinematic_chain as CHAINS

BONES = [(c[i], c[i + 1]) for c in CHAINS for i in range(len(c) - 1)]
FEET = [7, 10, 8, 11]          # ankles + toes (HumanML3D 22-joint skeleton)
FPS = 20.0
MOVING = 0.3                   # m/s; only judge footfalls while the person actually walks
PLANTED = 0.1                  # m/s; a foot this slow counts as planted
BLOCKS = {"root rot. velocity": (0, 1), "root lin. velocity": (1, 3), "root height": (3, 4),
          "joint positions (ric)": (4, 67), "joint rotations": (67, 193),
          "local velocities": (193, 259), "foot contact": (259, 263)}


def audit_one(path):
    try:
        d = json.load(open(path))
        J = np.asarray(d["ped_in_ped_frame"], dtype=np.float64)
    except Exception:
        return None
    if J.ndim != 3 or J.shape[1] != 22 or len(J) < 20:
        return None
    src = path.split("/diffusion/")[1].split("/")[0]
    bl = np.stack([np.linalg.norm(J[:, a] - J[:, b], axis=-1) for a, b in BONES], 1)   # (T,B)
    v = np.diff(J, axis=0) * FPS
    root = np.linalg.norm(v[:, 0][:, [0, 2]], axis=-1)
    foot = np.linalg.norm(v[:, FEET][:, :, [0, 2]], axis=-1)
    mn = foot.min(1)
    mov = root > MOVING
    rel = J - J[:, :1]
    ground = np.percentile(J[:, :, 1], 1)
    jerk = np.linalg.norm(np.diff(J, n=3, axis=0), axis=-1).mean(-1) * FPS ** 3
    out = dict(src=src, T=len(J), bone_mean=bl.mean(0),
               bone_cv=float(np.median(bl.std(0) / (bl.mean(0) + 1e-8))),
               root_speed=float(np.median(root[mov])) if mov.sum() >= 10 else np.nan,
               skate_ratio=float(np.median(mn[mov] / (root[mov] + 1e-8))) if mov.sum() >= 10 else np.nan,
               planted_frac=float((mn[mov] < PLANTED).mean()) if mov.sum() >= 10 else np.nan,
               ankle_swing=float(max(rel[:, 10].std(0).max(), rel[:, 7].std(0).max())),
               ground_pen=float(max(0.0, ground - J[:, :, 1].min())),
               jerk_p99=float(np.percentile(jerk, 99)), moving=bool(mov.sum() >= 10))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/erik/NAS/methods/diffusion_gen/data/diffusion")
    ap.add_argument("--sources", nargs="+", default=["ava", "nuscenes", "waymo"])
    # NOTE: <source>/full_dataset/ == <source>/train/ + <source>/val/, so globbing */*.json
    # double-counts every sequence. full_dataset is the canonical complete set (14,172 total).
    ap.add_argument("--subdir", default="full_dataset")
    ap.add_argument("--limit", type=int, default=0, help="per-source cap (0 = all)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="research/data/pose_quality_audit.npz")
    a = ap.parse_args()

    files = []
    for s in a.sources:
        f = sorted(glob.glob(f"{a.data_root}/{s}/{a.subdir}/*.json"))
        files += f[: a.limit] if a.limit else f
    print(f"auditing {len(files)} sequences with {a.workers} workers ...", flush=True)
    with Pool(a.workers) as p:
        R = [r for r in p.imap_unordered(audit_one, files, chunksize=16) if r]
    print(f"{len(R)} usable sequences\n")

    # ---- 1. body-shape template ------------------------------------------------
    BM = np.stack([r["bone_mean"] for r in R])
    ref = np.median(BM, 0)
    dev = np.abs(BM - ref) / (ref + 1e-8)
    print("1. BODY SHAPE")
    print(f"   max relative deviation of any bone in any sequence from the dataset median: "
          f"{dev.max():.2e}")
    print(f"   distinct skeletons (rounded to 1 mm): {len(np.unique(np.round(BM,3),axis=0))}")
    print(f"   median within-sequence bone-length CV: {np.median([r['bone_cv'] for r in R]):.2e}")
    verdict = "ONE FIXED TEMPLATE for every pedestrian" if dev.max() < 1e-3 else "per-person shapes"
    print(f"   => {verdict}\n")

    # ---- 2. foot skate ---------------------------------------------------------
    M = [r for r in R if r["moving"]]
    print(f"2. FOOT SKATE  (n={len(M)} sequences with locomotion)")
    print(f"   {'source':10s} {'root m/s':>9s} {'skate ratio':>12s} {'% frames planted':>17s}")
    for s in a.sources:
        S = [r for r in M if r["src"] == s]
        if not S:
            continue
        g = lambda k: np.nanmedian([x[k] for x in S])
        print(f"   {s:10s} {g('root_speed'):9.2f} {g('skate_ratio'):12.2f} "
              f"{100*g('planted_frac'):16.1f}%")
    sk = np.array([r["skate_ratio"] for r in M], dtype=float)
    pf = np.array([r["planted_frac"] for r in M], dtype=float)
    print(f"   overall skate ratio  p10 {np.nanpercentile(sk,10):.2f} | median "
          f"{np.nanmedian(sk):.2f} | p90 {np.nanpercentile(sk,90):.2f}   (0 = footfalls, 1 = sliding)")
    print(f"   sequences with a planted foot in <5% of moving frames: "
          f"{100*np.nanmean(pf < 0.05):.1f}%\n")

    # ---- 3. articulation vs real mocap ----------------------------------------
    print("3. ARTICULATION vs REAL MOCAP (std ratio, ours / HumanML3D)")
    try:
        hm = np.load("datasets/humanml3d/Std.npy")
        ours = np.load("research/data/heldout_split/val_sel/Std.npy")
        for k, (i, j) in BLOCKS.items():
            print(f"   {k:24s} {np.median(ours[i:j]/(hm[i:j]+1e-8)):6.2f}")
        print("   (<1 = less variable than mocap; part of any gap is content, not quality)\n")
    except Exception as e:
        print(f"   skipped: {e}\n")

    # ---- 4. failure signatures -------------------------------------------------
    gp = np.array([r["ground_pen"] for r in R]); jk = np.array([r["jerk_p99"] for r in R])
    aw = np.array([r["ankle_swing"] for r in R])
    print("4. FAILURE SIGNATURES")
    print(f"   ground penetration > 5 cm: {100*(gp>0.05).mean():.1f}% of sequences")
    print(f"   ankle swing < 0.05 m (near-static legs): {100*(aw<0.05).mean():.1f}%")
    print(f"   jerk p99 median {np.median(jk):.0f} m/s^3")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **{k: np.array([r[k] for r in R])
                                  for k in ("src","T","bone_cv","root_speed","skate_ratio",
                                            "planted_frac","ankle_swing","ground_pen","jerk_p99","moving")},
                        bone_mean=BM)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
