#!/usr/bin/env python3
"""Write a de-slid copy of the dataset: refit the legs, then regenerate the 263-D features.

Pilot for the question "can the foot sliding be removed from the labels at all, and does a VAE
trained on corrected labels reconstruct without sliding?". See refit_contacts.py for the method and
its honest caveat (the legs become procedural gait).

The featurizer is the repo's own HumanML3D extractor, checked to reproduce the shipped vectors_263
from the shipped joints to ~2e-5 mean absolute error, so regenerated features are comparable.

Usage:
  python research/src/refit_dataset.py --out /home/erik/ssd2/datasets/egoped_deslid --limit 1400
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from mld.data.humanml.scripts.motion_process import extract_features
from mld.data.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from research.src.refit_contacts import refit

FEAT = dict(feet_thre=0.002, n_raw_offsets=torch.from_numpy(t2m_raw_offsets),
            kinematic_chain=t2m_kinematic_chain, face_joint_indx=[2, 1, 17, 16],
            fid_r=[8, 11], fid_l=[7, 10])
OUT_ROOT = None


def one(path):
    try:
        d = json.load(open(path))
        J = np.asarray(d["ped_in_ped_frame"], dtype=np.float64)
        if J.ndim != 3 or J.shape[1] != 22 or len(J) < 20:
            return None
        R = refit(J)
        F = np.asarray(extract_features(R.copy(), **FEAT), dtype=np.float32)
        n = len(F)
        out = dict(scene_id=d.get("scene_id"), object_id=d.get("object_id"),
                   ego_in_ped_frame=np.asarray(d["ego_in_ped_frame"],
                                               dtype=np.float32)[:n + 1].round(5).tolist(),
                   ped_in_ped_frame=R[:n].round(5).tolist(),
                   vectors_263=F.round(5).tolist())
        src, sub, name = path.split("/diffusion/")[1].split("/")
        dst = os.path.join(OUT_ROOT, src, sub)
        os.makedirs(dst, exist_ok=True)
        with open(os.path.join(dst, name), "w") as f:
            json.dump(out, f, separators=(",", ":"))
        return (src, name, n)
    except Exception:
        return None


def init(out):
    global OUT_ROOT
    OUT_ROOT = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/erik/NAS/methods/diffusion_gen/data/diffusion")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+", default=["ava", "nuscenes", "waymo"])
    ap.add_argument("--limit", type=int, default=0, help="per-source cap on full_dataset (0 = all)")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    files = []
    for s in a.sources:
        f = sorted(glob.glob(f"{a.data_root}/{s}/full_dataset/*.json"))
        files += f[: a.limit] if a.limit else f
    print(f"refitting {len(files)} sequences -> {a.out}", flush=True)
    with Pool(a.workers, initializer=init, initargs=(a.out,)) as p:
        R = [r for r in p.imap_unordered(one, files, chunksize=8) if r]
    print(f"wrote {len(R)} sequences")

    # mirror the original train/val membership by hardlinking from full_dataset
    for s in a.sources:
        made = {n for (src, n, _) in R if src == s}
        for sub in ("train", "val"):
            src_names = {os.path.basename(x)
                         for x in glob.glob(f"{a.data_root}/{s}/{sub}/*.json")} & made
            d = os.path.join(a.out, s, sub)
            os.makedirs(d, exist_ok=True)
            for n in src_names:
                tgt, lnk = os.path.join(a.out, s, "full_dataset", n), os.path.join(d, n)
                if not os.path.exists(lnk):
                    try:
                        os.link(tgt, lnk)
                    except OSError:
                        pass
            print(f"  {s}/{sub}: {len(src_names)}")

    # recompute normalization over the refitted training features
    V = []
    for s in a.sources:
        for p_ in glob.glob(f"{a.out}/{s}/train/*.json"):
            try:
                V.append(np.asarray(json.load(open(p_))["vectors_263"], dtype=np.float64))
            except Exception:
                pass
    X = np.concatenate(V, 0)
    ms = os.path.join(a.out, "mean_std")
    os.makedirs(ms, exist_ok=True)
    np.save(os.path.join(ms, "Mean.npy"), X.mean(0).astype(np.float32))
    np.save(os.path.join(ms, "Std.npy"), (X.std(0) + 1e-8).astype(np.float32))
    print(f"normalization from {X.shape[0]} frames -> {ms}")


if __name__ == "__main__":
    main()
