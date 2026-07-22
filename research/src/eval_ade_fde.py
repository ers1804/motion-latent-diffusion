#!/usr/bin/env python3
"""Evaluator-independent conditioning metric: root-trajectory ADE/FDE (item 5).

For each evaluation sample, generate K motions conditioned on the ego
trajectory, recover the pedestrian root (pelvis) trajectory on the ground
plane (XZ) via feats2joints (denorm + recover_from_ric), and compare with the
ground-truth root trajectory over the true sequence length:

  ADE  = mean_t || root_gen(t) - root_gt(t) ||_2   (average displacement error)
  FDE  = || root_gen(L-1) - root_gt(L-1) ||_2      (final displacement error)
  minADE_K / minFDE_K = best of K samples per condition (standard for
  stochastic models), then averaged over the dataset.

Unlike R-Precision, this uses NO learned evaluator and NO model-specific
embedding space, so it is directly comparable across architectures.

Usage:
  python research/src/eval_ade_fde.py --model h4 [--k 5] [--split val_test]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

NAS = "/home/erik/NAS/methods/diffusion_gen"
HM = f"{NAS}/models/helma_models/models/mld"
VT = "research/data/heldout_split/val_test"
FV = f"{NAS}/data/vae/mean_std_txt/ava_nuscenes_waymo"

MODELS = {
    "h4": dict(cfg="configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml",
               ckpt=f"{HM}/ego_motion_diffusion_h4_trans_dec/checkpoints/epoch=3399.ckpt",
               guidance=10),
    "h2": dict(cfg="configs/config_ego_motion_new_vae_stoch.yaml",
               ckpt=f"{HM}/ego_motion_diffusion_interaction_crop_weighted_1_helma/checkpoints/epoch=4399.ckpt",
               guidance=5),
    "h6": dict(cfg="configs/config_ego_motion_new_vae_stoch_latent_4_h6_unfreeze_ego.yaml",
               ckpt=f"{HM}/ego_motion_diffusion_h6_unfreeze_ego/checkpoints/epoch=3399.ckpt",
               guidance=10),
    "h4_uncond": dict(cfg="configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml",
                      ckpt=f"{HM}/ego_motion_diffusion_h4_trans_dec/checkpoints/epoch=3399.ckpt",
                      guidance=10, zero_ego=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--k", type=int, default=5, help="samples per condition")
    ap.add_argument("--split", default="val_test", choices=["val_test", "full_val"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--save-roots", action="store_true",
                    help="also save generated/GT root XZ trajectories (item 9 behavioral probe)")
    args = ap.parse_args()
    spec = MODELS[args.model]
    split_dir = VT if args.split == "val_test" else FV

    overrides = [
        f"DATASET.EGOMOTION.ROOT=[{NAS}/data/diffusion/ava, {NAS}/data/diffusion/nuscenes, {NAS}/data/diffusion/waymo]",
        f"DATASET.EGOMOTION.MEAN_STD_PATH={split_dir}",
        f"DATASET.EGOMOTION.EGO_MEAN_STD_PATH={split_dir}",
        "DATASET.EGOMOTION.INTERACTION_CROP=False",
        "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=False",
        f"TEST.CHECKPOINTS={spec['ckpt']}",
        "METRIC.TYPE=['EgoMotionMetrics']",
        "TEST.REPLICATION_TIMES=1",
        "TEST.SPLIT=val",
        f"model.guidance_scale={spec['guidance']}",
        f"TEST.BATCH_SIZE={args.batch}",
    ]
    sys.argv = ["eval_ade_fde", "--cfg", spec["cfg"], "--nodebug", "--overrides"] + overrides

    from mld.config import parse_args
    from mld.data.get_data import get_datasets
    from mld.models.get_model import get_model
    import pytorch_lightning as pl

    cfg = parse_args(phase="test")
    cfg.FOLDER = "results"
    pl.seed_everything(cfg.SEED_VALUE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = get_datasets(cfg, logger=None, phase="test")
    datamodule = datasets[0]
    model = get_model(cfg, datamodule)
    state = torch.load(spec["ckpt"], map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    model = model.to(device).eval()

    datamodule.setup(stage="test")
    loader = datamodule.test_dataloader()

    ade_all, fde_all = [], []           # per condition, per k
    roots_gen, roots_gt, lens_all = [], [], []   # optional dumps (--save-roots)
    n_cond = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            lengths = batch["length"]
            if spec.get("zero_ego"):
                batch["ego"] = torch.zeros_like(batch["ego"])
            # GT root trajectory (denorm via datamodule.feats2joints)
            joints_ref = datamodule.feats2joints(batch["motion"].detach().cpu())
            gt_root = joints_ref[:, :, 0, [0, 2]].numpy()  # (B, T, 2)
            B = gt_root.shape[0]
            ades = np.zeros((B, args.k)); fdes = np.zeros((B, args.k))
            batch_gen = []
            for k in range(args.k):
                rs = model.test_diffusion_forward(batch)
                gen_root = rs["joints_rst"].detach().cpu()[:, :, 0, [0, 2]].numpy()
                if args.save_roots:
                    batch_gen.append(gen_root.astype("float32"))
                for i in range(B):
                    L = int(lengths[i])
                    d = np.linalg.norm(gen_root[i, :L] - gt_root[i, :L], axis=-1)
                    ades[i, k] = d.mean()
                    fdes[i, k] = d[L - 1]
            ade_all.append(ades); fde_all.append(fdes)
            if args.save_roots:
                roots_gen.append(np.stack(batch_gen, axis=1))  # (B, K, T, 2)
                roots_gt.append(gt_root.astype("float32"))
                lens_all.extend(int(l) for l in lengths)
            n_cond += B

    ades = np.concatenate(ade_all)  # (N, K)
    fdes = np.concatenate(fde_all)
    print(f"model={args.model} split={args.split} K={args.k} N={n_cond}")
    print(f"ADE_mean  = {ades.mean():.4f}   FDE_mean  = {fdes.mean():.4f}")
    print(f"minADE_{args.k} = {ades.min(axis=1).mean():.4f}   minFDE_{args.k} = {fdes.min(axis=1).mean():.4f}")
    out = f"research/data/ade_fde_{args.model}_{args.split}_k{args.k}.npz"
    if args.save_roots:
        np.savez(out, ades=ades, fdes=fdes,
                 roots_gen=np.concatenate(roots_gen), roots_gt=np.concatenate(roots_gt),
                 lengths=np.array(lens_all))
    else:
        np.savez(out, ades=ades, fdes=fdes)
    print("saved", out)


if __name__ == "__main__":
    main()
