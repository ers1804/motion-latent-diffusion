#!/usr/bin/env python3
"""Pilot verdict: can a VAE represent non-sliding motion?

Reconstructs the SAME de-slid held-out sequences through two VAEs and measures foot skate:
  - the original VAE (trained on the sliding labels)  -> does it re-impose sliding?
  - a VAE fine-tuned on the de-slid labels            -> does the sliding go away?

If the fine-tuned VAE reconstructs at ~0, the latent space is not the obstacle and correcting the
labels is a viable (if expensive) path. If it stays near 1, the bottleneck itself imposes sliding
and no amount of relabelling would help.
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from mld.data.humanml.scripts.motion_process import recover_from_ric
from research.src.skate_decomposition import skate

D = "/home/erik/ssd2/datasets/egoped_deslid"


def load_vae(ckpt, latent=4):
    from mld.config import parse_args
    from mld.data.get_data import get_datasets
    from mld.models.get_model import get_model
    sys.argv = ["p", "--cfg", "configs/config_ego_motion_vae.yaml", "--nodebug", "--overrides",
                "NAME=pilot_eval", "TRAIN.RESUME=", "TRAIN.STAGE=vae",
                f"model.latent_dim=[{latent}, 256]",
                f"DATASET.EGOMOTION.ROOT=[{D}/ava, {D}/nuscenes, {D}/waymo]",
                f"DATASET.EGOMOTION.MEAN_STD_PATH={D}/mean_std",
                f"DATASET.EGOMOTION.EGO_MEAN_STD_PATH={D}/mean_std",
                "DATASET.EGOMOTION.INTERACTION_CROP=False",
                "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=False",
                "TEST.SPLIT=val", "TEST.BATCH_SIZE=16", "METRIC.TYPE=['EgoMotionMetrics']"]
    cfg = parse_args(phase="test"); cfg.FOLDER = "results"
    dm = get_datasets(cfg, logger=None, phase="test")[0]
    model = get_model(cfg, dm)
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    vd = {k.replace("vae.", ""): v for k, v in sd.items() if k.split(".")[0] == "vae"}
    model.vae.load_state_dict(vd, strict=True)
    return model.vae.cuda().eval(), dm


def run(vae, dm, mean, std):
    out = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            m = batch["motion"].cuda(); L = batch["length"]
            z, _ = vae.encode(m, L)
            rec = vae.decode(z, L).cpu()
            raw = rec * torch.tensor(std) + torch.tensor(mean)
            J = recover_from_ric(raw.float(), 22).numpy()
            for i in range(J.shape[0]):
                s = skate(J[i], int(L[i]))
                if s:
                    out.append(s[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--original", default="/home/erik/NAS/methods/diffusion_gen/models/helma_models/"
                    "models/vae/ego_motion_vae_latent_4_wo_traj_interaction_crop_weighted_sampling/"
                    "checkpoints/epoch=5999.ckpt")
    a = ap.parse_args()
    mean = np.load(f"{D}/mean_std/Mean.npy"); std = np.load(f"{D}/mean_std/Std.npy")

    # input reference: the de-slid labels themselves
    ref = []
    for s in ("ava", "nuscenes", "waymo"):
        for p in glob.glob(f"{D}/{s}/val/*.json"):
            V = np.asarray(json.load(open(p))["vectors_263"], dtype=np.float32)
            J = recover_from_ric(torch.tensor(V)[None], 22)[0].numpy()
            x = skate(J, len(J))
            if x:
                ref.append(x[0])

    res = {}
    for tag, ck in (("original VAE (trained on sliding labels)", a.original),
                    ("VAE fine-tuned on de-slid labels", a.finetuned)):
        vae, dm = load_vae(ck)
        dm.setup(stage="test")
        res[tag] = run(vae, dm, mean, std)
        del vae
        torch.cuda.empty_cache()

    print(f"\nfoot-skate on the SAME de-slid held-out sequences "
          f"(0 = clean footfalls, 1 = body slides)\n")
    print(f"{'stage':46s} {'n':>5s} {'skate':>7s}")
    print(f"{'de-slid labels (input to the VAEs)':46s} {len(ref):5d} {np.median(ref):7.3f}")
    for k, v in res.items():
        print(f"{'reconstruction: '+k:46s} {len(v):5d} {np.median(v):7.3f}")
    print(f"\nfor reference, on the ORIGINAL labels the same VAE reconstructed at 0.94 "
          f"and the labels themselves at 0.96")


if __name__ == "__main__":
    main()
