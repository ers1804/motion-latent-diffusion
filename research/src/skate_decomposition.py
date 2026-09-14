#!/usr/bin/env python3
"""Where does the foot sliding come from: the labels, the VAE, or the diffusion model?

research/src/pose_quality_audit.py showed the pseudo-GT slides (foot-skate ratio 0.97). This
decomposes the pipeline on the SAME held-out conditions and the SAME metric:

    ground truth  ->  VAE round-trip (encode+decode GT)  ->  generated samples

If the VAE round-trip already slides, the sliding is baked into the latent space that every
diffusion model decodes through, and no training-side loss on the denoiser can remove it.
If generation slides MORE than GT, the model amplifies the artifact and is worth fixing.

Usage: python research/src/skate_decomposition.py --models h4 ia uncond_pipeline
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from research.src.eval_ade_fde import MODELS, NAS, VT, FV   # reuse the model registry

FEET = [7, 10, 8, 11]
FPS, MOVING, PLANTED = 20.0, 0.3, 0.1


def skate(J, L):
    """J: (T,22,3) joints for one sequence, L: valid length. Returns (ratio, planted_frac) or None."""
    J = J[:L]
    if L < 20:
        return None
    v = np.diff(J, axis=0) * FPS
    root = np.linalg.norm(v[:, 0][:, [0, 2]], axis=-1)
    mn = np.linalg.norm(v[:, FEET][:, :, [0, 2]], axis=-1).min(1)
    mov = root > MOVING
    if mov.sum() < 10:
        return None
    return float(np.median(mn[mov] / (root[mov] + 1e-8))), float((mn[mov] < PLANTED).mean())


def collect(Jb, lengths, acc):
    for i in range(Jb.shape[0]):
        r = skate(Jb[i], int(lengths[i]))
        if r:
            acc[0].append(r[0]); acc[1].append(r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["h4", "ia", "uncond_pipeline"])
    ap.add_argument("--split", default="val_test", choices=["val_test", "full_val"])
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    split_dir = VT if args.split == "val_test" else FV
    rows = []

    for mi, name in enumerate(args.models):
        spec = MODELS[name]
        overrides = [
            f"DATASET.EGOMOTION.ROOT=[{NAS}/data/diffusion/ava, {NAS}/data/diffusion/nuscenes, {NAS}/data/diffusion/waymo]",
            f"DATASET.EGOMOTION.MEAN_STD_PATH={split_dir}",
            f"DATASET.EGOMOTION.EGO_MEAN_STD_PATH={split_dir}",
            "DATASET.EGOMOTION.INTERACTION_CROP=False",
            "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=False",
            f"TEST.CHECKPOINTS={spec['ckpt']}", "METRIC.TYPE=['EgoMotionMetrics']",
            "TEST.REPLICATION_TIMES=1", "TEST.SPLIT=val",
            f"model.guidance_scale={spec['guidance']}", f"TEST.BATCH_SIZE={args.batch}",
        ] + spec.get("extra", [])
        sys.argv = ["skate", "--cfg", spec["cfg"], "--nodebug", "--overrides"] + overrides

        from mld.config import parse_args
        from mld.data.get_data import get_datasets
        from mld.models.get_model import get_model
        import pytorch_lightning as pl

        cfg = parse_args(phase="test"); cfg.FOLDER = "results"
        pl.seed_everything(cfg.SEED_VALUE)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dm = get_datasets(cfg, logger=None, phase="test")[0]
        model = get_model(cfg, dm)
        model.load_state_dict(torch.load(spec["ckpt"], map_location="cpu")["state_dict"])
        model = model.to(dev).eval()
        dm.setup(stage="test")

        gt, vae, gen = ([], []), ([], []), ([], [])
        with torch.no_grad():
            for batch in dm.test_dataloader():
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                lengths = batch["length"]
                if spec.get("zero_ego"):
                    batch["ego"] = torch.zeros_like(batch["ego"])
                if mi == 0:                                   # GT is model-independent
                    collect(dm.feats2joints(batch["motion"].detach().cpu()).numpy(), lengths, gt)
                    z, _ = model.vae.encode(batch["motion"], lengths)
                    rec = model.vae.decode(z, lengths).detach().cpu()
                    collect(dm.feats2joints(rec).numpy(), lengths, vae)
                rs = model.test_diffusion_forward(batch)
                collect(rs["joints_rst"].detach().cpu().numpy(), lengths, gen)
        if mi == 0:
            rows.append(("ground truth (pseudo-GT labels)", gt))
            rows.append(("VAE round-trip (encode+decode GT)", vae))
        rows.append((f"generated: {name}", gen))
        del model
        torch.cuda.empty_cache()

    print(f"\nfoot-skate decomposition, split={args.split}   "
          f"(ratio 0 = clean footfalls, 1 = body slides as a rigid unit)\n")
    print(f"{'stage':38s} {'n':>5s} {'skate ratio':>12s} {'% frames planted':>17s}")
    out = {}
    for label, (r, p) in rows:
        if not r:
            continue
        print(f"{label:38s} {len(r):5d} {np.median(r):12.2f} {100*np.median(p):16.1f}%")
        out[label] = np.array([np.median(r), np.median(p)])
    np.savez("research/data/skate_decomposition.npz", **out)
    print("\nsaved research/data/skate_decomposition.npz")


if __name__ == "__main__":
    main()
