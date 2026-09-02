#!/usr/bin/env python3
"""Pretrained text-to-motion MLD as an external baseline (item 10b, P1/P2/P3).

Runs the OFFICIAL MLD HumanML3D checkpoint (1222_mld_humanml3d_FID041.ckpt) on
our held-out ego conditions, driven by text prompts, and scores it with the SAME
t2m evaluator path as every other baseline.

NORMALIZATION (the trap):
  pretrained MLD emits motion in *HumanML3D dataset* normalization, while our
  eval path feeds *our* normalization straight to the t2m encoders (EgoMotion
  has no renorm4t2m). We therefore convert
      MLD out --(*hml_std + hml_mean)--> raw --((-our_mean)/our_std)--> ours
  Sanity gate: gt_Diversity must land ~5.5; anything near 0.6 means the
  normalization is wrong (the June-2026 collapse bug).

Prompts:
  P1  naive         "a pedestrian tries to cross the street and reacts to the ego vehicle"
  P2  idiomatic     "a person walks forward and then stops"
  P3  oracle        per-sample synthesized `body` caption (GT-derived -> label ORACLE)

Usage:
  python research/src/eval_pretrained_mld.py --prompt P2 [--split val_test] [--limit 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

NAS = "/home/erik/NAS/methods/diffusion_gen"
VT = "research/data/heldout_split/val_test"
FV = f"{NAS}/data/vae/mean_std_txt/ava_nuscenes_waymo"
MLD_CKPT = "checkpoints/mld_humanml3d_checkpoint/1222_mld_humanml3d_FID041.ckpt"
HML = "datasets/humanml3d"

PROMPTS = {
    "P1": "a pedestrian tries to cross the street and reacts to the ego vehicle",
    "P2": "a person walks forward and then stops",
    "P3": None,   # per-sample body caption
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, choices=list(PROMPTS))
    ap.add_argument("--split", default="val_test", choices=["val_test", "full_val"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    split_dir = VT if args.split == "val_test" else FV
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- our data (GT in OUR normalization) ------------------------------
    from mld.data.EgoMotion import EgoMotionDataset, ego_motion_collate
    from torch.utils.data import DataLoader
    our_mean = np.load(f"{split_dir}/Mean.npy"); our_std = np.load(f"{split_dir}/Std.npy")
    roots = [f"{NAS}/data/diffusion/{s}" for s in ("ava", "nuscenes", "waymo")]
    ds = EgoMotionDataset(data_root=roots, split="val", split_list_root=split_dir,
                          mean=our_mean, std=our_std,
                          captions_path="research/data/synth_captions.json",
                          caption_vocab="body")
    if args.limit:
        ds.sample_paths = ds.sample_paths[:args.limit]
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4,
                        collate_fn=ego_motion_collate)

    # ---- pretrained MLD ---------------------------------------------------
    overrides = [f"TEST.CHECKPOINTS={MLD_CKPT}", "TEST.SPLIT=val",
                 f"TEST.BATCH_SIZE={args.batch}", "METRIC.TYPE=['TM2TMetrics']",
                 # normally injected by get_datasets(), which we bypass
                 "DATASET.NFEATS=263", "DATASET.NJOINTS=22", "DATASET.NCLASSES=10",
                 # use the SAME t2m evaluator weights as every other baseline
                 "TEST.DATASETS=['egomotion']", "EVAL.DATASETS=['egomotion']"]
    sys.argv = ["eval_pretrained_mld", "--cfg", "configs/config_mld_humanml3d.yaml",
                "--nodebug", "--overrides"] + overrides
    from mld.config import parse_args
    cfg = parse_args(phase="test")
    cfg.FOLDER = "results"
    # mld.py only needs `feats2joints` from the datamodule for our use; give it a
    # shim that decodes in HumanML3D space (the space this model generates in).
    class _DMShim:
        def __init__(self, mean, std):
            self.mean, self.std = mean, std
            self.njoints = 22
        def feats2joints(self, features):
            from mld.data.humanml.scripts.motion_process import recover_from_ric
            mean = torch.tensor(self.mean).to(features)
            std = torch.tensor(self.std).to(features)
            return recover_from_ric(features * std + mean, 22)

    from mld.models.get_model import get_model
    model = get_model(cfg, _DMShim(np.load(f"{HML}/Mean.npy"), np.load(f"{HML}/Std.npy")))
    sd = torch.load(MLD_CKPT, map_location="cpu")["state_dict"]
    res = model.load_state_dict(sd, strict=False)
    if res is not None:
        print(f"[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    # verify the denoiser actually received pretrained weights (not random init)
    ck_den = {k: v for k, v in sd.items() if k.startswith("denoiser.")}
    cur = dict(model.named_parameters())
    same = sum(1 for k, v in ck_den.items()
               if k in cur and cur[k].shape == v.shape
               and torch.allclose(cur[k].detach().cpu().float(), v.float(), atol=1e-6))
    print(f"[load] denoiser tensors matching checkpoint: {same}/{len(ck_den)}")
    assert same > 0, "denoiser weights did NOT load — would be a random-init baseline"
    model = model.to(device).eval()

    hml_mean = torch.tensor(np.load(f"{HML}/Mean.npy"), dtype=torch.float32)
    hml_std = torch.tensor(np.load(f"{HML}/Std.npy"), dtype=torch.float32)
    o_mean = torch.tensor(our_mean, dtype=torch.float32)
    o_std = torch.tensor(our_std, dtype=torch.float32)

    gen_all, gt_all, lens_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            lengths = batch["length"]
            texts = ([PROMPTS[args.prompt]] * len(lengths) if args.prompt != "P3"
                     else list(batch["text"]))
            b = {"text": texts, "length": lengths,
                 "motion": batch["motion"].to(device)}
            rs = model.test_diffusion_forward(b)
            gen = rs["m_rst"].detach().cpu().float()        # HumanML3D-normalized
            raw = gen * hml_std + hml_mean                  # -> raw
            ours = (raw - o_mean) / (o_std + 1e-8)          # -> OUR normalization
            T = batch["motion"].shape[1]
            if ours.shape[1] < T:
                ours = torch.cat([ours, torch.zeros(ours.shape[0], T - ours.shape[1], ours.shape[2])], 1)
            gen_all.append(ours[:, :T]); gt_all.append(batch["motion"].cpu().float())
            lens_all.extend(int(l) for l in lengths)

    gen = torch.cat(gen_all); gt = torch.cat(gt_all)

    # ---- score through the SAME path as all other baselines ---------------
    from baselines.baseline_utils import (load_t2m_evaluator, compute_motion_embeddings,
                                          compute_metrics, print_metrics, save_metrics)
    move_enc, mot_enc = load_t2m_evaluator(cfg, device)
    pe = compute_motion_embeddings(gen, lens_all, move_enc, mot_enc, device)
    ge = compute_motion_embeddings(gt, lens_all, move_enc, mot_enc, device)
    m = compute_metrics(pe, ge, diversity_times=min(300, len(lens_all) - 1))
    print_metrics(m, title=f"Pretrained MLD — prompt {args.prompt} ({args.split}, N={len(lens_all)})")

    gtd = float(m.get("gt_Diversity", m.get("gt_diversity", -1)))
    print(f"\n[SANITY GATE] gt_Diversity = {gtd:.3f}  -> "
          f"{'PASS (~5.5)' if 4.5 < gtd < 6.5 else 'FAIL — normalization wrong, DISCARD this number'}")
    save_metrics(m, f"research/data/pretrained_mld_{args.prompt}_{args.split}.json")


if __name__ == "__main__":
    main()
