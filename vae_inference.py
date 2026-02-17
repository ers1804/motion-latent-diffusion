"""
VAE Inference Script
====================
Load pretrained VAE weights and run inference on a given motion sequence.
Computes reconstruction error between input and VAE output.

Usage examples:
  # Using a full MLD checkpoint (VAE weights extracted automatically):
  python vae_inference.py \
      --cfg ./configs/config_vae_humanml3d.yaml \
      --checkpoint ./checkpoints/mld_humanml3d_checkpoint/1222_mld_humanml3d_FID041.ckpt \
      --motion_npy /path/to/motion.npy

  # Using a standalone VAE checkpoint:
  python vae_inference.py \
      --cfg ./configs/config_vae_humanml3d.yaml \
      --checkpoint /path/to/vae_only.ckpt \
      --vae_only \
      --motion_npy /path/to/motion.npy

  # Process an entire folder of .npy files:
  python vae_inference.py \
      --cfg ./configs/config_vae_humanml3d.yaml \
      --checkpoint ./checkpoints/mld_humanml3d_checkpoint/1222_mld_humanml3d_FID041.ckpt \
      --motion_dir /path/to/motion_folder/

  # Use mean (no sampling) for deterministic output:
  python vae_inference.py \
      --cfg ./configs/config_vae_humanml3d.yaml \
      --checkpoint ./checkpoints/mld_humanml3d_checkpoint/1222_mld_humanml3d_FID041.ckpt \
      --motion_npy /path/to/motion.npy \
      --use_mean

  # Save reconstructed motion to a specific output directory:
  python vae_inference.py \
      --cfg ./configs/config_vae_humanml3d.yaml \
      --checkpoint ./checkpoints/mld_humanml3d_checkpoint/1222_mld_humanml3d_FID041.ckpt \
      --motion_npy /path/to/motion.npy \
      --output_dir ./outputs/vae_recon/
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import json

from mld.config import get_module_config, instantiate_from_config
from mld.data.humanml.scripts.motion_process import recover_from_ric


def build_cfg(args):
    """Build the OmegaConf config the same way train.py / test.py do."""
    cfg_base = OmegaConf.load("./configs/base.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg_assets = OmegaConf.load(args.cfg_assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)
    return cfg


def build_vae(cfg):
    """Instantiate the VAE from config (no dataset or full model needed)."""
    # The motion_vae config needs DATASET.NFEATS and TRAIN.ABLATION resolved.
    # For HumanML3D the feature dim is 263.
    nfeats = cfg.model.motion_vae.params.get("nfeats", 263)
    print(f"[VAE] nfeats = {nfeats}")
    vae = instantiate_from_config(cfg.model.motion_vae)
    return vae


def load_vae_weights(vae, checkpoint_path, vae_only=False):
    """
    Load VAE weights from a checkpoint.

    If vae_only=False (default), the checkpoint is assumed to be a full MLD
    model checkpoint and VAE weights are extracted from keys prefixed with 'vae.'.

    If vae_only=True, the checkpoint is assumed to contain only VAE weights
    directly (or a full checkpoint whose state_dict maps directly to the VAE).
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle Lightning checkpoint wrapping
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if vae_only:
        # Try loading directly; if it fails, try stripping 'vae.' prefix
        try:
            vae.load_state_dict(state_dict, strict=True)
            print("[VAE] Loaded standalone VAE weights (strict).")
        except RuntimeError:
            vae_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("vae."):
                    vae_dict[k.replace("vae.", "", 1)] = v
            vae.load_state_dict(vae_dict, strict=True)
            print("[VAE] Loaded VAE weights after stripping 'vae.' prefix.")
    else:
        # Extract vae.* keys from full model checkpoint
        vae_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("vae."):
                vae_dict[k.replace("vae.", "", 1)] = v
        if len(vae_dict) == 0:
            raise RuntimeError(
                "No keys starting with 'vae.' found in checkpoint. "
                "If this is a standalone VAE checkpoint, pass --vae_only."
            )
        vae.load_state_dict(vae_dict, strict=True)
        print(f"[VAE] Loaded {len(vae_dict)} VAE parameters from full model checkpoint.")

    return vae


def load_mean_std(cfg):
    """Load the mean/std normalization arrays used during training."""
    dataset_name = cfg.TRAIN.DATASETS[0].lower()

    if dataset_name in ["humanml3d", "kit"]:
        name = "t2m" if dataset_name == "humanml3d" else dataset_name
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean = np.load(os.path.join(data_root, "Mean.npy"))
        std = np.load(os.path.join(data_root, "Std.npy"))
    elif dataset_name == "egomotion":
        if hasattr(cfg.DATASET.EGOMOTION, "MEAN_STD_PATH") and cfg.DATASET.EGOMOTION.MEAN_STD_PATH:
            p = cfg.DATASET.EGOMOTION.MEAN_STD_PATH
        elif hasattr(cfg.DATASET, "HUMANML3D"):
            p = cfg.DATASET.HUMANML3D.ROOT
        else:
            return None, None
        mean = np.load(os.path.join(p, "Mean.npy"))
        std = np.load(os.path.join(p, "Std.npy"))
    else:
        print(f"[WARN] Unknown dataset '{dataset_name}', skipping mean/std loading.")
        return None, None

    return mean, std


def feats_to_joints(feats, mean, std, njoints=22):
    """
    Convert 263-dim HumanML3D features to joint positions [T, J, 3].

    Applies denormalization (mean/std), then recover_from_ric to get
    [T, njoints, 3] joint positions in the HumanML3D coordinate frame.
    """
    feats_tensor = torch.from_numpy(feats).float().unsqueeze(0)  # [1, T, 263]
    if mean is not None:
        mean_t = torch.tensor(mean).float()
        std_t = torch.tensor(std).float()
        feats_tensor = feats_tensor * std_t + mean_t
    joints = recover_from_ric(feats_tensor, njoints)  # [1, T, J, 3]
    return joints.squeeze(0).numpy()  # [T, J, 3]


def normalize(motion, mean, std):
    """Normalize motion features with dataset mean/std."""
    return (motion - mean) / std


def denormalize(motion, mean, std):
    """Denormalize motion features back to original scale."""
    return motion * std + mean


@torch.no_grad()
def vae_reconstruct(vae, motion_tensor, lengths, use_mean=False):
    """
    Run a motion sequence through the VAE (encode then decode).

    Args:
        vae: The MldVae model.
        motion_tensor: [B, T, D] float tensor of normalized motion features.
        lengths: list of int, actual lengths per batch element.
        use_mean: if True, use the mean of the latent distribution (deterministic);
                  otherwise sample from the distribution.

    Returns:
        recon: [B, T, D] reconstructed motion tensor.
        z: latent code.
        dist: the latent distribution object.
    """
    z, dist = vae.encode(motion_tensor, lengths)

    if use_mean:
        z = dist.mean  # deterministic

    recon = vae.decode(z, lengths)
    return recon, z, dist


def compute_metrics(original, reconstructed, lengths=None):
    """
    Compute reconstruction error metrics.

    Args:
        original: [B, T, D] numpy array.
        reconstructed: [B, T, D] numpy array.
        lengths: optional list of actual sequence lengths (to mask padding).

    Returns:
        dict of metric name -> value.
    """
    metrics = {}

    if lengths is not None:
        # Build a mask to ignore padding frames
        B, T, D = original.shape
        mask = np.zeros((B, T), dtype=bool)
        for i, l in enumerate(lengths):
            mask[i, :l] = True
        mask_3d = mask[..., np.newaxis]  # [B, T, 1]

        diff = (original - reconstructed) * mask_3d
        n_valid = mask_3d.sum() * D  # total valid elements

        metrics["MSE"] = float((diff ** 2).sum() / n_valid)
        metrics["MAE"] = float(np.abs(diff).sum() / n_valid)
        metrics["RMSE"] = float(np.sqrt(metrics["MSE"]))

        # Per-frame MSE (averaged over valid frames)
        per_frame_mse = (diff ** 2).sum(axis=-1)  # [B, T]
        metrics["Per-frame MSE (mean)"] = float(per_frame_mse[mask].mean())
        metrics["Per-frame MSE (std)"] = float(per_frame_mse[mask].std())
    else:
        diff = original - reconstructed
        metrics["MSE"] = float((diff ** 2).mean())
        metrics["MAE"] = float(np.abs(diff).mean())
        metrics["RMSE"] = float(np.sqrt(metrics["MSE"]))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="VAE Inference & Reconstruction Error")
    parser.add_argument("--cfg", type=str, default="./configs/config_vae_humanml3d.yaml",
                        help="Path to experiment config yaml.")
    parser.add_argument("--cfg_assets", type=str, default="./configs/assets.yaml",
                        help="Path to assets config yaml.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt file containing VAE weights.")
    parser.add_argument("--vae_only", action="store_true",
                        help="If set, the checkpoint contains only VAE weights (no full model).")
    parser.add_argument("--motion_npy", type=str, default=None,
                        help="Path to a single .npy motion file [T, D].")
    parser.add_argument("--motion_dir", type=str, default=None,
                        help="Path to a directory of .npy motion files.")
    parser.add_argument("--already_normalized", action="store_true",
                        help="Set if the input .npy is already normalized with dataset mean/std.")
    parser.add_argument("--use_mean", action="store_true",
                        help="Use latent mean instead of sampling (deterministic reconstruction).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save reconstructed .npy files. If not set, no files are saved.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on.")
    parser.add_argument("--render", action="store_true",
                        help="Convert original & reconstructed motions to joints and save "
                             "side-by-side .npy files ready for Blender rendering.")
    parser.add_argument("--render_mode", type=str, default="video",
                        choices=["video", "sequence", "frame"],
                        help="Render mode passed to render.py (default: video).")
    parser.add_argument("--side_by_side_offset", type=float, default=2.0,
                        help="X-axis offset (meters) between original and reconstructed skeletons "
                             "in side-by-side rendering (default: 2.0).")
    args = parser.parse_args()

    if args.motion_npy is None and args.motion_dir is None:
        parser.error("Provide either --motion_npy or --motion_dir.")

    if args.render and args.output_dir is None:
        args.output_dir = "./outputs/vae_recon/"
        print(f"[Render] --render requires output files; setting --output_dir={args.output_dir}")

    # ---- Build config & model ----
    cfg = build_cfg(args)
    vae = build_vae(cfg)
    vae = load_vae_weights(vae, args.checkpoint, vae_only=args.vae_only)
    vae = vae.to(args.device).eval()
    print(f"[VAE] Model on {args.device}, eval mode.")

    # ---- Load normalization stats ----
    mean, std = load_mean_std(cfg)
    if mean is not None:
        print(f"[Data] Mean shape: {mean.shape}, Std shape: {std.shape}")
    else:
        print("[Data] No mean/std loaded — assuming input is already normalized.")
        args.already_normalized = True

    # ---- Collect motion file paths ----
    motion_paths = []
    if args.motion_npy:
        motion_paths.append(args.motion_npy)
    if args.motion_dir:
        for f in sorted(os.listdir(args.motion_dir)):
            if f.endswith(".npy") or f.endswith(".json"):
                motion_paths.append(os.path.join(args.motion_dir, f))

    if len(motion_paths) == 0:
        print("No .npy or .json files found. Exiting.")
        return

    # ---- Optional output directory ----
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- Run inference on each file ----
    all_metrics = {}
    for path in motion_paths:
        name = os.path.basename(path)
        print(f"\n{'='*60}")
        print(f"Processing: {name}")

        if path.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
            if "vectors_263" not in data:
                print(f"  [SKIP] No 'vectors_263' key found in JSON.")
                continue
            motion = np.array(data["vectors_263"])
            print(f"  Loaded motion from JSON with shape {motion.shape}.")
        else:
            motion = np.load(path)  # expected shape [T, D]
        if motion.ndim == 1:
            print(f"  [SKIP] Unexpected 1-D array with shape {motion.shape}")
            continue
        if motion.ndim == 3:
            # Already batched [B, T, D] — take first element
            print(f"  Input shape [B,T,D] = {motion.shape}, using first element.")
            motion = motion[0]

        T, D = motion.shape
        print(f"  Frames: {T}, Features: {D}")

        # Normalize if needed
        if not args.already_normalized and mean is not None:
            motion_norm = normalize(motion, mean, std)
        else:
            motion_norm = motion.copy()

        # To tensor [1, T, D]
        motion_tensor = torch.from_numpy(motion_norm).float().unsqueeze(0).to(args.device)
        lengths = [T]

        # Reconstruct
        recon_tensor, z, dist = vae_reconstruct(vae, motion_tensor, lengths, use_mean=args.use_mean)

        recon_norm = recon_tensor.cpu().numpy()[0]  # [T, D]

        # ---- Compute errors in normalized space ----
        metrics_norm = compute_metrics(
            motion_norm[np.newaxis], recon_norm[np.newaxis], lengths=lengths
        )
        print(f"  Reconstruction error (normalized feature space):")
        for k, v in metrics_norm.items():
            print(f"    {k}: {v:.6f}")

        # ---- Compute errors in original (denormalized) space ----
        if mean is not None and not args.already_normalized:
            recon_denorm = denormalize(recon_norm, mean, std)
            metrics_orig = compute_metrics(
                motion[np.newaxis], recon_denorm[np.newaxis], lengths=lengths
            )
            print(f"  Reconstruction error (original feature space):")
            for k, v in metrics_orig.items():
                print(f"    {k}: {v:.6f}")
        else:
            recon_denorm = recon_norm

        # ---- Latent stats ----
        print(f"  Latent shape: {z.shape}")
        print(f"  Latent mean: {dist.mean.mean().item():.4f}, "
              f"Latent std: {dist.stddev.mean().item():.4f}")

        # ---- Save reconstructed motion ----
        if args.output_dir:
            stem = os.path.splitext(name)[0]
            out_path = os.path.join(args.output_dir, f"{stem}_recon.npy")
            np.save(out_path, recon_denorm)
            print(f"  Saved reconstruction to {out_path}")

        # ---- Render: convert to joints and save side-by-side ----
        if args.render and args.output_dir:
            stem = os.path.splitext(name)[0]
            render_dir = os.path.join(args.output_dir, "render")
            os.makedirs(render_dir, exist_ok=True)

            # Get the denormalized features for both original and reconstruction
            if mean is not None and not args.already_normalized:
                orig_feats_denorm = motion        # already in original space
                recon_feats_denorm = recon_denorm  # denormalized above
            else:
                # If already normalized, we still need to denormalize for joint recovery
                # recover_from_ric expects denormalized features
                orig_feats_denorm = denormalize(motion, mean, std) if mean is not None else motion
                recon_feats_denorm = denormalize(recon_norm, mean, std) if mean is not None else recon_norm

            # Convert 263-dim features -> [T, 22, 3] joints
            # feats_to_joints handles denormalization internally from normalized input
            joints_orig = feats_to_joints(motion_norm, mean, std, njoints=22)
            joints_recon = feats_to_joints(recon_norm, mean, std, njoints=22)
            print(f"  Joints original shape:  {joints_orig.shape}")
            print(f"  Joints recon shape:     {joints_recon.shape}")

            # Save individual joint files
            orig_joints_path = os.path.join(render_dir, f"{stem}_orig.npy")
            recon_joints_path = os.path.join(render_dir, f"{stem}_recon.npy")
            np.save(orig_joints_path, joints_orig)
            np.save(recon_joints_path, joints_recon)

            # Create side-by-side: offset reconstruction along X axis
            joints_recon_offset = joints_recon.copy()
            joints_recon_offset[..., 0] += args.side_by_side_offset
            # Stack along the joints dimension: [T, 2*J, 3]
            # But the renderer expects [T, J, 3] per skeleton.
            # Instead we save a combined npy with 44 "joints" — two skeletons.
            joints_combined = np.concatenate([joints_orig, joints_recon_offset], axis=1)
            combined_path = os.path.join(render_dir, f"{stem}_sidebyside.npy")
            np.save(combined_path, joints_combined)

            print(f"  Saved joint files for rendering:")
            print(f"    Original:     {orig_joints_path}")
            print(f"    Reconstructed: {recon_joints_path}")
            print(f"    Side-by-side:  {combined_path}")
            print(f"")
            print(f"  To render individually with Blender:")
            print(f"    blender --background --python render.py -- \\")
            print(f"      --cfg=./configs/render_mld.yaml \\")
            print(f"      --dir={render_dir} \\")
            print(f"      --mode={args.render_mode} \\")
            print(f"      --joint_type=humanml3d")

        all_metrics[name] = metrics_norm

    # ---- Summary ----
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("Summary across all files:")
        agg = {}
        for name, m in all_metrics.items():
            for k, v in m.items():
                agg.setdefault(k, []).append(v)
        for k, vals in agg.items():
            arr = np.array(vals)
            print(f"  {k}: mean={arr.mean():.6f}, std={arr.std():.6f}, "
                  f"min={arr.min():.6f}, max={arr.max():.6f}")


if __name__ == "__main__":
    main()
