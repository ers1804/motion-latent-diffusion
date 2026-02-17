"""
Diagnose diffusion training loss floor
========================================
Checks whether the ~0.4 MSE loss is the irreducible noise floor caused by
the VAE's stochastic encoding (rsample), NOT a training bug.

Theory:
  z_0 ~ N(mu, sigma^2 I)    [VAE posterior for one sample]
  epsilon ~ N(0, I)
  z_t = sqrt(abar_t) * z_0 + sqrt(1-abar_t) * epsilon

The denoiser predicts epsilon but doesn't know which z_0 was sampled.
The Bayes-optimal MSE loss is:

  L* = E_t [ abar_t * sigma^2 / (abar_t * sigma^2 + 1 - abar_t) ]

For sigma^2 ≈ 1 (well-regularized VAE): L* ≈ E_t[abar_t] ≈ 0.3-0.5

Usage:
  python diagnose_loss_floor.py --cfg ./configs/config_ego_motion.yaml
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
from omegaconf import OmegaConf

from mld.config import get_module_config, instantiate_from_config
from mld.data.get_data import get_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/config_ego_motion.yaml")
    parser.add_argument("--cfg_assets", type=str, default="./configs/assets.yaml")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of VAE encodings to estimate posterior stats")
    args = parser.parse_args()

    # Build config
    cfg_base = OmegaConf.load("./configs/base.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg_assets = OmegaConf.load(args.cfg_assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    vae = instantiate_from_config(cfg.model.motion_vae)
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE, map_location="cpu", weights_only=False)["state_dict"]
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("vae."):
            vae_dict[k.replace("vae.", "")] = v
    vae.load_state_dict(vae_dict, strict=True)
    vae.to(device).eval()

    # Load dataset
    datasets = get_datasets(cfg, phase="train")
    datamodule = datasets[0]
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    motion = batch["motion"].to(device)   # (B, T, 263)
    lengths = batch["length"]

    print(f"Motion shape: {motion.shape}")
    print(f"Lengths: {lengths}")
    print()

    # ── Estimate VAE posterior stats ──────────────────────────────────────
    z_samples = []
    with torch.no_grad():
        for i in range(args.n_samples):
            z, dist = vae.encode(motion, lengths)
            z_samples.append(z.squeeze(0))  # (B, 256)
            if i == 0:
                mu = dist.loc.squeeze(0)        # (B, 256) - posterior mean
                std = dist.scale.squeeze(0)     # (B, 256) - posterior std

    z_samples = torch.stack(z_samples, dim=0)  # (n_samples, B, 256)

    # Per-dimension stats
    empirical_mean = z_samples.mean(dim=0)      # (B, 256)
    empirical_std = z_samples.std(dim=0)         # (B, 256)

    print("=== VAE Posterior Statistics (per batch element) ===")
    for b in range(motion.shape[0]):
        print(f"\nSample {b}:")
        print(f"  Posterior mu:  mean={mu[b].mean():.4f}, std={mu[b].std():.4f}, "
              f"norm={mu[b].norm():.4f}")
        print(f"  Posterior std: mean={std[b].mean():.4f}, std={std[b].std():.4f}, "
              f"min={std[b].min():.4f}, max={std[b].max():.4f}")
        print(f"  Posterior var (sigma^2): mean={std[b].pow(2).mean():.4f}")
        print(f"  Empirical mean norm: {empirical_mean[b].norm():.4f}")
        print(f"  Empirical std:  mean={empirical_std[b].mean():.4f}")

    # ── Compute theoretical noise floor ───────────────────────────────────
    # Get alpha_bar schedule from noise scheduler
    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        variance_type="fixed_small",
        clip_sample=False,
        prediction_type="epsilon",
    )

    abar = noise_scheduler.alphas_cumprod.numpy()  # (1000,)
    timesteps = np.arange(1000)

    # For each batch element
    print("\n=== Theoretical Irreducible Loss Floor ===")
    print("(This is the MINIMUM possible MSE for epsilon prediction)\n")

    for b in range(motion.shape[0]):
        sigma2 = std[b].pow(2).mean().item()  # average posterior variance

        # L* = E_t [ abar_t * sigma2 / (abar_t * sigma2 + 1 - abar_t) ]
        noise_floor = np.mean(abar * sigma2 / (abar * sigma2 + 1 - abar))

        # Also compute for sigma2=1 (fully regularized)
        noise_floor_1 = np.mean(abar * 1.0 / (abar * 1.0 + 1 - abar))

        # And for using the mean (sigma2=0)
        noise_floor_0 = 0.0

        print(f"Sample {b}:")
        print(f"  sigma^2 = {sigma2:.4f}")
        print(f"  Noise floor (actual sigma^2={sigma2:.3f}): {noise_floor:.4f}")
        print(f"  Noise floor (if sigma^2=1.0):              {noise_floor_1:.4f}")
        print(f"  Noise floor (if using mean, sigma^2=0):    {noise_floor_0:.4f}")
        print()

    print("=== Average alpha_bar statistics ===")
    print(f"  E[abar_t] = {abar.mean():.4f}  (= noise floor when sigma^2=1)")
    print(f"  abar[0]   = {abar[0]:.6f}  (t=0, almost no noise)")
    print(f"  abar[499] = {abar[499]:.6f}  (t=500, mid)")
    print(f"  abar[999] = {abar[999]:.6f}  (t=999, maximum noise)")

    print("\n=== CONCLUSION ===")
    sigma2_avg = std.pow(2).mean().item()
    floor = np.mean(abar * sigma2_avg / (abar * sigma2_avg + 1 - abar))
    print(f"Your observed loss:      ~0.40")
    print(f"Theoretical noise floor: {floor:.4f}")
    if abs(floor - 0.4) < 0.1:
        print(f"\n*** Your loss IS at the irreducible floor! ***")
        print(f"The model has CONVERGED. The 0.4 loss is caused by the VAE's")
        print(f"stochastic encoding (rsample), NOT by a training bug.")
        print(f"\nFix: Use VAE posterior MEAN instead of sampling during")
        print(f"diffusion training. This makes z_0 deterministic and the")
        print(f"loss should go to ~0.")
    else:
        print(f"\nThe loss does NOT match the floor. There may be a real bug.")


if __name__ == "__main__":
    main()
