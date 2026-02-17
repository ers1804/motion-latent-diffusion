"""
Standalone Denoiser Convergence Test
=====================================
Bypasses the entire MLD/Lightning pipeline and tests ONLY:
  Can the SkipTransformerEncoder denoiser memorize a single z_0?

Tests multiple configurations to isolate the issue:
  1. B=1 (your current setup) — expected: slow/stuck
  2. B=32 (same z_0, different t and epsilon) — expected: converges fast

Usage:
  python test_denoiser_standalone.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import DDPMScheduler
import sys
sys.path.insert(0, '.')

from mld.models.architectures.mld_denoiser import MldDenoiser
from omegaconf import OmegaConf


def create_denoiser(device):
    ablation = OmegaConf.create({
        "SKIP_CONNECT": True,
        "PE_TYPE": "mld",
        "DIFF_PE_TYPE": "mld",
        "PREDICT_EPSILON": True,
        "VAE_TYPE": "mld",
        "MLP_DIST": False,
        "IS_DIST": False,
    })

    denoiser = MldDenoiser(
        ablation=ablation,
        nfeats=263,
        condition="ego",
        latent_dim=[1, 256],
        ff_size=1024,
        num_layers=9,
        num_heads=4,
        dropout=0.0,
        activation="gelu",
        text_encoded_dim=256,
        guidance_scale=1.0,
        guidance_uncondp=0.0,
    ).to(device)
    return denoiser


def run_test(batch_size, lr, n_steps, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    denoiser = create_denoiser(device)
    n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        variance_type="fixed_small",
        clip_sample=False,
        prediction_type="epsilon",
    )

    # Fixed z_0 and cond_emb (same sample, replicated to batch)
    z_0_single = torch.randn(1, 1, 256, device=device) * 1.5
    cond_single = torch.randn(1, 1, 256, device=device)

    # Replicate to batch — same z_0, different t and epsilon per element
    z_0 = z_0_single.expand(batch_size, -1, -1)
    cond_emb = cond_single.expand(batch_size, -1, -1)

    optimizer = AdamW(denoiser.parameters(), lr=lr, weight_decay=0.0)

    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  B={batch_size}, LR={lr}, steps={n_steps}, params={n_params:,}")
    print(f"  z_0 norm: {z_0_single.norm():.2f}, cond norm: {cond_single.norm():.2f}")
    print(f"{'='*60}")
    print(f"{'Step':>5} │ {'Loss':>10} │ {'Grad Norm':>10}")
    print(f"{'─'*5}─┼─{'─'*10}─┼─{'─'*10}")

    denoiser.train()
    for step in range(1, n_steps + 1):
        latents = z_0.clone()
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=cond_emb,
            lengths=None,
        )[0]

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()

        grad_norm = 0.0
        for p in denoiser.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        if step <= 5 or step % (n_steps // 20) == 0 or step == n_steps:
            print(f"{step:5d} │ {loss.item():10.6f} │ {grad_norm:10.4f}")

    print(f"\nFinal loss: {loss.item():.6f}")
    return loss.item()


if __name__ == "__main__":
    print("Testing if gradient variance from B=1 is the root cause\n")

    # Test 1: Your current setup (B=1, LR=1e-4)
    loss_1 = run_test(batch_size=1, lr=1e-4, n_steps=2000, label="B=1, LR=1e-4 (your setup)")

    # Test 2: B=32 (same z_0, but 32 different timesteps per step)
    loss_2 = run_test(batch_size=32, lr=1e-4, n_steps=2000, label="B=32, LR=1e-4 (reduced variance)")

    # Test 3: B=1, higher LR
    loss_3 = run_test(batch_size=1, lr=1e-3, n_steps=2000, label="B=1, LR=1e-3 (higher LR)")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  B=1,  LR=1e-4: {loss_1:.4f}  {'✓' if loss_1 < 0.05 else '✗'}")
    print(f"  B=32, LR=1e-4: {loss_2:.4f}  {'✓' if loss_2 < 0.05 else '✗'}")
    print(f"  B=1,  LR=1e-3: {loss_3:.4f}  {'✓' if loss_3 < 0.05 else '✗'}")
    print()
    if loss_2 < 0.1 and loss_1 > 0.3:
        print("DIAGNOSIS: Gradient variance from B=1 is the root cause.")
        print("FIX: Increase effective batch size during overfit training.")
        print("     Set BATCH_SIZE=32 and make the dataset repeat the sample.")
    elif loss_3 < 0.1:
        print("DIAGNOSIS: Learning rate too low for B=1.")
        print("FIX: Increase LR to 1e-3 or higher.")
    else:
        print("DIAGNOSIS: Deeper architecture issue — needs investigation.")
