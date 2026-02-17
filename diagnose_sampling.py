"""Test DDPM sampling (1000 steps) vs DDIM (50 steps) to fix inference divergence."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np
from omegaconf import OmegaConf
from mld.config import get_module_config
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from diffusers import DDIMScheduler, DDPMScheduler

# ── load ────────────────────────────────────────────────────────────────────
cfg_base = OmegaConf.load('./configs/base.yaml')
cfg_exp  = OmegaConf.merge(cfg_base, OmegaConf.load('./configs/config_ego_motion.yaml'))
cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
cfg_assets = OmegaConf.load('./configs/assets.yaml')
cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)
device = torch.device('cuda')

datasets = get_datasets(cfg, phase='train')
dm = datasets[0]
dm.setup(stage='fit')
batch = next(iter(dm.train_dataloader()))

model = get_model(cfg, dm)
ckpt_path = 'experiments/mld/ego_motion_diffusion_overfit_ava_deterministic_z_3/checkpoints/epoch=1999.ckpt'
state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
model.load_state_dict(state_dict, strict=False)
model.to(device).eval()

motion = batch['motion'][:1].to(device)
ego    = batch['ego'][:1].to(device)
lengths = [batch['length'][0]]

with torch.no_grad():
    _, dist_gt = model.vae.encode(motion, lengths)
    z_gt = dist_gt.loc                        # (1,1,256)
z_gt_flat = z_gt.permute(1,0,2)               # (B=1,1,256) — DDIM/DDPM format
cond_emb  = model.ego_encoder(ego)

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

def mse(a, b):
    return ((a - b)**2).mean().item()


def run_ddim(num_steps, eta=0.0, seed=42):
    """Standard DDIM with configurable steps and stochasticity."""
    sched = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False,
        set_alpha_to_one=False, steps_offset=0, prediction_type="epsilon",
    )
    sched.set_timesteps(num_steps, device=device)
    torch.manual_seed(seed)
    z = torch.randn_like(z_gt_flat) * sched.init_noise_sigma
    for t in sched.timesteps:
        with torch.no_grad():
            eps_pred = model.denoiser(
                sample=z, timestep=t.unsqueeze(0),
                encoder_hidden_states=cond_emb, lengths=lengths
            )[0]
        z = sched.step(eps_pred, t, z, eta=eta).prev_sample
    return z


def run_ddpm(seed=42):
    """Full DDPM sampling with 1000 steps."""
    sched = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False,
        prediction_type="epsilon", variance_type="fixed_small",
    )
    sched.set_timesteps(1000, device=device)
    torch.manual_seed(seed)
    z = torch.randn_like(z_gt_flat)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    for i, t in enumerate(sched.timesteps):
        with torch.no_grad():
            eps_pred = model.denoiser(
                sample=z, timestep=t.unsqueeze(0),
                encoder_hidden_states=cond_emb, lengths=lengths
            )[0]
        z = sched.step(eps_pred, t, z, generator=gen).prev_sample
    return z


def run_ddim_from_partial(start_t, num_steps=50, seed=42):
    """Start DDIM from GT z_0 + noise at timestep start_t."""
    sched = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False,
        set_alpha_to_one=False, steps_offset=0, prediction_type="epsilon",
    )
    sched.set_timesteps(num_steps, device=device)
    # Keep only timesteps <= start_t
    mask = sched.timesteps <= start_t
    timesteps_to_use = sched.timesteps[mask]
    
    torch.manual_seed(seed)
    noise = torch.randn_like(z_gt_flat)
    t_tensor = torch.tensor([start_t], device=device).long()
    z = sched.add_noise(z_gt_flat, noise, t_tensor)
    
    for t in timesteps_to_use:
        with torch.no_grad():
            eps_pred = model.denoiser(
                sample=z, timestep=t.unsqueeze(0),
                encoder_hidden_states=cond_emb, lengths=lengths
            )[0]
        z = sched.step(eps_pred, t, z).prev_sample
    return z


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: DDIM 50 steps (current baseline — broken)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Test 1: DDIM 50 steps (current)")
z_pred = run_ddim(50)
print(f"  norm={z_pred.norm():.2f} (GT={z_gt_flat.norm():.2f}), MSE={mse(z_pred, z_gt_flat):.4f}, cos={cosine_sim(z_pred, z_gt_flat):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Test 2: DDIM with many more steps
# ═══════════════════════════════════════════════════════════════════════════
for nsteps in [100, 200, 500]:
    print(f"\nTest 2: DDIM {nsteps} steps")
    z_pred = run_ddim(nsteps)
    print(f"  norm={z_pred.norm():.2f} (GT={z_gt_flat.norm():.2f}), MSE={mse(z_pred, z_gt_flat):.4f}, cos={cosine_sim(z_pred, z_gt_flat):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Stochastic DDIM (η=1.0, equivalent to DDPM)
# ═══════════════════════════════════════════════════════════════════════════
for eta in [0.5, 1.0]:
    print(f"\nTest 3: DDIM 50 steps, η={eta}")
    z_pred = run_ddim(50, eta=eta)
    print(f"  norm={z_pred.norm():.2f} (GT={z_gt_flat.norm():.2f}), MSE={mse(z_pred, z_gt_flat):.4f}, cos={cosine_sim(z_pred, z_gt_flat):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Full DDPM (1000 steps)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nTest 4: DDPM 1000 steps")
z_pred = run_ddpm()
print(f"  norm={z_pred.norm():.2f} (GT={z_gt_flat.norm():.2f}), MSE={mse(z_pred, z_gt_flat):.4f}, cos={cosine_sim(z_pred, z_gt_flat):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Test 5: DDIM from partial noise (start at t=200, 400, 600 instead of 980)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nTest 5: DDIM from partial noise")
for start_t in [100, 200, 400, 600, 800]:
    z_pred = run_ddim_from_partial(start_t, num_steps=200)
    print(f"  start_t={start_t}: norm={z_pred.norm():.2f}, MSE={mse(z_pred, z_gt_flat):.4f}, cos={cosine_sim(z_pred, z_gt_flat):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Compute feature/joint error for the best method
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Test 6: Compare joints for best methods")
from mld.data.humanml.scripts.motion_process import recover_from_ric

def z_to_joints(z_latent, tag=""):
    """z in (B,1,256) format → joints"""
    # Decode
    z_vae = z_latent.permute(1, 0, 2)  # → (1, B, 256)
    feats = model.vae.decode(z_vae, lengths)  # (B, T, 263)
    
    # Denormalize
    ds = dm.train_dataset
    mean_t = torch.tensor(ds.mean, dtype=feats.dtype, device=feats.device)
    std_t  = torch.tensor(ds.std, dtype=feats.dtype, device=feats.device)
    feats_denorm = feats * std_t + mean_t
    
    # To joints
    joints = recover_from_ric(feats_denorm[0].cpu(), 22)  # (T,22,3)
    return feats_denorm[0], joints

# GT recon (VAE encode→decode)
feat_gt, joints_gt = z_to_joints(z_gt_flat, "GT recon")

# DDPM 1000
z_ddpm = run_ddpm(seed=42)
feat_ddpm, joints_ddpm = z_to_joints(z_ddpm, "DDPM")
mpjpe_ddpm = (joints_ddpm[:joints_gt.shape[0]] - joints_gt).norm(dim=-1).mean().item()

# DDIM 50
z_ddim50 = run_ddim(50)
feat_ddim, joints_ddim = z_to_joints(z_ddim50, "DDIM50")
mpjpe_ddim = (joints_ddim[:joints_gt.shape[0]] - joints_gt).norm(dim=-1).mean().item()

# DDIM from t=200
z_partial = run_ddim_from_partial(200, num_steps=200)
feat_part, joints_part = z_to_joints(z_partial, "Partial")
mpjpe_part = (joints_part[:joints_gt.shape[0]] - joints_gt).norm(dim=-1).mean().item()

print(f"  DDIM 50:       MPJPE = {mpjpe_ddim:.4f}m")
print(f"  DDPM 1000:     MPJPE = {mpjpe_ddpm:.4f}m")
print(f"  Partial t=200: MPJPE = {mpjpe_part:.4f}m")
print(f"  (VAE recon:    MPJPE ≈ 0.17m baseline)")
