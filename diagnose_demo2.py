"""
Quick sanity check: Does the trained denoiser predict noise correctly?
And does the DDIM reverse produce a good z from the trained model?
"""
import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
from collections import OrderedDict
from omegaconf import OmegaConf
from mld.config import get_module_config, instantiate_from_config
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from diffusers import DDPMScheduler, DDIMScheduler

cfg_base = OmegaConf.load("./configs/base.yaml")
cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load("./configs/config_ego_motion.yaml"))
cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
cfg_assets = OmegaConf.load("./configs/assets.yaml")
cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(cfg.SEED_VALUE)

# Load dataset
datasets = get_datasets(cfg, phase="train")
dm = datasets[0]
dm.setup(stage="fit")
batch = next(iter(dm.train_dataloader()))

# Load model
model = get_model(cfg, dm)
ckpt_path = "experiments/mld/ego_motion_diffusion_overfit_ava_deterministic_z_3/checkpoints/epoch=1999.ckpt"
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict(state_dict, strict=False)
model.to(device).eval()

motion = batch["motion"][:1].to(device)
ego = batch["ego"][:1].to(device)
lengths = [batch["length"][0]]

# Get ground truth z
with torch.no_grad():
    z_gt, dist_gt = model.vae.encode(motion, lengths)
    z_mean = dist_gt.loc  # (1, B, 256) deterministic target
    
    cond_emb = model.ego_encoder(ego)

print(f"z_mean shape: {z_mean.shape}, norm: {z_mean.norm():.2f}")
print(f"cond_emb shape: {cond_emb.shape}, norm: {cond_emb.norm():.2f}")

# ── Test single-step noise prediction ─────────────────────────────────
print("\n=== Single-step noise prediction test ===")
z_0 = z_mean.permute(1, 0, 2)  # (B, 1, 256) for scheduler
noise_sched = model.noise_scheduler

for t_val in [0, 100, 500, 999]:
    t = torch.tensor([t_val], device=device).long()
    noise = torch.randn_like(z_0)
    z_t = noise_sched.add_noise(z_0, noise, t)
    
    with torch.no_grad():
        noise_pred = model.denoiser(
            sample=z_t, timestep=t,
            encoder_hidden_states=cond_emb,
            lengths=lengths
        )[0]
    
    mse = ((noise_pred - noise) ** 2).mean().item()
    cos = torch.nn.functional.cosine_similarity(
        noise_pred.flatten(), noise.flatten(), dim=0
    ).item()
    print(f"  t={t_val:4d}: noise_pred MSE={mse:.4f}, cosine={cos:.4f}, "
          f"pred_norm={noise_pred.norm():.2f}, noise_norm={noise.norm():.2f}")

# ── Test full DDIM reverse ────────────────────────────────────────────
print("\n=== Full DDIM reverse (50 steps) ===")
with torch.no_grad():
    z_pred = model._diffusion_reverse(cond_emb, lengths)  # (1, B, 256)

z_pred_flat = z_pred.permute(1, 0, 2)  # (B, 1, 256)
z_gt_flat = z_mean.permute(1, 0, 2)    # (B, 1, 256)

mse = ((z_pred_flat - z_gt_flat) ** 2).mean().item()
cos = torch.nn.functional.cosine_similarity(
    z_pred_flat.flatten(), z_gt_flat.flatten(), dim=0
).item()
print(f"  z_pred norm: {z_pred_flat.norm():.2f}")
print(f"  z_gt norm:   {z_gt_flat.norm():.2f}")
print(f"  MSE:         {mse:.4f}")
print(f"  Cosine:      {cos:.4f}")

# ── Decode both and compare feature-level ─────────────────────────────
print("\n=== Feature-level comparison ===")
with torch.no_grad():
    feats_pred = model.vae.decode(z_pred, lengths)
    feats_gt_recon = model.vae.decode(z_mean, lengths)

mse_feats = ((feats_pred - feats_gt_recon) ** 2).mean().item()
print(f"  Predicted feats range:     [{feats_pred.min():.3f}, {feats_pred.max():.3f}]")
print(f"  GT-recon feats range:      [{feats_gt_recon.min():.3f}, {feats_gt_recon.max():.3f}]")
print(f"  Feature MSE (pred vs GT):  {mse_feats:.4f}")

# Denormalize and compare
mean_npy = np.load(os.path.join(cfg.DATASET.EGOMOTION.MEAN_STD_PATH, "Mean.npy"))
std_npy = np.load(os.path.join(cfg.DATASET.EGOMOTION.MEAN_STD_PATH, "Std.npy"))

feats_pred_np = feats_pred[0].cpu().numpy() * std_npy + mean_npy
feats_gt_np = feats_gt_recon[0].cpu().numpy() * std_npy + mean_npy
feats_input_np = motion[0].cpu().numpy() * std_npy + mean_npy

print(f"\n  Denorm pred range:  [{feats_pred_np.min():.3f}, {feats_pred_np.max():.3f}]")
print(f"  Denorm GT range:    [{feats_gt_np.min():.3f}, {feats_gt_np.max():.3f}]")
print(f"  Denorm input range: [{feats_input_np.min():.3f}, {feats_input_np.max():.3f}]")

# ── Joint-level comparison ────────────────────────────────────────────
print("\n=== Joint-level comparison ===")
from mld.data.humanml.scripts.motion_process import recover_from_ric

joints_pred = recover_from_ric(torch.tensor(feats_pred_np).unsqueeze(0), 22)[0].numpy()
joints_gt = recover_from_ric(torch.tensor(feats_gt_np).unsqueeze(0), 22)[0].numpy()
joints_input = recover_from_ric(torch.tensor(feats_input_np).unsqueeze(0), 22)[0].numpy()

print(f"  Pred joints range:  [{joints_pred.min():.3f}, {joints_pred.max():.3f}]")
print(f"  GT joints range:    [{joints_gt.min():.3f}, {joints_gt.max():.3f}]")
print(f"  Input joints range: [{joints_input.min():.3f}, {joints_input.max():.3f}]")

# Per-joint MPJPE 
mpjpe_pred_vs_input = np.sqrt(((joints_pred[:len(joints_input)] - joints_input[:len(joints_pred)]) ** 2).sum(-1)).mean()
mpjpe_gt_vs_input = np.sqrt(((joints_gt[:len(joints_input)] - joints_input[:len(joints_gt)]) ** 2).sum(-1)).mean()
print(f"\n  MPJPE (pred vs input):    {mpjpe_pred_vs_input:.4f} m")
print(f"  MPJPE (VAE recon vs input): {mpjpe_gt_vs_input:.4f} m")
print(f"  (VAE recon is the best possible result)")

# ── Check if model is actually using the right checkpoint ─────────────
print(f"\n=== Using checkpoint: {ckpt_path} ===")
print(f"  Training was for {cfg.TRAIN.END_EPOCH} epochs")
print(f"  Training loss at end: ~0.09")
