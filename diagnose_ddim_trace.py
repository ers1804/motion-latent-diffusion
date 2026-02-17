"""Trace DDIM reverse step-by-step to find where error accumulates."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np
from omegaconf import OmegaConf
from mld.config import get_module_config
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from diffusers import DDIMScheduler

# ── load ────────────────────────────────────────────────────────────────────
cfg_base = OmegaConf.load('./configs/base.yaml')
cfg_exp  = OmegaConf.merge(cfg_base, OmegaConf.load('./configs/config_ego_motion.yaml'))
cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
cfg_assets = OmegaConf.load('./configs/assets.yaml')
cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)
device = torch.device('cuda')
torch.manual_seed(42)

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

z_gt_flat = z_gt.permute(1,0,2)               # (B=1, 1, 256) — what DDIM produces
cond_emb  = model.ego_encoder(ego)

# ── Build DDIM scheduler ────────────────────────────────────────────────────
scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=0,
    prediction_type="epsilon",
)
scheduler.set_timesteps(50, device=device)
timesteps = scheduler.timesteps              # e.g. [999, 979, …, 19, 0] or similar

print(f"DDIM timesteps ({len(timesteps)}):")
print(timesteps.cpu().numpy())

# ── DDIM reverse, step-by-step ──────────────────────────────────────────────
torch.manual_seed(42)
latents = torch.randn_like(z_gt_flat) * scheduler.init_noise_sigma
print(f"\nInit latents: norm={latents.norm():.4f}")
print(f"GT z:         norm={z_gt_flat.norm():.4f}")

print(f"\n{'step':>4} | {'t':>4} | {'lat_norm':>8} | {'MSE_to_gt':>9} | {'cos_to_gt':>9} | {'eps_norm':>8}")
print("-" * 65)

for i, t in enumerate(timesteps):
    with torch.no_grad():
        noise_pred = model.denoiser(
            sample=latents, timestep=t.unsqueeze(0),
            encoder_hidden_states=cond_emb, lengths=lengths
        )[0]

    # DDIM step
    output = scheduler.step(noise_pred, t, latents)
    latents = output.prev_sample
    
    # Also extract predicted x0 from this step
    pred_x0 = output.pred_original_sample if hasattr(output, 'pred_original_sample') else None

    lat_norm = latents.norm().item()
    mse_gt   = ((latents - z_gt_flat)**2).mean().item()
    cos_gt   = torch.nn.functional.cosine_similarity(
        latents.flatten(), z_gt_flat.flatten(), dim=0).item()

    if i % 5 == 0 or i >= len(timesteps) - 5:
        extra = ""
        if pred_x0 is not None:
            x0_mse = ((pred_x0 - z_gt_flat)**2).mean().item()
            x0_cos = torch.nn.functional.cosine_similarity(
                pred_x0.flatten(), z_gt_flat.flatten(), dim=0).item()
            extra = f" | x0_MSE={x0_mse:.4f}, x0_cos={x0_cos:.4f}"
        print(f"{i:4d} | {t.item():4d} | {lat_norm:8.2f} | {mse_gt:9.4f} | {cos_gt:9.4f} | {noise_pred.norm():8.2f}{extra}")

print(f"\nFinal z_pred norm: {latents.norm():.4f}")
print(f"GT z norm:         {z_gt_flat.norm():.4f}")
print(f"MSE:               {((latents - z_gt_flat)**2).mean():.4f}")
print(f"Cosine:            {torch.nn.functional.cosine_similarity(latents.flatten(), z_gt_flat.flatten(), dim=0):.4f}")

# ── Now try: what if we START from GT z_0 (no noise) and verify the model's
#    single-step denoise at the ACTUAL DDIM timesteps? ────────────────────────
print("\n\n=== Single-step noise prediction at actual DDIM timesteps ===")
print(f"{'t':>4} | {'abar':>8} | {'MSE':>8} | {'cos':>7}")
print("-" * 40)
for t_val in timesteps[::5].tolist() + timesteps[-3:].tolist():
    t_val = int(t_val)
    mses, coss = [], []
    for _ in range(20):
        t = torch.tensor([t_val], device=device).long()
        noise = torch.randn_like(z_gt_flat)
        z_t = scheduler.add_noise(z_gt_flat, noise, t)
        with torch.no_grad():
            noise_pred = model.denoiser(
                sample=z_t, timestep=t,
                encoder_hidden_states=cond_emb, lengths=lengths
            )[0]
        mses.append(((noise_pred - noise)**2).mean().item())
        coss.append(torch.nn.functional.cosine_similarity(
            noise_pred.flatten(), noise.flatten(), dim=0).item())
    abar = scheduler.alphas_cumprod[t_val].item()
    print(f"{t_val:4d} | {abar:8.4f} | {np.mean(mses):8.4f} | {np.mean(coss):7.4f}")
