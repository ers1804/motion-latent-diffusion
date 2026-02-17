"""Quick standalone test: verify sample/x0 prediction works end-to-end."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from mld.config import get_module_config
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model

# ── load with the new config (PREDICT_EPSILON: False) ──────────────────────
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
model.to(device)
model.train()

# Verify flags
print(f"predict_epsilon: {model.predict_epsilon}")
print(f"noise_scheduler prediction_type: {model.noise_scheduler.config.prediction_type}")
print(f"scheduler prediction_type: {model.scheduler.config.prediction_type}")

motion = batch['motion'].to(device)
ego    = batch['ego'].to(device)
lengths = batch['length']

# ── Manual forward pass with sample prediction ─────────────────────────────
with torch.no_grad():
    _, dist = model.vae.encode(motion, lengths)
    z_gt = dist.loc  # (1, B, 256)
    cond_emb = model.ego_encoder(ego)

z_gt_flat = z_gt.permute(1, 0, 2)  # (B, 1, 256)

# Add noise at various timesteps and predict x0
print(f"\nz_gt norm: {z_gt_flat[0].norm():.4f}")
print(f"\nSample (x0) prediction test:")
print(f"{'t':>5} | {'x0_pred_MSE':>11} | {'x0_pred_cos':>11} | {'x0_pred_norm':>12}")
print("-" * 55)

for t_val in [0, 50, 100, 200, 500, 800, 950, 999]:
    t = torch.full((z_gt_flat.shape[0],), t_val, device=device).long()
    noise = torch.randn_like(z_gt_flat)
    z_t = model.noise_scheduler.add_noise(z_gt_flat, noise, t)
    
    # Model predicts x0 directly  
    x0_pred = model.denoiser(
        sample=z_t, timestep=t,
        encoder_hidden_states=cond_emb, lengths=lengths,
        return_dict=False,
    )[0]
    
    mse = F.mse_loss(x0_pred, z_gt_flat).item()
    cos = F.cosine_similarity(x0_pred[0].flatten(), z_gt_flat[0].flatten(), dim=0).item()
    norm = x0_pred[0].norm().item()
    
    print(f"{t_val:5d} | {mse:11.4f} | {cos:11.4f} | {norm:12.2f}")

# ── Quick 100-step overfit training to verify convergence ──────────────────
print("\n\nQuick overfit training (100 steps)...")
model.train()
optim = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-4)

for step in range(100):
    with torch.no_grad():
        _, dist = model.vae.encode(motion, lengths)
        z_gt = dist.loc
        cond_emb = model.ego_encoder(ego)
    
    z_gt_flat = z_gt.permute(1, 0, 2)
    noise = torch.randn_like(z_gt_flat)
    t = torch.randint(0, 1000, (z_gt_flat.shape[0],), device=device).long()
    z_t = model.noise_scheduler.add_noise(z_gt_flat, noise, t)
    
    x0_pred = model.denoiser(
        sample=z_t, timestep=t,
        encoder_hidden_states=cond_emb, lengths=lengths,
        return_dict=False,
    )[0]
    
    loss = F.mse_loss(x0_pred, z_gt_flat)
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if step % 20 == 0:
        print(f"  step {step:3d}: loss = {loss.item():.6f}")

# ── After 100 steps, test DDIM reverse ─────────────────────────────────────
print("\nTesting DDIM reverse after 100 training steps...")
model.eval()
from diffusers import DDIMScheduler
sched = DDIMScheduler(
    num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
    beta_schedule="scaled_linear", clip_sample=False,
    set_alpha_to_one=False, steps_offset=0, prediction_type="sample",
)
sched.set_timesteps(50, device=device)

with torch.no_grad():
    cond_emb = model.ego_encoder(ego[:1])
    
torch.manual_seed(42)
z = torch.randn(1, 1, 256, device=device) * sched.init_noise_sigma
for t in sched.timesteps:
    with torch.no_grad():
        x0_pred = model.denoiser(
            sample=z, timestep=t.unsqueeze(0),
            encoder_hidden_states=cond_emb, lengths=[lengths[0]],
        )[0]
    z = sched.step(x0_pred, t, z).prev_sample

z_target = z_gt.permute(1, 0, 2)[:1]
cos = F.cosine_similarity(z.flatten(), z_target.flatten(), dim=0).item()
mse_val = F.mse_loss(z, z_target).item()
print(f"DDIM result: norm={z.norm():.2f} (GT={z_target.norm():.2f}), MSE={mse_val:.4f}, cos={cos:.4f}")
