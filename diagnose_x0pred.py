"""Full pipeline diagnostic for x0pred model — check every stage."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from mld.config import get_module_config
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.data.humanml.scripts.motion_process import recover_from_ric

device = torch.device('cuda')
torch.manual_seed(42)

# ── 1. Load config + model ─────────────────────────────────────────────────
print("=" * 70)
print("STAGE 1: Load model")
cfg_base = OmegaConf.load('./configs/base.yaml')
cfg_exp  = OmegaConf.merge(cfg_base, OmegaConf.load('./configs/config_ego_motion.yaml'))
cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
cfg_assets = OmegaConf.load('./configs/assets.yaml')
cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

datasets = get_datasets(cfg, phase='train')
dm = datasets[0]
dm.setup(stage='fit')

model = get_model(cfg, dm)
ckpt_path = 'experiments/mld/ego_motion_diffusion_overfit_ava_x0pred/checkpoints/epoch=1199.ckpt'
state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
result = model.load_state_dict(state_dict, strict=False)
if result is not None:
    missing, unexpected = result
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  MISSING: {missing[:10]}")
else:
    print(f"  Loaded (no missing/unexpected info returned)")
model.to(device).eval()

print(f"  predict_epsilon: {model.predict_epsilon}")
print(f"  scheduler prediction_type: {model.scheduler.config.prediction_type}")
print(f"  noise_scheduler prediction_type: {model.noise_scheduler.config.prediction_type}")

# ── 2. Get training data ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 2: Load training data")
batch = next(iter(dm.train_dataloader()))
motion = batch['motion'][:1].to(device)
ego    = batch['ego'][:1].to(device)
lengths = [batch['length'][0]]
print(f"  motion shape: {motion.shape}, ego shape: {ego.shape}, length: {lengths}")

# ── 3. VAE encode (get ground truth z) ─────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 3: VAE encode → z_gt")
with torch.no_grad():
    z_gt_seq, dist = model.vae.encode(motion, lengths)
    z_gt = dist.loc  # (1, B, 256)
z_gt_flat = z_gt.permute(1, 0, 2)  # (B, 1, 256)
print(f"  z_gt shape: {z_gt_flat.shape}")
print(f"  z_gt norm: {z_gt_flat.norm():.4f}")
print(f"  z_gt mean: {z_gt_flat.mean():.4f}, std: {z_gt_flat.std():.4f}")
print(f"  z_gt min: {z_gt_flat.min():.4f}, max: {z_gt_flat.max():.4f}")

# ── 4. Single-step x0 prediction at various timesteps ─────────────────────
print("\n" + "=" * 70)
print("STAGE 4: Single-step x0 prediction quality")
cond_emb = model.ego_encoder(ego)
print(f"  cond_emb shape: {cond_emb.shape}, norm: {cond_emb.norm():.4f}")

print(f"\n  {'t':>5} | {'x0_MSE':>8} | {'x0_cos':>8} | {'x0_norm':>8}")
print(f"  {'-'*45}")
for t_val in [0, 10, 50, 100, 500, 999]:
    mses, coss = [], []
    for _ in range(20):
        t = torch.full((1,), t_val, device=device).long()
        noise = torch.randn_like(z_gt_flat)
        z_t = model.noise_scheduler.add_noise(z_gt_flat, noise, t)
        with torch.no_grad():
            x0_pred = model.denoiser(
                sample=z_t, timestep=t,
                encoder_hidden_states=cond_emb, lengths=lengths
            )[0]
        mses.append(F.mse_loss(x0_pred, z_gt_flat).item())
        coss.append(F.cosine_similarity(x0_pred.flatten(), z_gt_flat.flatten(), dim=0).item())
    print(f"  {t_val:5d} | {np.mean(mses):8.4f} | {np.mean(coss):8.4f} | {x0_pred.norm():.4f}")

# ── 5. DDIM reverse ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 5: DDIM reverse (50 steps)")
with torch.no_grad():
    torch.manual_seed(42)
    # Use the model's own reverse method
    if model.do_classifier_free_guidance:
        uncond_ego = torch.zeros_like(ego)
        ego_cfg = torch.cat([uncond_ego, ego], dim=0)
        cond_emb_cfg = model.ego_encoder(ego_cfg)
    else:
        cond_emb_cfg = model.ego_encoder(ego)
    
    z_pred = model._diffusion_reverse(cond_emb_cfg, lengths)  # (1, B, 256)

z_pred_flat = z_pred.permute(1, 0, 2)  # (B, 1, 256)
print(f"  z_pred shape: {z_pred_flat.shape}")
print(f"  z_pred norm: {z_pred_flat.norm():.4f} (GT: {z_gt_flat.norm():.4f})")
print(f"  z_pred mean: {z_pred_flat.mean():.4f} (GT: {z_gt_flat.mean():.4f})")
print(f"  MSE(z_pred, z_gt): {F.mse_loss(z_pred_flat, z_gt_flat):.6f}")
print(f"  Cosine(z_pred, z_gt): {F.cosine_similarity(z_pred_flat.flatten(), z_gt_flat.flatten(), dim=0):.6f}")

# ── 6. VAE decode ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 6: VAE decode")
with torch.no_grad():
    feats_pred = model.vae.decode(z_pred, lengths)  # (B, T, 263)
    feats_gt   = model.vae.decode(z_gt, lengths)    # (B, T, 263) — VAE recon
print(f"  feats_pred shape: {feats_pred.shape}")
print(f"  feats_pred range: [{feats_pred.min():.4f}, {feats_pred.max():.4f}]")
print(f"  feats_gt range:   [{feats_gt.min():.4f}, {feats_gt.max():.4f}]")
print(f"  MSE(feats): {F.mse_loss(feats_pred, feats_gt):.6f}")

# ── 7. Denormalize ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 7: Denormalize features")
ds = dm.train_dataset
mean_t = torch.tensor(ds.mean, dtype=feats_pred.dtype, device=device)
std_t  = torch.tensor(ds.std, dtype=feats_pred.dtype, device=device)

feats_pred_denorm = feats_pred * std_t + mean_t
feats_gt_denorm   = feats_gt * std_t + mean_t
motion_denorm     = motion * std_t + mean_t  # input motion denormalized

print(f"  pred denorm range: [{feats_pred_denorm.min():.4f}, {feats_pred_denorm.max():.4f}]")
print(f"  gt denorm range:   [{feats_gt_denorm.min():.4f}, {feats_gt_denorm.max():.4f}]")
print(f"  input denorm range: [{motion_denorm.min():.4f}, {motion_denorm.max():.4f}]")

# ── 8. Recover joints ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 8: Recover joints (recover_from_ric)")
joints_pred  = recover_from_ric(feats_pred_denorm[0].cpu(), 22)
joints_gt    = recover_from_ric(feats_gt_denorm[0].cpu(), 22)
joints_input = recover_from_ric(motion_denorm[0].cpu(), 22)

T = min(joints_pred.shape[0], joints_gt.shape[0], joints_input.shape[0])
mpjpe_pred_vs_input = (joints_pred[:T] - joints_input[:T]).norm(dim=-1).mean().item()
mpjpe_gt_vs_input   = (joints_gt[:T] - joints_input[:T]).norm(dim=-1).mean().item()
mpjpe_pred_vs_gt    = (joints_pred[:T] - joints_gt[:T]).norm(dim=-1).mean().item()

print(f"  joints_pred shape: {joints_pred.shape}")
print(f"  MPJPE pred vs input:   {mpjpe_pred_vs_input:.4f}m")
print(f"  MPJPE VAE-gt vs input: {mpjpe_gt_vs_input:.4f}m  (VAE recon baseline)")
print(f"  MPJPE pred vs VAE-gt:  {mpjpe_pred_vs_gt:.4f}m")

# ── 9. Check what demo_ego_motion.py produces ─────────────────────────────
print("\n" + "=" * 70)
print("STAGE 9: Replicate demo_ego_motion.py flow")
import json
json_path = '/home/erik/NAS/methods/diffusion_gen/data/diffusion/ava/train/0001_33.json'
with open(json_path) as f:
    data = json.load(f)
ego_3d = np.array(data["ego_in_ped_frame"], dtype=np.float32)
ego_2d = ego_3d[:, [0, 2]]

# Load ego mean/std the way demo does
ego_mean_std_path = cfg.DATASET.EGOMOTION.EGO_MEAN_STD_PATH
ego_mean = np.load(os.path.join(ego_mean_std_path, "Ego_Mean.npy"))
ego_std  = np.load(os.path.join(ego_mean_std_path, "Ego_Std.npy"))
print(f"  Demo ego_mean: {ego_mean}")
print(f"  Demo ego_std:  {ego_std}")

# Demo normalization
ego_demo = (ego_2d - ego_mean) / (ego_std + 1e-8)
actual_len = len(ego_demo)
if actual_len < 196:
    padding = np.zeros((196 - actual_len, 2), dtype=np.float32)
    ego_demo = np.concatenate([ego_demo, padding], axis=0)
else:
    ego_demo = ego_demo[:196]

# Compare with dataset ego
ego_dataset = ego[:1].cpu().numpy()[0]  # From dataloader
print(f"  Demo ego shape: {ego_demo.shape}, Dataset ego shape: {ego_dataset.shape}")
print(f"  ego mismatch (MSE): {np.mean((ego_demo[:actual_len] - ego_dataset[:actual_len])**2):.8f}")

# Generate through demo path
ego_demo_t = torch.tensor(ego_demo, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    if model.do_classifier_free_guidance:
        uncond_ego = torch.zeros_like(ego_demo_t)
        ego_input = torch.cat([uncond_ego, ego_demo_t], dim=0)
    else:
        ego_input = ego_demo_t
    cond_demo = model.ego_encoder(ego_input)
    torch.manual_seed(42)
    z_demo = model._diffusion_reverse(cond_demo, [196])
    feats_demo = model.vae.decode(z_demo, [196])

# Demo denormalizes using mean/std from files
mean_path = cfg.DATASET.EGOMOTION.MEAN_STD_PATH
mean_np = np.load(os.path.join(mean_path, "Mean.npy"))
std_np  = np.load(os.path.join(mean_path, "Std.npy"))
feats_demo_np = feats_demo[0].cpu().numpy()
feats_demo_denorm = feats_demo_np * std_np + mean_np
joints_demo = recover_from_ric(torch.tensor(feats_demo_denorm), 22).numpy()
print(f"  Demo joints shape: {joints_demo.shape}")
print(f"  Demo joints range: [{joints_demo.min():.4f}, {joints_demo.max():.4f}]")

# Sanity: what does input motion joints look like?
print(f"\n  Input joints range: [{joints_input.numpy().min():.4f}, {joints_input.numpy().max():.4f}]")
print(f"  VAE-gt joints range: [{joints_gt.numpy().min():.4f}, {joints_gt.numpy().max():.4f}]")
print(f"  Pred joints range: [{joints_pred.numpy().min():.4f}, {joints_pred.numpy().max():.4f}]")

# Frame-by-frame: first 5 frames, root joint position  
print(f"\n  Root joint (joint 0) first 5 frames:")
print(f"  {'frame':>5} | {'input':>30} | {'pred':>30} | {'vae-gt':>30}")
for i in range(min(5, T)):
    inp = joints_input[i, 0].numpy()
    pred = joints_pred[i, 0].numpy()
    gt = joints_gt[i, 0].numpy()
    print(f"  {i:5d} | ({inp[0]:8.3f}, {inp[1]:8.3f}, {inp[2]:8.3f}) | ({pred[0]:8.3f}, {pred[1]:8.3f}, {pred[2]:8.3f}) | ({gt[0]:8.3f}, {gt[1]:8.3f}, {gt[2]:8.3f})")
