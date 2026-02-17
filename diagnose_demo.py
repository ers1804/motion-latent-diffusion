"""
Quick diagnostic: compare demo pipeline vs training pipeline output.
Loads the same ego sample and checks if the outputs match.
"""
import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
from collections import OrderedDict
from omegaconf import OmegaConf
from mld.config import get_module_config, instantiate_from_config
from mld.data.get_data import get_datasets

# ── Config ────────────────────────────────────────────────────────────
cfg_base = OmegaConf.load("./configs/base.yaml")
cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load("./configs/config_ego_motion.yaml"))
cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
cfg_assets = OmegaConf.load("./configs/assets.yaml")
cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(cfg.SEED_VALUE)

# ── Dataset (to get the same batch training sees) ─────────────────────
datasets = get_datasets(cfg, phase="train")
datamodule = datasets[0]
datamodule.setup(stage="fit")
train_loader = datamodule.train_dataloader()
batch = next(iter(train_loader))

print("=== Batch from DataLoader ===")
print(f"  motion shape: {batch['motion'].shape}")
print(f"  ego shape:    {batch['ego'].shape}")
print(f"  lengths:      {batch['length'][:5]}")
print(f"  motion range: [{batch['motion'].min():.3f}, {batch['motion'].max():.3f}]")
print(f"  ego range:    [{batch['ego'].min():.3f}, {batch['ego'].max():.3f}]")

# ── Load model the SAME way demo_ego_motion.py does ──────────────────
from mld.models.get_model import get_model
model = get_model(cfg, datamodule)

# Find the latest checkpoint
ckpt_dir = None
exp_dirs = sorted([d for d in os.listdir("experiments/mld") if cfg.NAME in d])
if exp_dirs:
    ckpt_dir = os.path.join("experiments/mld", exp_dirs[-1], "checkpoints")
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')])
    if ckpts:
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    else:
        print("No checkpoints found!")
        sys.exit(1)
else:
    print(f"No experiment directory matching '{cfg.NAME}' found in experiments/mld/")
    print("Available:", os.listdir("experiments/mld/"))
    sys.exit(1)

print(f"\n=== Loading checkpoint: {ckpt_path} ===")
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

# Check what keys are in the checkpoint
ego_keys = [k for k in state_dict.keys() if 'ego' in k]
denoiser_keys = [k for k in state_dict.keys() if 'denoiser' in k]
vae_keys = [k for k in state_dict.keys() if 'vae' in k]
print(f"  Checkpoint keys: {len(state_dict)} total")
print(f"  ego_encoder keys: {len(ego_keys)}")
print(f"  denoiser keys:    {len(denoiser_keys)}")
print(f"  vae keys:         {len(vae_keys)}")
if ego_keys:
    print(f"  Sample ego keys: {ego_keys[:3]}")

# Load with strict=False (like demo does)
# First check: does the base class load_state_dict handle ego correctly?
# The override only touches text_encoder, which we don't have
model_keys = set(k for k, _ in model.named_parameters())
ckpt_keys = set(state_dict.keys())
missing = model_keys - ckpt_keys
unexpected = ckpt_keys - set(k for k, _ in model.named_parameters()) - set(k for k, _ in model.named_buffers())
print(f"\n  Missing from ckpt ({len(missing)}): {list(missing)[:5] if missing else 'none'}")
print(f"  Unexpected in ckpt ({len(unexpected)}): {list(unexpected)[:5] if unexpected else 'none'}")

model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()

# ── Test 1: Run through training pipeline (test_diffusion_forward) ────
print("\n=== Test 1: model.test_diffusion_forward (training pipeline) ===")
batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
with torch.no_grad():
    rs_set = model.test_diffusion_forward(batch_gpu)
feats_train = rs_set["m_rst"]
print(f"  Output features shape: {feats_train.shape}")
print(f"  Output features range: [{feats_train.min():.3f}, {feats_train.max():.3f}]")
print(f"  Output features mean:  {feats_train.mean():.4f}")
print(f"  Output features std:   {feats_train.std():.4f}")

# ── Test 2: Run through demo pipeline (generate_motion) ──────────────
print("\n=== Test 2: demo pipeline (generate_motion logic) ===")
# Take the first sample's ego from the batch
ego_single = batch['ego'][0].to(device)  # (T, 2) - already normalized by dataset
length = batch['length'][0]
print(f"  Input ego shape: {ego_single.shape}")
print(f"  Input ego range: [{ego_single.min():.3f}, {ego_single.max():.3f}]")
print(f"  Requested length: {length}")

with torch.no_grad():
    ego_tensor = ego_single.unsqueeze(0)  # (1, T, 2)
    if model.do_classifier_free_guidance:
        uncond_ego = torch.zeros_like(ego_tensor)
        ego_input = torch.cat([uncond_ego, ego_tensor], dim=0)
    else:
        ego_input = ego_tensor
    
    cond_emb = model.ego_encoder(ego_input)
    print(f"  cond_emb shape: {cond_emb.shape}")
    print(f"  cond_emb range: [{cond_emb.min():.3f}, {cond_emb.max():.3f}]")
    
    z = model._diffusion_reverse(cond_emb, [length])
    print(f"  z shape: {z.shape}")
    print(f"  z range: [{z.min():.3f}, {z.max():.3f}]")
    
    feats_demo = model.vae.decode(z, [length])
    print(f"  feats shape: {feats_demo.shape}")
    print(f"  feats range: [{feats_demo.min():.3f}, {feats_demo.max():.3f}]")
    print(f"  feats mean:  {feats_demo.mean():.4f}")
    print(f"  feats std:   {feats_demo.std():.4f}")

# ── Test 3: Compare with GT VAE reconstruction ───────────────────────
print("\n=== Test 3: VAE encode→decode (upper bound quality) ===")
motion_gpu = batch['motion'][:1].to(device)
with torch.no_grad():
    z_gt, dist_gt = model.vae.encode(motion_gpu, [length])
    feats_recon = model.vae.decode(z_gt, [length])
    # Also try with mean
    z_mean = dist_gt.loc
    feats_recon_mean = model.vae.decode(z_mean, [length])
    
print(f"  z_gt range:        [{z_gt.min():.3f}, {z_gt.max():.3f}], norm={z_gt.norm():.2f}")
print(f"  z_mean range:      [{z_mean.min():.3f}, {z_mean.max():.3f}], norm={z_mean.norm():.2f}")
print(f"  z_diffusion range: [{z.min():.3f}, {z.max():.3f}], norm={z.norm():.2f}")
print(f"  z_diffusion vs z_mean MSE: {((z.permute(1,0,2) - z_mean.permute(1,0,2))**2).mean():.6f}")

# ── Test 4: Check denormalization ─────────────────────────────────────
print("\n=== Test 4: Denormalization check ===")
mean_path = cfg.DATASET.EGOMOTION.MEAN_STD_PATH
if mean_path:
    mean_npy = np.load(os.path.join(mean_path, "Mean.npy"))
    std_npy = np.load(os.path.join(mean_path, "Std.npy"))
    print(f"  Motion mean shape: {mean_npy.shape}, range: [{mean_npy.min():.3f}, {mean_npy.max():.3f}]")
    print(f"  Motion std shape:  {std_npy.shape}, range: [{std_npy.min():.3f}, {std_npy.max():.3f}]")
    print(f"  Any zero std? {(std_npy == 0).sum()} dims")
    print(f"  Any near-zero std? {(std_npy < 1e-6).sum()} dims")
    
    # Denormalize the demo output
    feats_denorm = feats_demo[0].cpu().numpy() * std_npy + mean_npy
    print(f"  Denormalized features range: [{feats_denorm.min():.3f}, {feats_denorm.max():.3f}]")
    
    # Compare to GT denormalized
    gt_motion = batch['motion'][0].numpy()  # This is still normalized
    gt_denorm = gt_motion * std_npy + mean_npy
    print(f"  GT denormalized range: [{gt_denorm.min():.3f}, {gt_denorm.max():.3f}]")

# ── Test 5: Check joints conversion ──────────────────────────────────
print("\n=== Test 5: Joints conversion ===")
from mld.data.humanml.scripts.motion_process import recover_from_ric

# From demo output (denormalized)
feats_for_joints = torch.tensor(feats_denorm, dtype=torch.float32).unsqueeze(0)
joints_demo = recover_from_ric(feats_for_joints, 22)
print(f"  Demo joints shape: {joints_demo.shape}")
print(f"  Demo joints range: [{joints_demo.min():.3f}, {joints_demo.max():.3f}]")
print(f"  Demo joints[0] pelvis: {joints_demo[0, 0, 0, :]}")

# From GT (denormalized)
gt_for_joints = torch.tensor(gt_denorm, dtype=torch.float32).unsqueeze(0)
joints_gt = recover_from_ric(gt_for_joints, 22)
print(f"  GT joints shape: {joints_gt.shape}")
print(f"  GT joints range: [{joints_gt.min():.3f}, {joints_gt.max():.3f}]")
print(f"  GT joints[0] pelvis: {joints_gt[0, 0, 0, :]}")

# From model's feats2joints (uses normalized features + internal denorm)
with torch.no_grad():
    joints_model = model.feats2joints(feats_demo)
print(f"  Model feats2joints shape: {joints_model.shape}")
print(f"  Model feats2joints range: [{joints_model.min():.3f}, {joints_model.max():.3f}]")

print("\n=== SUMMARY ===")
print(f"  Diffusion z matches GT z_mean? MSE = {((z.permute(1,0,2) - z_mean.permute(1,0,2))**2).mean():.6f}")
if feats_train is not None:
    print(f"  Train pipeline feats range: [{feats_train.min():.3f}, {feats_train.max():.3f}]")
print(f"  Demo pipeline feats range:  [{feats_demo.min():.3f}, {feats_demo.max():.3f}]")
