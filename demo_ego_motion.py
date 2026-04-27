"""
Demo script for Ego-Conditioned Motion Generation

Usage:
    python demo_ego_motion.py \
        --checkpoint path/to/checkpoint.ckpt \
        --ego_json path/to/sample.json \
        --output_dir outputs/

    # Or use a sample from your dataset:
    python demo_ego_motion.py \
        --checkpoint path/to/checkpoint.ckpt \
        --data_dir data/diffusion/waymo/val \
        --num_samples 5 \
        --output_dir outputs/
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def _pad_or_crop(
    sequence: np.ndarray,
    max_length: int
) -> tuple:
    """
    Pad sequence to max_length or crop if too long.
    Returns: (padded_sequence, actual_length)
    """
    actual_length = len(sequence)

    if actual_length >= max_length:
        # Crop from the beginning (keep recent frames)
        sequence = sequence[:max_length]
        actual_length = max_length
    else:
        # Pad with zeros at the end
        pad_length = max_length - actual_length
        padding = np.zeros((pad_length, sequence.shape[-1]), dtype=sequence.dtype)
        sequence = np.concatenate([sequence, padding], axis=0)

    return sequence, actual_length


def _interaction_crop_aligned(
    ego_2d: np.ndarray,
    motion: np.ndarray,
    ego_ped_dists: np.ndarray,
    max_length: int,
    interaction_crop: bool = True,
) -> tuple:
    """Crop ego and motion together to max_length.

    When interaction_crop is True the window is centred on the frame of
    closest ego-pedestrian approach (matching EgoMotionDataset behaviour),
    with ±25 % uniform jitter.  Otherwise a random contiguous window is used.

    Returns: (ego_cropped, motion_cropped, actual_length)
    """
    actual_length = min(len(ego_2d), len(motion))
    # Align both to the same length first
    ego_2d = ego_2d[:actual_length]
    motion = motion[:actual_length]

    if actual_length >= max_length:
        if interaction_crop and ego_ped_dists is not None:
            center = int(np.argmin(ego_ped_dists[:actual_length]))
            jitter = int(max_length * 0.25)
            center += np.random.randint(-jitter, jitter + 1)
            start_idx = center - max_length // 2
            start_idx = max(0, min(start_idx, actual_length - max_length))
        else:
            start_idx = np.random.randint(0, actual_length - max_length + 1)
        ego_2d = ego_2d[start_idx:start_idx + max_length]
        motion = motion[start_idx:start_idx + max_length]
        actual_length = max_length
    else:
        pad = max_length - actual_length
        ego_2d = np.concatenate(
            [ego_2d, np.zeros((pad, ego_2d.shape[-1]), dtype=ego_2d.dtype)], axis=0
        )
        motion = np.concatenate(
            [motion, np.zeros((pad, motion.shape[-1]), dtype=motion.dtype)], axis=0
        )

    return ego_2d, motion, actual_length


def load_config(config_path):
    """Load config using OmegaConf, merging with base.yaml and module configs."""
    from omegaconf import OmegaConf
    from mld.config import get_module_config

    base_path = os.path.join(os.path.dirname(config_path), 'base.yaml')
    cfg = OmegaConf.load(config_path)
    if os.path.exists(base_path):
        cfg_base = OmegaConf.load(base_path)
        cfg = OmegaConf.merge(cfg_base, cfg)

    # Resolve module configs (adds 'target' keys etc.)
    cfg_model = get_module_config(cfg.model, cfg.model.target if 'target' in cfg.model else 'mld.models.modeltype.mld.MLD')
    cfg = OmegaConf.merge(cfg, cfg_model)

    # Merge assets if available
    assets_path = os.path.join(os.path.dirname(config_path), 'assets.yaml')
    if os.path.exists(assets_path):
        cfg_assets = OmegaConf.load(assets_path)
        cfg = OmegaConf.merge(cfg, cfg_assets)

    return cfg


def load_model(cfg, checkpoint_path, device):
    """Load trained model from checkpoint."""
    from mld.data.get_data import get_datasets
    from mld.models.get_model import get_model
    
    # Load dataset to get nfeats
    dataset = get_datasets(cfg, phase="test")[0]
    
    # Create model
    model = get_model(cfg, dataset)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)  # strict=False for VAE architecture mismatch
    
    model.to(device)
    model.eval()
    
    return model, dataset


def load_ego_from_json(
    json_path,
    max_motion_length=196,
    ego_mean=None,
    ego_std=None,
    interaction_crop=True,
):
    """Load and crop ego trajectory + GT motion from a JSON file.

    When interaction_crop is True the crop window is centred on the frame of
    closest ego-pedestrian approach (mirroring EgoMotionDataset behaviour),
    with ±25 % uniform jitter.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract ego: (T, 3) -> (T, 2) using x, z
    ego_3d = np.array(data["ego_in_ped_frame"], dtype=np.float32)
    ego_2d = ego_3d[:, [0, 2]]  # x, z only

    if ego_mean is None:
        ego_mean = np.array([0.03992962314720422, 5.138034295405447], dtype=np.float32)
    if ego_std is None:
        ego_std = np.array([13.274100607793672, 15.879450116636379], dtype=np.float32)

    # If mean/std are 3D (x, y, z), slice to match ego_2d's (x, z) columns
    if ego_mean.shape[-1] == 3:
        ego_mean = ego_mean[..., [0, 2]]
    if ego_std.shape[-1] == 3:
        ego_std = ego_std[..., [0, 2]]

    gt_motion = None
    if "vectors_263" in data:
        gt_motion = np.array(data["vectors_263"], dtype=np.float32)

    # Compute raw ego-ped distances BEFORE normalization (for interaction crop)
    ego_ped_dists = None
    if "ped_in_ped_frame" in data:
        ped_joints = np.array(data["ped_in_ped_frame"], dtype=np.float32)  # (T, 22, 3)
        ped_root_xz = ped_joints[:, 0, [0, 2]]  # (T, 2) pelvis
        T = min(len(ego_2d), len(ped_root_xz))
        ego_ped_dists = np.linalg.norm(ego_2d[:T] - ped_root_xz[:T], axis=-1)

    # Interaction-aware aligned crop (matches EgoMotionDataset._pad_or_crop_ego_motion)
    if gt_motion is not None:
        ego_2d, gt_motion, actual_len = _interaction_crop_aligned(
            ego_2d, gt_motion, ego_ped_dists, max_motion_length, interaction_crop
        )
    else:
        ego_2d, actual_len = _pad_or_crop(ego_2d, max_motion_length)

    # Normalize ego after cropping
    ego_2d = (ego_2d - ego_mean) / (ego_std + 1e-8)

    return ego_2d, actual_len, gt_motion, data.get("scene_id", ""), data.get("object_id", "")


def generate_motion(model, ego, length, device, mean=None, std=None,
                     guide_trajectory=None, guide_scale=100.0,
                     guide_start_t=1000, guide_stop_t=0,
                     guide_normalize=False, guide_verbose=False):
    """
    Generate pedestrian motion from ego trajectory.
    
    Returns vectors_263 features (not joints).
    
    Args:
        model: Trained MLD model
        ego: (T_ego, 2) numpy array (normalized)
        length: Desired output motion length
        device: torch device
        mean: Motion mean for denormalization (optional)
        std: Motion std for denormalization (optional)
        guide_trajectory: (T, 2) numpy array — target root (x,z) positions
                          in **denormalized** (real-world) metres.
                          If provided, uses guided diffusion.
        guide_scale: Gradient scale for trajectory guidance.
        guide_normalize: If True, normalize gradient to unit direction.
        guide_verbose: If True, print per-step diagnostics.
    
    Returns:
        features: (T, 263) numpy array - motion features
    """
    # Prepare batch
    ego_tensor = torch.tensor(ego, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, 2)
    lengths = [length]
    
    # Step 1: Encode ego trajectory
    with torch.no_grad():
        if model.do_classifier_free_guidance:
            uncond_ego = torch.zeros_like(ego_tensor)
            ego_input = torch.cat([uncond_ego, ego_tensor], dim=0)
        else:
            ego_input = ego_tensor
        cond_emb = model.ego_encoder(ego_input)

    # Step 2: Diffusion reverse — guided or standard
    if guide_trajectory is not None:
        traj_target = torch.tensor(
            guide_trajectory, dtype=torch.float32
        ).unsqueeze(0).to(device)                         # (1, T, 2)
        z = model._diffusion_reverse_guided(
            cond_emb, lengths,
            traj_target=traj_target,
            guide_scale=guide_scale,
            guide_start_t=guide_start_t,
            guide_stop_t=guide_stop_t,
            normalize_grad=guide_normalize,
            verbose=guide_verbose,
        )
    else:
        with torch.no_grad():
            z = model._diffusion_reverse(cond_emb, lengths)

    # Step 3: VAE decode to get features (vectors_263)
    with torch.no_grad():
        feats_rst = model.vae.decode(z, lengths)           # (B, T, 263)
    
    # Get the generated features
    features = feats_rst[0].cpu().numpy()  # (T, 263)
    
    # Optionally denormalize
    if mean is not None and std is not None:
        features = features * std + mean
    
    return features


def inject_trajectory(features, target_root_xz):
    """
    Post-hoc trajectory injection: replace the root velocity features in
    the generated motion with velocities derived from a target root (x,z)
    trajectory. Preserves all other pose features (joints, contacts, etc.).

    This works because the HumanML3D 263-dim features encode the root
    trajectory via:
      col 0: rotation velocity (yaw around Y)
      col 1: x velocity (in root's local frame)
      col 2: z velocity (in root's local frame)
      col 3: y height

    The local velocities are rotated to the global frame then integrated
    (cumsum) to recover positions. To inject a new trajectory, we compute
    the inverse: target global displacements → rotate to local frame →
    store as velocity features.

    Args:
        features: (T, 263) numpy array — denormalized generated features.
        target_root_xz: (T, 2) numpy array — target root (x, z) positions
                         in the same coordinate system as the features.

    Returns:
        corrected: (T, 263) numpy array — features with replaced trajectory.
    """
    from mld.data.humanml.common.quaternion import qrot, qinv

    features = features.copy()
    T = features.shape[0]
    feats_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, T, 263)

    # ---- 1. Get the root rotation from the GENERATED features ----
    # (we keep the generated root rotation, only replace the translation)
    rot_vel = feats_t[0, :, 0]               # (T,)
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=0)   # cumulative rotation angle

    r_rot_quat = torch.zeros(T, 4)
    r_rot_quat[:, 0] = torch.cos(r_rot_ang)
    r_rot_quat[:, 2] = torch.sin(r_rot_ang)

    # ---- 2. Compute global displacements from target positions ----
    target_t = torch.tensor(target_root_xz, dtype=torch.float32)  # (T, 2)
    # Global displacement (in 3D, Y=0): (T-1, 3) 
    global_disp = torch.zeros(T, 3)
    global_disp[1:, 0] = target_t[1:, 0] - target_t[:-1, 0]  # dx
    global_disp[1:, 2] = target_t[1:, 1] - target_t[:-1, 1]  # dz

    # ---- 3. Rotate global displacements to root-local frame ----
    # The feature encoding does: local_disp → qrot(qinv(rot), local_disp) → global
    # So to go back: global_disp → qrot(rot, global_disp) → local
    local_disp = qrot(r_rot_quat.unsqueeze(0), global_disp.unsqueeze(0))  # (1, T, 3)
    local_disp = local_disp[0]  # (T, 3)

    # ---- 4. Replace velocity features ----
    # Features: data[t, 1:3] corresponds to the displacement applied at frame t+1
    # i.e., r_pos[1:, [0,2]] = data[:-1, 1:3]
    # So: data[t, 1] = local_disp[t+1, 0], data[t, 2] = local_disp[t+1, 2]
    features[:-1, 1] = local_disp[1:, 0].numpy()   # x velocity
    features[:-1, 2] = local_disp[1:, 2].numpy()   # z velocity

    return features


def features_to_joints(features, njoints=22):
    """
    Convert 263-dim features to 3D joints using recover_from_ric.
    
    Args:
        features: (T, 263) numpy array (denormalized)
    
    Returns:
        joints: (T, njoints, 3) numpy array
    """
    from mld.data.humanml.scripts.motion_process import recover_from_ric
    
    # recover_from_ric expects tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)
    if features_tensor.dim() == 2:
        features_tensor = features_tensor.unsqueeze(0)  # (1, T, 263)
    
    joints = recover_from_ric(features_tensor, njoints)
    
    return joints[0].numpy()  # (T, njoints, 3)


def save_results(output_dir, sample_name, features, joints, ego, gt_motion=None):
    """Save generated motion and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save generated features (vectors_263)
    feat_path = os.path.join(output_dir, f"{sample_name}_features.npy")
    np.save(feat_path, features)
    print(f"  Saved features: {feat_path}")
    
    # Save generated joints
    joints_path = os.path.join(output_dir, f"{sample_name}_joints.npy")
    np.save(joints_path, joints)
    print(f"  Saved joints: {joints_path}")
    
    # Save ego trajectory
    ego_path = os.path.join(output_dir, f"{sample_name}_ego.npy")
    np.save(ego_path, ego)
    print(f"  Saved ego: {ego_path}")
    
    # Save ground truth if available
    if gt_motion is not None:
        gt_path = os.path.join(output_dir, f"{sample_name}_gt_features.npy")
        np.save(gt_path, gt_motion)
        print(f"  Saved GT features: {gt_path}")


def compute_trajectory_metrics(pred_traj, gt_traj, length):
    """
    Compute ADE and FDE between predicted and ground truth root trajectories.

    Args:
        pred_traj: (T, 2) predicted root positions (x, z)
        gt_traj: (T, 2) ground truth root positions (x, z)
        length: Number of valid frames to consider

    Returns:
        ade: Average Displacement Error (mean L2 over all valid frames)
        fde: Final Displacement Error (L2 at last valid frame)
    """
    pred = pred_traj[:length]
    gt = gt_traj[:length]
    displacements = np.linalg.norm(pred - gt, axis=-1)  # (T,)
    ade = float(np.mean(displacements))
    fde = float(displacements[-1])
    return ade, fde


def plot_trajectories(
    gen_joints,
    gt_joints,
    ego_denorm,
    length,
    output_path,
    sample_name,
    ade=None,
    fde=None,
):
    """
    Plot top-down (x, z) trajectories of generated motion, GT motion, and ego.

    Args:
        gen_joints: (T, 22, 3) generated joints
        gt_joints: (T, 22, 3) ground truth joints (or None)
        ego_denorm: (T, 2) denormalized ego trajectory (x, z)
        length: Number of valid frames
        output_path: Directory to save the plot
        sample_name: Name for the output file
        ade: ADE metric value (optional, displayed in legend)
        fde: FDE metric value (optional, displayed in legend)
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Generated root trajectory (joint 0, x and z)
    gen_root = gen_joints[:length, 0, [0, 2]]  # (T, 2)
    ax.plot(gen_root[:, 0], gen_root[:, 1], '-o', color='#2196F3',
            markersize=2, linewidth=1.5, label='Generated')
    ax.plot(gen_root[0, 0], gen_root[0, 1], 's', color='#2196F3', markersize=8)
    ax.plot(gen_root[-1, 0], gen_root[-1, 1], '*', color='#2196F3', markersize=12)

    # Ground truth root trajectory
    if gt_joints is not None:
        gt_root = gt_joints[:length, 0, [0, 2]]  # (T, 2)
        ax.plot(gt_root[:, 0], gt_root[:, 1], '-o', color='#4CAF50',
                markersize=2, linewidth=1.5, label='Ground Truth')
        ax.plot(gt_root[0, 0], gt_root[0, 1], 's', color='#4CAF50', markersize=8)
        ax.plot(gt_root[-1, 0], gt_root[-1, 1], '*', color='#4CAF50', markersize=12)

    # Ego trajectory
    ego_plot = ego_denorm[:length]
    ax.plot(ego_plot[:, 0], ego_plot[:, 1], '-o', color='#FF5722',
            markersize=2, linewidth=1.5, label='Ego')
    ax.plot(ego_plot[0, 0], ego_plot[0, 1], 's', color='#FF5722', markersize=8)
    ax.plot(ego_plot[-1, 0], ego_plot[-1, 1], '*', color='#FF5722', markersize=12)

    # Metrics text
    metrics_str = ''
    if ade is not None:
        metrics_str += f'ADE: {ade:.4f}'
    if fde is not None:
        metrics_str += f'  FDE: {fde:.4f}'
    if metrics_str:
        ax.text(0.02, 0.98, metrics_str, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title(f'Top-Down Trajectories: {sample_name}')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_path, f"{sample_name}_trajectories.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved trajectory plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Ego-Conditioned Motion Generation Demo")
    parser.add_argument("--config", type=str, default="configs/config_ego_motion.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--ego_json", type=str, default=None,
                        help="Path to single JSON file with ego trajectory")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to directory with JSON files (e.g., data/diffusion/waymo/val)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate from data_dir")
    parser.add_argument("--output_dir", type=str, default="outputs/ego_demo",
                        help="Output directory for generated motions")
    parser.add_argument("--motion_length", type=int, default=-1,
                        help="Output motion length in frames (-1 = auto-detect from ego trajectory)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--num_repetitions", type=int, default=1,
                        help="Generate multiple motions per ego (for diversity check)")
    parser.add_argument("--mean_std_path", type=str, default="/home/erik/NAS/methods/diffusion_gen/data/vae/mean_std_txt/ava_human_nuscenes_waymo",
                        help="Path to mean/std for denormalization (directory with Mean.npy, Std.npy)")
    parser.add_argument("--trajectories", action="store_true",
                        help="Plot top-down trajectories and compute ADE/FDE metrics")
    parser.add_argument("--guide", action="store_true",
                        help="Enable trajectory guidance (GMD-style). Uses GT root trajectory as target.")
    parser.add_argument("--guide_scale", type=float, default=100.0,
                        help="Trajectory guidance gradient scale (higher = stronger steering)")
    parser.add_argument("--guide_start_t", type=int, default=1000,
                        help="Only guide when t <= this value (default: 1000 = guide all steps)")
    parser.add_argument("--guide_stop_t", type=int, default=0,
                        help="Stop guiding when t < this value (default: 0 = guide to the end)")
    parser.add_argument("--guide_normalize", action="store_true",
                        help="Normalize guidance gradient (unit direction, magnitude controlled only by scale)")
    parser.add_argument("--guide_verbose", action="store_true",
                        help="Print per-step guidance diagnostics (loss, gradient norm, shift)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--inject_traj", action="store_true",
                        help="Post-hoc: replace root trajectory features with GT trajectory (keeps generated pose)")
    parser.add_argument("--no_interaction_crop", action="store_true",
                        help="Disable interaction-centred cropping (use random crop instead, as in vanilla MLD)")
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    cfg = load_config(args.config)
    cfg.TEST.CHECKPOINTS = args.checkpoint
    
    # Load model
    model, dataset = load_model(cfg, args.checkpoint, device)
    print("Model loaded successfully")
    
    # Load mean/std for denormalization
    # Try in order: --mean_std_path, config MEAN_STD_PATH, HUMANML3D.ROOT
    mean, std = None, None
    candidate_paths = [
        args.mean_std_path,
        getattr(cfg.DATASET.EGOMOTION, 'MEAN_STD_PATH', None),
        getattr(cfg.DATASET, 'HUMANML3D', {}).get('ROOT', None) if hasattr(cfg.DATASET, 'HUMANML3D') else None,
    ]
    for mean_std_path in candidate_paths:
        if not mean_std_path:
            continue
        mean_file = os.path.join(mean_std_path, "Mean.npy")
        std_file = os.path.join(mean_std_path, "Std.npy")
        if os.path.exists(mean_file) and os.path.exists(std_file):
            mean = np.load(mean_file)
            std = np.load(std_file)
            print(f"Loaded mean/std from {mean_std_path}")
            break
    if mean is None:
        print("WARNING: No valid Mean.npy/Std.npy found! Features will NOT be denormalized.")
        print("  This will produce garbage joints. Provide --mean_std_path or set MEAN_STD_PATH in config.")
    
    # Get ego parameters from config
    max_motion_length = cfg.DATASET.SAMPLER.MAX_LEN
    ego_mean_std_path = getattr(cfg.DATASET.EGOMOTION, 'EGO_MEAN_STD_PATH', None)
    if ego_mean_std_path:
        ego_mean_file = os.path.join(ego_mean_std_path, "Ego_Mean.npy")
        ego_std_file = os.path.join(ego_mean_std_path, "Ego_Std.npy")
        if os.path.exists(ego_mean_file) and os.path.exists(ego_std_file):
            ego_mean = np.load(ego_mean_file)
            ego_std = np.load(ego_std_file)
            print(f"Loaded ego mean/std from {ego_mean_std_path}")
        else:
            print(f"Warning: Ego Mean/Std not found at {ego_mean_std_path}")
            ego_mean, ego_std = None, None
    else:
        ego_mean, ego_std = None, None

    # Slice 3D ego mean/std to 2D (x, z) to match ego trajectory columns
    if ego_mean is not None and ego_mean.shape[-1] == 3:
        ego_mean = ego_mean[..., [0, 2]]
    if ego_std is not None and ego_std.shape[-1] == 3:
        ego_std = ego_std[..., [0, 2]]
    
    # Collect input files
    json_files = []
    if args.ego_json:
        json_files = [args.ego_json]
    elif args.data_dir:
        json_files = sorted(glob(os.path.join(args.data_dir, "*.json")))[:args.num_samples]
    else:
        print("Error: Provide either --ego_json or --data_dir")
        return
    
    print(f"Processing {len(json_files)} samples...")
    
    # Process each sample
    for json_path in json_files:
        sample_name = Path(json_path).stem
        print(f"\n{'='*50}")
        print(f"Processing: {sample_name}")

        # Set seed per-sample for reproducibility
        if args.seed is not None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)
        
        # Load ego trajectory (interaction-centred crop applied inside)
        ego, ego_len, gt_motion, scene_id, object_id = load_ego_from_json(
            json_path,
            max_motion_length=max_motion_length,
            ego_mean=ego_mean,
            ego_std=ego_std,
            interaction_crop=not args.no_interaction_crop,
        )
        print(f"  Ego length: {ego_len} frames")
        if gt_motion is not None:
            print(f"  GT motion length: {len(gt_motion)} frames")

        # Generate motion(s)
        for rep in range(args.num_repetitions):
            rep_suffix = f"_rep{rep}" if args.num_repetitions > 1 else ""

            # Honour explicit --motion_length override; otherwise use the cropped length
            motion_length = ego_len if args.motion_length <= 0 else args.motion_length
            
            # Generate features (vectors_263)
            # Build guidance target from GT root trajectory if --guide
            guide_traj = None
            if args.guide and gt_motion is not None:
                gt_joints_for_guide = features_to_joints(gt_motion, njoints=22)
                guide_traj = gt_joints_for_guide[:, 0, [0, 2]]  # (T, 2) root xz
                print(f"  Trajectory guidance enabled (scale={args.guide_scale})")
            features = generate_motion(
                model, ego, motion_length, device, mean, std,
                guide_trajectory=guide_traj,
                guide_scale=args.guide_scale,
                guide_start_t=args.guide_start_t,
                guide_stop_t=args.guide_stop_t,
                guide_normalize=args.guide_normalize,
                guide_verbose=args.guide_verbose,
            )
            print(f"  Generated features shape: {features.shape} (motion_length={motion_length})")

            # Post-hoc trajectory injection (if requested)
            if args.inject_traj and gt_motion is not None:
                gt_joints_for_inject = features_to_joints(gt_motion, njoints=22)
                gt_root_xz = gt_joints_for_inject[:, 0, [0, 2]]  # (T, 2)
                features = inject_trajectory(features, gt_root_xz)
                print(f"  Post-hoc trajectory injection applied")

            # Convert to joints using recover_from_ric
            joints = features_to_joints(features, njoints=22)
            print(f"  Generated joints shape: {joints.shape}")
            
            path_to_save = os.path.join(args.output_dir, cfg.NAME)
            os.mkdir(path_to_save) if not os.path.exists(path_to_save) else None

            # Denormalize ego for saving/plotting
            ego_denorm = ego * ego_std + ego_mean if ego_std is not None and ego_mean is not None else ego

            # Save results
            save_results(
                path_to_save,
                f"{sample_name}{rep_suffix}",
                features,
                joints,
                ego_denorm,
                gt_motion
            )

            # Trajectory plotting and metrics
            if args.trajectories:
                # Convert GT features to joints for comparison
                gt_joints = None
                # if gt_motion is not None and mean is not None and std is not None:
                #     gt_features_denorm = gt_motion * std + mean
                gt_joints = features_to_joints(gt_motion, njoints=22)

                # Compute ADE/FDE if GT is available
                ade, fde = None, None
                if gt_joints is not None:
                    gen_root = joints[:, 0, [0, 2]]  # (T, 2)
                    gt_root = gt_joints[:, 0, [0, 2]]  # (T, 2)
                    ade, fde = compute_trajectory_metrics(gen_root, gt_root, motion_length)
                    print(f"  ADE: {ade:.4f}  FDE: {fde:.4f}")

                plot_trajectories(
                    gen_joints=joints,
                    gt_joints=gt_joints,
                    ego_denorm=ego_denorm,
                    length=motion_length,
                    output_path=path_to_save,
                    sample_name=f"{sample_name}{rep_suffix}",
                    ade=ade,
                    fde=fde,
                )
    
    print(f"\n{'='*50}")
    print(f"Done! Results saved to: {args.output_dir}")
    print("\nOutput files:")
    print("  - *_features.npy: Generated motion features (T, 263)")
    print("  - *_joints.npy: Generated 3D joints (T, 22, 3)")
    print("  - *_ego.npy: Ego trajectory (T, 2)")
    print("  - *_gt_features.npy: Ground truth features (if available)")
    if args.trajectories:
        print("  - *_trajectories.png: Top-down trajectory plot with ADE/FDE")


if __name__ == "__main__":
    main()
