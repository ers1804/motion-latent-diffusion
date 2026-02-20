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


def load_ego_from_json(json_path, max_ego_len=196, ego_scale=50.0, ego_mean=None, ego_std=None):
    """Load ego trajectory from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Extract ego: (T, 3) -> (T, 2) using x, z
    ego_3d = np.array(data["ego_in_ped_frame"], dtype=np.float32)
    ego_2d = ego_3d[:, [0, 2]]  # x, z only

    if ego_mean is None:
        ego_mean = np.array([0.03992962314720422, 5.138034295405447], dtype=np.float32)
    if ego_std is None:
        ego_std = np.array([13.274100607793672, 15.879450116636379], dtype=np.float32)
    
    # Also load ground truth motion if available (for comparison)
    gt_motion = None
    if "vectors_263" in data:
        gt_motion = np.array(data["vectors_263"], dtype=np.float32)
    # Normalize
    ego_2d = (ego_2d - ego_mean) / (ego_std + 1e-8)
    # Pad/crop to max_ego_len
    actual_len = len(ego_2d)
    if actual_len >= len(gt_motion):
        ego_2d = ego_2d[:len(gt_motion)]
        actual_len = len(gt_motion)
    else:
        padding = np.zeros((len(gt_motion) - actual_len, 2), dtype=np.float32)
        ego_2d = np.concatenate([ego_2d, padding], axis=0)
    
    
    return ego_2d, actual_len, gt_motion, data.get("scene_id", ""), data.get("object_id", "")


def generate_motion(model, ego, length, device, mean=None, std=None):
    """
    Generate pedestrian motion from ego trajectory.
    
    Returns vectors_263 features (not joints).
    
    Args:
        model: Trained MLD model
        ego: (T_ego, 2) numpy array
        length: Desired output motion length
        device: torch device
        mean: Motion mean for denormalization (optional)
        std: Motion std for denormalization (optional)
    
    Returns:
        features: (T, 263) numpy array - motion features
    """
    # Prepare batch
    ego_tensor = torch.tensor(ego, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, 2)
    lengths = [length]
    
    with torch.no_grad():
        # Step 1: Encode ego trajectory
        if model.do_classifier_free_guidance:
            uncond_ego = torch.zeros_like(ego_tensor)
            ego_input = torch.cat([uncond_ego, ego_tensor], dim=0)
        else:
            ego_input = ego_tensor
        
        cond_emb = model.ego_encoder(ego_input)
        
        # Step 2: Diffusion reverse to get latent z
        z = model._diffusion_reverse(cond_emb, lengths)
        
        # Step 3: VAE decode to get features (vectors_263)
        feats_rst = model.vae.decode(z, lengths)  # (B, T, 263)
    
    # Get the generated features
    features = feats_rst[0].cpu().numpy()  # (T, 263)
    
    # Optionally denormalize
    if mean is not None and std is not None:
        features = features * std + mean
    
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
    max_ego_len = cfg.DATASET.EGOMOTION.MAX_EGO_LEN
    ego_scale = cfg.DATASET.EGOMOTION.EGO_SCALE
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
        
        # Load ego trajectory
        ego, ego_len, gt_motion, scene_id, object_id = load_ego_from_json(
            json_path, max_ego_len, ego_scale, ego_mean, ego_std
        )
        print(f"  Ego length: {ego_len} frames")
        if gt_motion is not None:
            print(f"  GT motion length: {len(gt_motion)} frames")
        
        # Generate motion(s)
        for rep in range(args.num_repetitions):
            rep_suffix = f"_rep{rep}" if args.num_repetitions > 1 else ""
            
            # Use actual ego length if --motion_length not explicitly set
            motion_length = ego_len if args.motion_length <= 0 else args.motion_length
            ego, ego_len = _pad_or_crop(ego, motion_length)
            gt_motion, motion_length = _pad_or_crop(gt_motion, motion_length)
            
            # Generate features (vectors_263)
            features = generate_motion(model, ego, motion_length, device, mean, std)
            print(f"  Generated features shape: {features.shape} (motion_length={motion_length})")
            
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
