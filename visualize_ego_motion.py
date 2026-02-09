"""
Visualization script for Ego-Conditioned Motion Generation results.

Usage:
    python visualize_ego_motion.py --output_dir outputs/ego_demo --sample_name scene001_obj01

This creates a simple matplotlib visualization showing:
- Ego trajectory (bird's eye view)
- Generated pedestrian trajectory (bird's eye view)
- Ground truth trajectory if available
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def extract_root_trajectory(motion, njoints=22):
    """
    Extract root (pelvis) trajectory from motion features or joints.
    
    If motion is (T, 263): Extract root velocity and integrate
    If motion is (T, 22, 3): Just take joint 0
    """
    if len(motion.shape) == 2 and motion.shape[1] == 263:
        # HumanML3D format: first 4 dims are root features
        # [root_rot_velocity, root_x_velocity, root_z_velocity, root_y]
        # Integrate velocities to get positions
        root_x_vel = motion[:, 1]
        root_z_vel = motion[:, 2]
        
        # Integrate
        root_x = np.cumsum(root_x_vel)
        root_z = np.cumsum(root_z_vel)
        
        return np.stack([root_x, root_z], axis=1)
    
    elif len(motion.shape) == 3:
        # Already joints: (T, J, 3)
        # Take pelvis (joint 0) x, z
        return motion[:, 0, [0, 2]]
    
    else:
        raise ValueError(f"Unknown motion shape: {motion.shape}")


def plot_bird_eye_view(ax, ego, gen_traj, gt_traj=None, title="Bird's Eye View"):
    """Plot trajectories from bird's eye view."""
    
    # Plot ego trajectory
    ax.plot(ego[:, 0], ego[:, 1], 'b-', linewidth=2, label='Ego Vehicle')
    ax.scatter(ego[0, 0], ego[0, 1], c='blue', s=100, marker='s', zorder=5)
    ax.scatter(ego[-1, 0], ego[-1, 1], c='blue', s=100, marker='^', zorder=5)
    
    # Plot generated trajectory
    ax.plot(gen_traj[:, 0], gen_traj[:, 1], 'r-', linewidth=2, label='Generated Ped')
    ax.scatter(gen_traj[0, 0], gen_traj[0, 1], c='red', s=100, marker='o', zorder=5)
    ax.scatter(gen_traj[-1, 0], gen_traj[-1, 1], c='red', s=100, marker='>', zorder=5)
    
    # Plot ground truth if available
    if gt_traj is not None:
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'g--', linewidth=2, label='GT Ped', alpha=0.7)
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], c='green', s=100, marker='o', zorder=5)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def plot_trajectories_over_time(ax, gen_traj, gt_traj=None, title="Position over Time"):
    """Plot x, z positions over time."""
    t = np.arange(len(gen_traj))
    
    ax.plot(t, gen_traj[:, 0], 'r-', label='Gen X')
    ax.plot(t, gen_traj[:, 1], 'r--', label='Gen Z')
    
    if gt_traj is not None:
        t_gt = np.arange(len(gt_traj))
        ax.plot(t_gt, gt_traj[:, 0], 'g-', alpha=0.7, label='GT X')
        ax.plot(t_gt, gt_traj[:, 1], 'g--', alpha=0.7, label='GT Z')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Position (meters)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def visualize_sample(output_dir, sample_name, save_fig=True):
    """Visualize a single sample."""
    
    # Load files
    gen_path = os.path.join(output_dir, f"{sample_name}_generated.npy")
    ego_path = os.path.join(output_dir, f"{sample_name}_ego.npy")
    gt_path = os.path.join(output_dir, f"{sample_name}_gt.npy")
    
    if not os.path.exists(gen_path):
        print(f"Error: Generated motion not found at {gen_path}")
        return
    
    generated = np.load(gen_path)
    ego = np.load(ego_path) if os.path.exists(ego_path) else None
    gt_motion = np.load(gt_path) if os.path.exists(gt_path) else None
    
    print(f"Generated shape: {generated.shape}")
    if ego is not None:
        print(f"Ego shape: {ego.shape}")
    if gt_motion is not None:
        print(f"GT shape: {gt_motion.shape}")
    
    # Extract root trajectories
    gen_traj = extract_root_trajectory(generated)
    gt_traj = extract_root_trajectory(gt_motion) if gt_motion is not None else None
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bird's eye view
    plot_bird_eye_view(
        axes[0], 
        ego if ego is not None else np.zeros((2, 2)), 
        gen_traj, 
        gt_traj,
        title=f"Bird's Eye View: {sample_name}"
    )
    
    # Time series
    plot_trajectories_over_time(
        axes[1],
        gen_traj,
        gt_traj,
        title="Root Position over Time"
    )
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = os.path.join(output_dir, f"{sample_name}_visualization.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {fig_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Ego-Conditioned Motion Results")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory with generated .npy files")
    parser.add_argument("--sample_name", type=str, default=None,
                        help="Specific sample to visualize (without extension)")
    parser.add_argument("--all", action="store_true",
                        help="Visualize all samples in output_dir")
    parser.add_argument("--no_save", action="store_true",
                        help="Don't save figures")
    args = parser.parse_args()
    
    if args.sample_name:
        visualize_sample(args.output_dir, args.sample_name, save_fig=not args.no_save)
    elif args.all:
        # Find all generated files
        import glob
        gen_files = glob.glob(os.path.join(args.output_dir, "*_generated.npy"))
        for gen_file in gen_files:
            sample_name = os.path.basename(gen_file).replace("_generated.npy", "")
            print(f"\n{'='*50}")
            print(f"Visualizing: {sample_name}")
            visualize_sample(args.output_dir, sample_name, save_fig=not args.no_save)
    else:
        print("Provide --sample_name or --all")


if __name__ == "__main__":
    main()
