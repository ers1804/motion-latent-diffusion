"""Compose qualitative figure: 3 samples × 5 timesteps with trajectory thumbnails."""
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

samples = ["0002_26", "0005_26", "0008_37"]
timesteps = [1, 2, 3, 4, 5]  # frame indices in /tmp/qual_frames
labels_t = ["t=5", "t=40", "t=80", "t=120", "t=160"]
frames_dir = Path("/tmp/qual_frames")
traj_dir = Path("/home/erik/ssd2/gitprojects/motion-latent-diffusion/outputs/ego_demo/ego_motion_diffusion_h4_trans_dec")

fig, axes = plt.subplots(
    nrows=len(samples),
    ncols=len(timesteps) + 1,
    figsize=(11.0, 4.5),
    gridspec_kw={"width_ratios": [1.0] * len(timesteps) + [1.6]},
)

for r, s in enumerate(samples):
    for c, t in enumerate(timesteps):
        ax = axes[r][c]
        img = mpimg.imread(frames_dir / f"{s}_{t}.png")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if r == 0:
            ax.set_title(labels_t[c], fontsize=9)
    # Trajectory thumbnail
    ax_t = axes[r][-1]
    traj_img = mpimg.imread(traj_dir / f"{s}_trajectories.png")
    ax_t.imshow(traj_img)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    for spine in ax_t.spines.values():
        spine.set_visible(False)
    if r == 0:
        ax_t.set_title("BEV trajectory", fontsize=9)
    # Row label on the left
    axes[r][0].set_ylabel(f"sample {s}", fontsize=8, rotation=90, labelpad=5)

plt.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.02, wspace=0.04, hspace=0.05)
out = "/home/erik/ssd2/gitprojects/motion-latent-diffusion/research/paper/figures/qualitatives.pdf"
plt.savefig(out, bbox_inches="tight", pad_inches=0.04, dpi=200)
print(f"Wrote {out}")
