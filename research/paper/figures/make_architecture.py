#!/usr/bin/env python3
"""Corrected EgoPed architecture figure: full-sequence ego conditioning via
self-attention over the concatenated [z_t, t_emb, ego tokens] sequence
(the model as actually trained: trans_enc). Replaces the outdated
cross-attention/trans_dec diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#1f2a37"; NAVY = "#1e3a5f"; BLUE = "#1e5ea8"; ORANGE = "#d97706"
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=10.5, tc=None, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or INK, fontweight="bold" if bold else "normal",
            family="monospace")

def arrow(x1, y1, x2, y2, label=None, lx=0.12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.6, color=INK))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2, label, fontsize=8.5,
                color=NAVY, family="monospace", ha="left", va="center")

# title
ax.text(5, 9.62, "EgoPed Architecture", ha="center", fontsize=16,
        color=NAVY, fontweight="bold", family="monospace")
ax.text(5, 9.12, "frozen ego encoder + full-sequence self-attention denoiser",
        ha="center", fontsize=10.5, color=BLUE, family="monospace")

# inputs row
box(0.4, 7.3, 2.6, 1.0, "ego trajectory\n(B, 196, 2)", "#fde8cd", ORANGE)
box(3.7, 7.3, 2.6, 1.0, "noisy latent z_t\n(B, 4, 256)", "#dbe7f5", NAVY)
box(7.0, 7.3, 2.6, 1.0, "timestep t\n(embedding)", "#dbe7f5", NAVY)

# ego encoder
box(0.4, 5.55, 2.6, 1.0, "Ego Encoder (frozen)\n196 x 256-d tokens", "#e9dcf7", "#7c3aed")
arrow(1.7, 7.3, 1.7, 6.55)

# concat
box(1.6, 3.95, 6.8, 0.95,
    "concatenate:  [ z_t (4) | t_emb (1) | ego tokens (196) ]  =  201 tokens",
    "#eef2f6", "#939eaa", fs=10)
arrow(1.7, 5.55, 2.6, 4.9)
arrow(5.0, 7.3, 5.0, 4.9)
arrow(8.3, 7.3, 7.6, 4.9)

# denoiser
box(1.6, 2.35, 6.8, 1.0,
    "Transformer Denoiser (self-attention, only trained)\n"
    "joint attention over all 201 tokens -> read out first 4 tokens",
    "#2e6fbe", "#1e3a5f", fs=10, tc="white", bold=True)
arrow(5.0, 3.95, 5.0, 3.35)

# decoder + output
box(1.6, 0.95, 3.15, 0.85, "VAE Decoder (frozen)\nHumanML3D, 4 x 256", "#dbe7f5", NAVY, fs=9.5)
box(5.25, 0.95, 3.15, 0.85, "pedestrian motion\n(B, 196, 263)", "#d3f0dd", "#15803d", fs=9.5)
arrow(3.2, 2.35, 3.2, 1.8, label="denoised z_0")
arrow(4.75, 1.375, 5.25, 1.375)

# side annotations
ax.text(8.75, 6.05, "* frozen weights preserve\n  generative diversity\n  (unfreezing collapses MM)",
        fontsize=8.5, color=ORANGE, family="monospace", va="center")
ax.text(8.75, 2.85, "* every denoising step sees\n  all 196 ego timesteps\n"
                    "* cross-attention variant\n  performs comparably (ablation)",
        fontsize=8.5, color=ORANGE, family="monospace", va="center")

ax.text(5, 0.25, "Diffusion runs inside the frozen HumanML3D VAE latent space. Only the denoiser is trained.",
        ha="center", fontsize=9, color="#5b6672", family="monospace")

plt.tight_layout()
plt.savefig("architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved architecture.png")
