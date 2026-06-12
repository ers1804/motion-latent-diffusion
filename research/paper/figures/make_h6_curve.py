"""Generate H6 training trajectory figure for the paper.
FID and R@1 vs epoch on twin axes with H4 reference lines.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# H6 training-time validation trajectory, extracted 2026-06-12 directly from the
# SLURM logs (all val points, every 100 epochs):
#   seg-1: diffusion_h6_unfreeze_ego_365378.txt
#   seg-2: diffusion_h6_seg2_368901.txt
# (epoch, FID, R@1)
h6_seg1 = [
    (99,    9.835, 0.036),
    (199,   9.351, 0.032),
    (299,   7.926, 0.036),
    (399,   8.624, 0.038),
    (499,   8.373, 0.038),
    (599,   7.292, 0.034),
    (699,   8.443, 0.037),
    (799,   6.664, 0.044),
    (899,   6.277, 0.052),
    (999,   6.011, 0.065),
    (1099,  5.283, 0.079),
    (1199,  5.693, 0.099),
    (1299,  5.375, 0.125),
    (1399,  5.367, 0.140),
    (1499,  5.053, 0.158),
    (1599,  5.076, 0.177),
    (1699,  5.351, 0.222),
    (1799,  5.180, 0.238),
    (1899,  4.857, 0.257),
    (1999,  4.967, 0.271),
    (2099,  4.789, 0.273),
    (2199,  5.262, 0.311),
    (2299,  4.918, 0.311),
    (2399,  5.320, 0.317),
    (2499,  5.385, 0.332),
    (2599,  5.453, 0.336),
    (2699,  5.283, 0.355),
    (2799,  5.826, 0.352),
    (2899,  5.836, 0.350),
    (2999,  5.889, 0.373),
    (3099,  5.657, 0.363),
    (3199,  5.570, 0.358),
    (3299,  5.273, 0.398),
]
# seg-2 trajectory (resume from ep=3299, ran to END_EPOCH=5000)
h6_seg2 = [
    (3299, 5.273, 0.398),
    (3399, 5.141, 0.380),
    (3499, 5.415, 0.367),
    (3599, 5.631, 0.383),
    (3699, 6.042, 0.400),
    (3799, 5.908, 0.389),
    (3899, 5.213, 0.390),
    (3999, 6.221, 0.407),
    (4099, 5.497, 0.401),
    (4199, 5.526, 0.390),
    (4299, 5.465, 0.381),
    (4399, 5.659, 0.397),
    (4499, 5.854, 0.392),
    (4599, 6.339, 0.417),
    (4699, 5.854, 0.395),
    (4799, 5.542, 0.404),
    (4899, 5.426, 0.400),
    (4999, 5.561, 0.408),
]

epochs = [e for e, *_ in h6_seg1] + [e for e, *_ in h6_seg2[1:]]
fids = [f for _, f, _ in h6_seg1] + [f for _, f, _ in h6_seg2[1:]]
r1s = [r for _, _, r in h6_seg1] + [r for _, _, r in h6_seg2[1:]]

# H4 reference (best operating point: epoch=3399, CFG=10)
H4_FID = 3.392
H4_R1 = 0.548

fig, ax_fid = plt.subplots(figsize=(5.0, 2.8))
ax_r1 = ax_fid.twinx()

c_fid = "#c0392b"
c_r1 = "#2c3e50"

l_fid, = ax_fid.plot(epochs, fids, color=c_fid, marker="o", markersize=3.0, label="H6 FID")
l_r1,  = ax_r1.plot(epochs, r1s,  color=c_r1, marker="s", markersize=3.0, label="H6 R@1")

ax_fid.axhline(H4_FID, color=c_fid, linestyle="--", linewidth=0.9, alpha=0.7)
ax_r1.axhline(H4_R1,  color=c_r1,  linestyle="--", linewidth=0.9, alpha=0.7)

ax_fid.text(epochs[-1] + 50, H4_FID + 0.05, f"H4 FID = {H4_FID}",
            color=c_fid, fontsize=7, ha="right", va="bottom")
ax_r1.text(epochs[0] + 50, H4_R1 + 0.01, f"H4 R@1 = {H4_R1}",
           color=c_r1, fontsize=7, ha="left", va="bottom")

ax_fid.set_xlabel("Epoch")
ax_fid.set_ylabel("FID $\\downarrow$", color=c_fid)
ax_r1.set_ylabel("R@1 $\\uparrow$", color=c_r1)

ax_fid.tick_params(axis="y", colors=c_fid)
ax_r1.tick_params(axis="y", colors=c_r1)
ax_fid.spines["left"].set_color(c_fid)
ax_r1.spines["right"].set_color(c_r1)
ax_fid.spines["top"].set_visible(False)
ax_r1.spines["top"].set_visible(False)

# Annotate peak R@1
peak_idx = r1s.index(max(r1s))
ax_r1.annotate(
    f"max R@1\n(ep={epochs[peak_idx]}, {r1s[peak_idx]:.3f})",
    xy=(epochs[peak_idx], r1s[peak_idx]),
    xytext=(epochs[peak_idx] - 700, r1s[peak_idx] - 0.10),
    fontsize=7, color=c_r1,
    arrowprops=dict(arrowstyle="->", color=c_r1, lw=0.5),
)

ax_fid.grid(True, axis="y", alpha=0.25, linewidth=0.5)
ax_fid.set_xlim(0, max(epochs) + 100)

# Combined legend
lines = [l_fid, l_r1]
ax_fid.legend(lines, [l.get_label() for l in lines], loc="upper center",
              ncol=2, frameon=False, fontsize=8)

out = "/home/erik/ssd2/gitprojects/motion-latent-diffusion/research/paper/figures/h6_training_curve.pdf"
plt.savefig(out)
print(f"Wrote {out}")
