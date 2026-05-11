"""H4 training trajectory: shows best checkpoint at ep=3399 with monotonic degradation after."""
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

# H4 training-time validation trajectory (run_020 in research-state.yaml)
h4 = [
    (3199, 3.766, 0.529),
    (3399, 3.770, 0.535),  # BEST
    (3499, 3.805, 0.531),
    (3599, 3.882, 0.530),
    (3799, 3.955, 0.522),
    (3999, 4.087, 0.515),
    (4199, 4.178, 0.514),
    (4399, 4.241, 0.516),
    (4599, 4.272, 0.512),
    (4799, 4.382, 0.510),
    (4999, 4.508, 0.511),
]
epochs, fids, r1s = zip(*h4)

fig, ax_fid = plt.subplots(figsize=(5.0, 2.8))
ax_r1 = ax_fid.twinx()

c_fid = "#c0392b"
c_r1 = "#2c3e50"

ax_fid.plot(epochs, fids, color=c_fid, marker="o", markersize=3.5, label="FID")
ax_r1.plot(epochs, r1s, color=c_r1, marker="s", markersize=3.5, label="R@1")

# Mark the best checkpoint
best_idx = 1  # epoch=3399
ax_fid.axvline(epochs[best_idx], color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
ax_fid.annotate(
    f"best ckpt\nep={epochs[best_idx]}",
    xy=(epochs[best_idx], fids[best_idx]),
    xytext=(epochs[best_idx] + 250, fids[best_idx] - 0.15),
    fontsize=7, color="black",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
)

ax_fid.set_xlabel("Epoch")
ax_fid.set_ylabel("FID $\\downarrow$", color=c_fid)
ax_r1.set_ylabel("R@1 $\\uparrow$", color=c_r1)

ax_fid.tick_params(axis="y", colors=c_fid)
ax_r1.tick_params(axis="y", colors=c_r1)
ax_fid.spines["left"].set_color(c_fid)
ax_r1.spines["right"].set_color(c_r1)
ax_fid.spines["top"].set_visible(False)
ax_r1.spines["top"].set_visible(False)

ax_fid.grid(True, axis="y", alpha=0.25, linewidth=0.5)
ax_fid.set_xlim(min(epochs) - 50, max(epochs) + 50)

# Combined legend in upper-right
lines_fid, _ = ax_fid.get_legend_handles_labels()
lines_r1,  _ = ax_r1.get_legend_handles_labels()
ax_fid.legend(lines_fid + lines_r1, ["FID", "R@1"],
              loc="upper left", ncol=2, frameon=False, fontsize=8)

out = "/home/erik/ssd2/gitprojects/motion-latent-diffusion/research/paper/figures/h4_training_curve.pdf"
plt.savefig(out)
print(f"Wrote {out}")
