"""Test FID against per-condition accuracy, one point per training run (Fig. fid_vs_conditioning).

Reads the test report and the K=5 test dumps of the selected checkpoints; run from the repository root:
  python research/paper/figures/make_fid_vs_conditioning.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "src"))
from compare_resplit import cell_of, load, metrics  # noqa: E402

DATA = ROOT / "research" / "data"
OUT = Path(__file__).resolve().parent / "fid_vs_conditioning.pdf"

mpl.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK, MUTED, GRID = "#1f2a37", "#6b7280", "#e5e7eb"
# Validated categorical slots 1-3 (dataviz palette); the ego-blind priors are a neutral reference, not a series.
FAMILY = {"egoped": ("EgoPed", "#2a78d6"), "fullseq": ("Full-seq", "#eb6834"),
          "unfrozen": ("Full-seq, trainable encoder", "#1baf7a"), "prior": ("Trained prior (no ego)", "#8c8c8c")}
CELL = {"pooled": "egoped", "pooled_ia": "egoped", "egoped": "fullseq", "egoped_ia": "fullseq",
        "unfrozen": "unfrozen", "uncond": "prior", "uncond_ia": "prior"}


def main():
    fid = {}
    for line in open(DATA / "resplit_test_report.txt"):
        if line.startswith("eval_test_rs1"):
            name = re.sub(r"_ep\d+_cfg[\d.]+$", "", line.split()[0].removeprefix("eval_test_"))
            fid[name] = float(line.split()[1])
    sel = json.load(open(DATA / "resplit_selection.json"))
    runs = []
    for run, s in sorted(sel.items()):
        m = metrics(load(DATA / f"ade_fde_{run}_ep{s['epoch']}_cfg{s['guidance']}_scene_test_k5.npz"), slice(None))
        cell = cell_of(run)
        runs.append(dict(run=run, cell=cell, family=CELL[cell], ia=cell.endswith("_ia"), fid=fid[run], **m))

    f = np.array([r["fid"] for r in runs])
    rho_ade = spearmanr(f, [r["ADE"] for r in runs]).correlation
    rho_ecs = spearmanr(f, [r["ECS"] for r in runs]).correlation
    print(f"{len(runs)} runs; Spearman FID vs ADE {rho_ade:+.2f}, FID vs ECS {rho_ecs:+.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharex=True)
    for ax, key, label in ((axes[0], "ADE", "Mean ADE (m) $\\downarrow$"), (axes[1], "ECS", "ECS $\\uparrow$")):
        for r in runs:
            name, color = FAMILY[r["family"]]
            ax.plot(r["fid"], r[key], marker="^" if r["ia"] else "o", ms=4.6, ls="none", color=color,
                    mec="white", mew=0.6, zorder=3)
        ax.set_ylabel(label, color=INK)
        ax.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.tick_params(colors=MUTED, length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.set_xlabel("Test FID $\\downarrow$", color=INK)
    axes[0].set_title(f"(a) Spearman $\\rho = {rho_ade:+.2f}$", fontsize=7, color=INK, loc="left")
    axes[1].set_title(f"(b) Spearman $\\rho = {rho_ecs:+.2f}$", fontsize=7, color=INK, loc="left")

    handles = [Line2D([], [], marker="s", ls="none", ms=4.5, color=c, label=n) for n, c in FAMILY.values()]
    handles += [Line2D([], [], marker="o", ls="none", ms=4.5, color=MUTED, mfc="none", label="uniform data"),
                Line2D([], [], marker="^", ls="none", ms=4.5, color=MUTED, mfc="none", label="IA data")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2),
               handletextpad=0.3, columnspacing=0.9, labelcolor=INK)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
