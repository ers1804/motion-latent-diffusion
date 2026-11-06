#!/usr/bin/env python3
"""Write generated test-set motions in the layout of the CARLA port (one folder per condition, joints as .npy).

Input: the joints npz of teaser_generate.py and the condition list of select_qualitatives.py.
  python research/src/export_qualitatives.py --npz research/data/teaser_rs1_pooled_qual_scene_test.npz \
      --conditions research/data/qualitatives_rs1_pooled_scene_test.json --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intention_labels import is_stopping  # noqa: E402

INK, MUTED, GRID = "#1f2a37", "#6b7280", "#e5e7eb"
EGO, GT, GEN = "#d03b3b", "#1f2a37", "#2a78d6"


def folder_name(i, c):
    category = "" if c["category"] == c["source"] else f"{c['category']}_"  # AVA rows are all in, not a category
    return f"{i:02d}_{category}{c['source']}_{Path(c['path']).stem}"


def draw(ax, c, ego, gt, gen):
    for k, g in enumerate(gen):
        ax.plot(g[:, 0, 0], g[:, 0, 2], color=GEN, lw=.8, alpha=.55, label="generated root (8 samples)" if k == 0 else None)
    ax.plot(gt[:, 0, 0], gt[:, 0, 2], color=GT, lw=1.6, label="ground-truth root")
    ax.plot(ego[:, 0], ego[:, 1], color=EGO, lw=1.6, label="ego vehicle")
    ax.plot(ego[0, 0], ego[0, 1], "o", color=EGO, ms=4, mec="white", mew=.6)
    ax.plot(0, 0, "o", color=GT, ms=4, mec="white", mew=.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color=GRID, lw=.4)
    ax.tick_params(colors=MUTED, labelsize=6, length=2)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(.5)
    gt_word = "stops" if c["gt_stop"] else "walks"
    ax.set_title(f"{c['category']} | {c['source']} {Path(c['path']).stem}\nGT {gt_word}, stops in {c['stops_in_samples']}/{len(gen)}, "
                 f"ADE {c['ade']:.1f} m, {c['L'] / 20:.1f} s", fontsize=6.5, color=INK, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    conditions = json.load(open(args.conditions))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = 4
    sheet, axes = plt.subplots((len(conditions) + cols - 1) // cols, cols, figsize=(cols * 3.2, 3.0 * ((len(conditions) + cols - 1) // cols)))
    axes = axes.ravel()
    index = []
    for i, c in enumerate(conditions):
        row, L = c["row"], c["L"]
        assert str(d[f"path_{row}"]) == c["path"] and int(d[f"len_{row}"]) == L, (row, c["path"])
        gen, gt, ego = d[f"joints_{row}"][:, :L], d[f"gt_{row}"][:L], d[f"ego_{row}"][:L]
        c = dict(c, stops_in_samples=int(sum(is_stopping(g[:, 0][:, [0, 2]]) for g in gen)))
        f = out / folder_name(i, c)
        f.mkdir(exist_ok=True)
        np.save(f / "ego.npy", ego.astype("float32"))
        np.save(f / "gt_joints.npy", gt.astype("float32"))
        for k, g in enumerate(gen):
            np.save(f / f"sample{k}_joints.npy", g.astype("float32"))
        meta = dict(c, folder=f.name, samples=len(gen), frames=L, fps=20)
        (f / "meta.json").write_text(json.dumps(meta, indent=1))
        fig, ax = plt.subplots(figsize=(4, 4))
        draw(ax, c, ego, gt, gen)
        ax.legend(fontsize=6, frameon=False, loc="best")
        fig.savefig(f / "overview.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        draw(axes[i], c, ego, gt, gen)
        index.append(meta)
    for ax in axes[len(conditions):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    sheet.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    sheet.tight_layout(rect=(0, .02, 1, 1))
    sheet.savefig(out / "overview_all.png", dpi=130)
    (out / "conditions.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} conditions to {out}")


if __name__ == "__main__":
    main()
