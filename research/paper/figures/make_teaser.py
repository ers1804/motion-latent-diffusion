"""Compose the teaser: one model, one noise seed, two ego trajectories.

Panels come from render_teaser.py. Travel ranges are computed from the generated
samples, so the labels cannot drift from the data.

  python make_teaser.py --npz <teaser npz> --renders <dir> --rows 137 333 --out teaser.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                            "pdf.fonttype": 42, "font.size": 8})
RED, BLUE, PALE = "#b8081f", "#08218c", "#8cadeb"
CROP_TOP, CROP_BOTTOM = .045, .115  # empty ground, to fit the first page


def travel(d, row):
    L = int(d[f"len_{row}"])
    root = d[f"joints_{row}"][:, :L, 0][:, :, [0, 2]]
    return np.linalg.norm(root[:, -1] - root[:, 0], axis=1)


def label_times(ax, labels, shape):
    h, w = shape[:2]
    full = h / (1 - CROP_TOP - CROP_BOTTOM)
    peds = [lab for lab in labels if lab["kind"] == "ped"]
    for lab in labels:
        u, v, _ = lab["uv"]
        y = (1 - v) * full - CROP_TOP * full
        if not (0.01 < u < .99 and 0.01 * h < y < .99 * h):
            continue
        # one pose drawn: the pedestrian stays there the whole time
        text = f"0\u2013{lab['t']:.1f} s" if lab["kind"] == "ped" and len(peds) == 1 else f"{lab['t']:.1f} s"
        ax.annotate(text, (u * w, y), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=BLUE if lab["kind"] == "ped" else RED,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--rows", type=int, nargs=2, required=True)
    ap.add_argument("--titles", nargs=2, default=["Vehicle drives past: pedestrian stays", "Vehicle slows to a stop: pedestrian walks on"])
    ap.add_argument("--out", default="teaser.pdf")
    args = ap.parse_args()
    d = np.load(args.npz)
    images = [mpimg.imread(Path(args.renders) / f"t{r}.png") for r in args.rows]
    n = images[0].shape[0]
    images = [img[int(CROP_TOP * n):n - int(CROP_BOTTOM * n)] for img in images]
    h, w = images[0].shape[:2]
    width, gap = 7.16, .07
    panel_w = (width - gap) / 2
    panel_h = panel_w * h / w
    legend, strip = .17, .17
    H = legend + strip + panel_h
    fig = plt.figure(figsize=(width, H))

    for n, (row, img) in enumerate(zip(args.rows, images)):
        left = n * (panel_w + gap)
        t = travel(d, row)
        fig.text((left + .01) / width, (panel_h + strip / 2) / H, f"({'ab'[n]}) {args.titles[n]}", va="center", fontweight="bold", fontsize=7.8, color="#111")
        fig.text((left + panel_w - .01) / width, (panel_h + strip / 2) / H, f"{t.min():.1f}\u2013{t.max():.1f} m, {len(t)} samples", va="center", ha="right", fontsize=7.4, color="#333")
        ax = fig.add_axes([left / width, 0, panel_w / width, panel_h / H])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        ax.add_patch(Rectangle((0, 0), w - 1, h - 1, fill=False, ec="#9aa0a6", lw=.5))
        label_times(ax, json.load(open(Path(args.renders) / f"t{row}.png.json")), img.shape)

    handles = [Line2D([], [], color=RED, lw=2, label="input: ego trajectory"),
               Line2D([], [], color=BLUE, lw=2, label="generated body motion"),
               Line2D([], [], color=PALE, lw=1.5, label="root paths, other samples")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.2,
               handlelength=1.4, columnspacing=.9, handletextpad=.4, borderaxespad=0, bbox_to_anchor=(.5, 1.0))
    fig.savefig(args.out, dpi=400)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
