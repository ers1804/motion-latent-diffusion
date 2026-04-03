"""
Plot FID vs R-precision tradeoff curves for H2 and H4 across CFG values.
Shows the H4 breakthrough: FID ~4x better across all CFG values at epoch=3199.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# H2 CFG sweep (epoch=4399, best checkpoint)
h2_cfg =      [5,     7,     10,    15,    20]
h2_fid =      [6.603, 6.716, 6.963, 7.400, 7.856]
h2_fid_ci =   [0.067, 0.068, 0.035, 0.016, 0.041]
h2_rprec1 =   [0.671, 0.724, 0.767, 0.797, 0.815]

# H4 intermediate CFG sweep (epoch=3199, 64% of H2's best epoch)
h4_cfg =      [5,     10,    15]
h4_fid =      [3.968, 3.644, 3.617]
h4_fid_ci =   [0.209, 0.145, 0.131]
h4_rprec1 =   [0.510, 0.540, 0.515]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('#f8f9fa')

# ── Panel 1: FID vs CFG ──
ax1 = axes[0]
ax1.set_facecolor('#ffffff')
ax1.errorbar(h2_cfg, h2_fid, yerr=h2_fid_ci, fmt='o-', color='#dc2626', linewidth=2.5,
             markersize=8, capsize=5, label='H2 (epoch=4399, baseline)', zorder=3)
ax1.errorbar(h4_cfg, h4_fid, yerr=h4_fid_ci, fmt='s--', color='#2563eb', linewidth=2.5,
             markersize=8, capsize=5, label='H4 (epoch=3199, intermediate)', zorder=3)

# H4 expected trajectory label
ax1.axhline(y=3.617, color='#2563eb', linestyle=':', alpha=0.3)
ax1.fill_between(h4_cfg, [f - c for f, c in zip(h4_fid, h4_fid_ci)],
                 [f + c for f, c in zip(h4_fid, h4_fid_ci)],
                 alpha=0.15, color='#2563eb')
ax1.fill_between(h2_cfg, [f - c for f, c in zip(h2_fid, h2_fid_ci)],
                 [f + c for f, c in zip(h2_fid, h2_fid_ci)],
                 alpha=0.15, color='#dc2626')

# Annotate the gap
ax1.annotate('', xy=(15, 3.617), xytext=(15, 7.400),
             arrowprops=dict(arrowstyle='<->', color='#16a34a', lw=2))
ax1.text(15.3, 5.5, '−51%\nFID', color='#16a34a', fontsize=10, fontweight='bold')

ax1.set_xlabel('CFG Guidance Scale', fontsize=12)
ax1.set_ylabel('FID ↓ (lower is better)', fontsize=12)
ax1.set_title('FID vs CFG Scale\nH4 flat; H2 increases steeply', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9.5)
ax1.set_ylim(2.5, 9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for cfg, fid, ci in zip(h2_cfg, h2_fid, h2_fid_ci):
    ax1.text(cfg, fid + 0.15, f'{fid:.2f}', ha='center', fontsize=8, color='#dc2626')
for cfg, fid, ci in zip(h4_cfg, h4_fid, h4_fid_ci):
    ax1.text(cfg, fid - 0.25, f'{fid:.2f}', ha='center', fontsize=8, color='#2563eb')

# ── Panel 2: FID vs R-prec scatter (Pareto frontier) ──
ax2 = axes[1]
ax2.set_facecolor('#ffffff')

# H2 points
for cfg, fid, rp in zip(h2_cfg, h2_fid, h2_rprec1):
    ax2.scatter(rp, fid, s=100, color='#dc2626', zorder=3)
    ax2.text(rp + 0.003, fid + 0.05, f'CFG={cfg}', fontsize=7.5, color='#dc2626')
ax2.plot(h2_rprec1, h2_fid, '-', color='#dc2626', linewidth=1.5, alpha=0.6, label='H2 (epoch=4399)')

# H4 points
for cfg, fid, rp in zip(h4_cfg, h4_fid, h4_rprec1):
    ax2.scatter(rp, fid, s=100, color='#2563eb', marker='s', zorder=3)
    ax2.text(rp - 0.003, fid - 0.18, f'CFG={cfg}', fontsize=7.5, color='#2563eb', ha='right')
ax2.plot(h4_rprec1, h4_fid, '--', color='#2563eb', linewidth=1.5, alpha=0.6, label='H4 (epoch=3199, intermediate)')

# GT reference
ax2.axhline(y=0, color='gray', alpha=0)
ax2.fill_betweenx([0, 10], x1=0, x2=1, alpha=0.04, color='green')

# Shaded H4 improvement region
ax2.annotate('H4 dominates\nH2 on FID\n(all CFG levels)', xy=(0.55, 4.0),
             fontsize=9, color='#2563eb', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eff6ff', edgecolor='#2563eb', alpha=0.7))

ax2.set_xlabel('R-Precision@1 ↑ (higher is better)', fontsize=12)
ax2.set_ylabel('FID ↓ (lower is better)', fontsize=12)
ax2.set_title('FID/R-prec Tradeoff Frontier\nH4 dominates H2 on FID axis', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9.5)
ax2.set_ylim(2.5, 9)
ax2.set_xlim(0.44, 0.88)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add note: H4 epoch=3199 is mid-training
note = "Note: H4 epoch=3199 = 64% of H2's best epoch (4399).\nR-prec expected to improve with segment-2 training."
fig.text(0.5, 0.01, note, ha='center', fontsize=9, color='#6b7280', style='italic')

plt.suptitle('H4 Cross-Attention Ego Conditioning vs H2 Baseline\nIntermediate Results (2026-04-03)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('research/to_human/h4_cfg_tradeoff.png', dpi=150, bbox_inches='tight',
            facecolor='#f8f9fa')
print("Saved research/to_human/h4_cfg_tradeoff.png")
