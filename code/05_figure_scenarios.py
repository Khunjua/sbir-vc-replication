#!/usr/bin/env python3
"""
05_figure_scenarios.py -- reproduce Figure 1 of the paper.

Bar chart of the four validation-scenario concordance indices (S1
mixed k-fold, S3 within-old, S4 within-recent, S2 cross-break) with
95% bootstrap CIs and a visual annotation of the pooled-vs-time-split
gap.  Numbers are hard-coded from Table 1 of the paper.

Output: results/figure_1_scenarios.png  (220 dpi).
"""
from pathlib import Path
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["text.usetex"] = False

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

scenarios = [
    {
        "short": "S1",
        "label": "Mixed cohorts\n(random 5-fold)",
        "n_val": 8332,
        "c": 0.668,
        "ci_lo": 0.668 - 0.012,
        "ci_hi": 0.668 + 0.012,
        "color": "#3a4a5e",
    },
    {
        "short": "S3",
        "label": "Within early era\n(2010-13 -> 2014-16)",
        "n_val": 1547,
        "c": 0.637,
        "ci_lo": None,
        "ci_hi": None,
        "color": "#6b7a8a",
    },
    {
        "short": "S4",
        "label": "Within recent era\n(2018-20 -> 2021-22)",
        "n_val": 1449,
        "c": 0.616,
        "ci_lo": 0.579,
        "ci_hi": 0.653,
        "color": "#6b7a8a",
    },
    {
        "short": "S2",
        "label": "Cross-cohort\n(2010-18 -> 2019-22)",
        "n_val": 2941,
        "c": 0.565,
        "ci_lo": 0.538,
        "ci_hi": 0.592,
        "color": "#c04e2c",
    },
]

fig, ax = plt.subplots(figsize=(8.2, 5.6))
x = np.arange(len(scenarios))
heights = [s["c"] for s in scenarios]
colors = [s["color"] for s in scenarios]
err_lo = [(s["c"] - s["ci_lo"]) if s["ci_lo"] is not None else 0 for s in scenarios]
err_hi = [(s["ci_hi"] - s["c"]) if s["ci_hi"] is not None else 0 for s in scenarios]

ax.bar(
    x, heights, width=0.62, color=colors, edgecolor="black", linewidth=0.7,
    yerr=[err_lo, err_hi], capsize=7, ecolor="black",
    error_kw={"linewidth": 1.0, "capthick": 1.0},
)

for i, s in enumerate(scenarios):
    eu = err_hi[i] if s["ci_hi"] is not None else 0
    label = f"{s['c']:.3f}"
    if s["ci_lo"] is not None:
        label = f"{s['c']:.3f}\n[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
    ax.text(i, s["c"] + eu + 0.006, label, ha="center", va="bottom",
            fontsize=9.5, fontweight="bold" if i in (0, 3) else "normal")

for i, s in enumerate(scenarios):
    ax.text(i, 0.508, f"n_val = {s['n_val']:,}", ha="center", va="bottom",
            fontsize=8.5, color="#555")

ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.9)
ax.text(3.45, 0.503, "random ranking", fontsize=8, color="gray",
        ha="right", style="italic")

gap = scenarios[0]["c"] - scenarios[3]["c"]
inflation_pct = gap / scenarios[3]["c"] * 100
ax.annotate(
    "", xy=(3, 0.591), xytext=(0, 0.665),
    arrowprops=dict(arrowstyle="<->", color="#c04e2c", linewidth=1.3, mutation_scale=15),
)
ax.text(
    1.5, 0.628,
    f"gap ~ {gap:.3f}\n({inflation_pct:.0f}% relative)",
    ha="center", va="center",
    fontsize=10, color="#c04e2c", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
              edgecolor="#c04e2c", linewidth=0.8),
)

ax.set_ylabel("Concordance Index", fontsize=11.5)
ax.set_xticks(x)
ax.set_xticklabels([s["label"] for s in scenarios], fontsize=10)
ax.set_ylim(0.49, 0.72)
ax.set_yticks(np.arange(0.50, 0.72, 0.05))
ax.tick_params(axis="y", labelsize=10)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.grid(axis="y", linestyle="--", linewidth=0.4, color="#bbb", alpha=0.6)
ax.set_axisbelow(True)

ax.set_title("Predictive Performance by Validation Strategy",
             fontsize=12.5, fontweight="bold", pad=12)

plt.tight_layout()
png_path = OUT / "figure_1_scenarios.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Wrote: results/{png_path.name}")
