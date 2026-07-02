#!/usr/bin/env python3
"""
Generate Pareto plot using known baseline data + param counts from Python backbones.
Run from the repo root: python scripts/generate_pareto.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Known results from existing .pt checkpoints & paper ─────────────────────
# Compression ratios from the paper / existing experiment files
known_results = [
    # Classical codecs (no model, size=0)
    {"backbone": "LZMA",        "ratio": 3.22, "bpp": 2.490, "params": 0,       "model_size_mb": 0.00},
    {"backbone": "ZLIB",        "ratio": 2.56, "bpp": 3.130, "params": 0,       "model_size_mb": 0.00},
    # Mamba (from paper)
    {"backbone": "Mamba",       "ratio": 4.03, "bpp": 1.990, "params": 1100000, "model_size_mb": 4.40},
]

# ── Estimate param counts from our Python backbones ──────────────────────────
def count_params(backbone_name, d_model=256, num_layers=1):
    import importlib
    mod = importlib.import_module(f"{backbone_name.lower()}_backbone")
    model = mod.BoaConstrictor(d_model=d_model, num_layers=num_layers)
    return sum(p.numel() for p in model.parameters())

backbone_cfg = [
    ("GRU",         256, 1),
    ("LSTM",        256, 1),
    ("MinGRU",      256, 1),
    ("Transformer", 256, 1),
]

model_entries = []
for name, d, nl in backbone_cfg:
    try:
        p = count_params(name, d, nl)
        model_entries.append({
            "backbone": name,
            "ratio": None,          # Not yet trained — placeholder
            "bpp":   None,
            "params": p,
            "model_size_mb": p * 4 / 1e6,
        })
        print(f"  {name}: {p:,} params  ({p*4/1e6:.2f} MB)")
    except Exception as e:
        print(f"  {name}: FAILED — {e}")

# ── Build DataFrame ──────────────────────────────────────────────────────────
df_known   = pd.DataFrame(known_results)
df_models  = pd.DataFrame(model_entries)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#ffffff")

COLORS = {
    "LZMA":        "#555555",
    "ZLIB":        "#888888",
    "Mamba":       "#8b5cf6",
    "GRU":         "#2563eb",
    "LSTM":        "#16a34a",
    "MinGRU":      "#d97706",
    "Transformer": "#dc2626",
}
MARKERS = {"classical": "D", "trained": "o", "untrained": "s"}

# Plot classical codecs as horizontal dashed reference lines
for _, row in df_known[df_known["params"] == 0].iterrows():
    ax.axhline(row["ratio"], color=COLORS[row["backbone"]], linestyle="--",
               linewidth=1.4, alpha=0.7, zorder=1)
    ax.text(0.02, row["ratio"] + 0.04, f'{row["backbone"]}  {row["ratio"]:.2f}×',
            color=COLORS[row["backbone"]], fontsize=9, va="bottom", fontweight="bold")

# Plot Mamba (trained, known ratio)
for _, row in df_known[df_known["params"] > 0].iterrows():
    ax.scatter(row["model_size_mb"], row["ratio"],
               color=COLORS[row["backbone"]], s=180, zorder=5,
               marker=MARKERS["trained"], edgecolors="white", linewidths=1.5)
    ax.annotate(f'  {row["backbone"]}\n  {row["ratio"]:.2f}×',
                (row["model_size_mb"], row["ratio"]),
                fontsize=9, color=COLORS[row["backbone"]], fontweight="bold")

# Plot our new backbones (param count known, ratio TBD after training)
for _, row in df_models.iterrows():
    ax.scatter(row["model_size_mb"], 0,  # y=0 placeholder
               color=COLORS.get(row["backbone"], "gray"), s=160, zorder=5,
               marker=MARKERS["untrained"], edgecolors="white", linewidths=1.5,
               alpha=0.35)
    ax.annotate(f'  {row["backbone"]}\n  {row["model_size_mb"]:.2f} MB\n  (ratio TBD)',
                (row["model_size_mb"], 0.05),
                fontsize=8, color=COLORS.get(row["backbone"], "gray"),
                rotation=0, va="bottom")

# Axis styling
ax.set_xlabel("Model Size (MB)", fontsize=12, labelpad=8)
ax.set_ylabel("Compression Ratio (×)", fontsize=12, labelpad=8)
ax.set_title("Pareto Frontier: Compression Ratio vs. Model Size\n"
             "(BOA Constrictor — CMS Dataset)", fontsize=14, fontweight="bold", pad=14)
ax.set_xlim(-0.2, 6.0)
ax.set_ylim(-0.2, 5.0)
ax.grid(True, alpha=0.35, linestyle=":")

# Legend
legend_handles = [
    mpatches.Patch(color=COLORS["LZMA"],        label="LZMA baseline (3.22×)"),
    mpatches.Patch(color=COLORS["ZLIB"],        label="ZLIB baseline (2.56×)"),
    mpatches.Patch(color=COLORS["Mamba"],       label="Mamba — 4.03× (trained)"),
    mpatches.Patch(color=COLORS["GRU"],         label=f'GRU (C++/Python)'),
    mpatches.Patch(color=COLORS["LSTM"],        label=f'LSTM (C++/Python)'),
    mpatches.Patch(color=COLORS["MinGRU"],      label=f'MinGRU (C++/Python)'),
    mpatches.Patch(color=COLORS["Transformer"], label=f'Transformer (Python)'),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
          framealpha=0.9, edgecolor="#cccccc")

ax.text(0.01, 0.01,
        "■ = new backbones (ratio pending training)   ● = trained result   ◆ = codec baseline",
        transform=ax.transAxes, fontsize=7.5, color="#555555", va="bottom")

plt.tight_layout()
out = "pareto_plot.png"
plt.savefig(out, dpi=200)
print(f"\nPareto plot saved → {out}")
