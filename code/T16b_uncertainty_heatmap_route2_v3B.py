#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T16b_uncertainty_heatmap_route2_v3B.py

Task point 18: Uncertainty heatmap from tidy metrics.

Inputs:
  data/T14b_tidy_metrics_route2_v3B.csv

Outputs:
  - data/fig/T16b_heatmap_pi_hdi_width_mean.png
  - data/T16b_heatmap_pi_hdi_width_mean.csv

Definition:
  pi_hdi_width_mean = mean over active slots of (HDI_high - HDI_low) for pi.

Notes:
- Uses matplotlib only; does not set explicit colors (defaults apply).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy_csv", default="data/T14b_tidy_metrics_route2_v3B.csv")
    ap.add_argument("--value_col", default="pi_hdi_width_mean")
    ap.add_argument("--out_dir", default="data/fig")
    args = ap.parse_args()

    tidy = Path(args.tidy_csv)
    if not tidy.exists():
        raise FileNotFoundError(f"[T16b] tidy csv not found: {tidy}")
    out_dir = Path(args.out_dir)

    df = pd.read_csv(tidy)
    vc = args.value_col
    need = {"season", "week", vc}
    if not need.issubset(df.columns):
        missing = sorted(list(need - set(df.columns)))
        raise ValueError(f"[T16b] missing required columns: {missing}")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    p = df.pivot_table(index="season", columns="week", values=vc, aggfunc="mean")
    p = p.sort_index().reindex(sorted(p.columns), axis=1)

    out_csv = Path(f"data/T16b_heatmap_{vc}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(out_csv)

    arr = p.to_numpy()
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(arr, aspect="auto")
    ax.set_title(f"Uncertainty heatmap: {vc}")
    ax.set_xlabel("week")
    ax.set_ylabel("season")
    ax.set_xticks(np.arange(len(p.columns)))
    ax.set_xticklabels([str(c) for c in p.columns], rotation=90)
    ax.set_yticks(np.arange(len(p.index)))
    ax.set_yticklabels([str(i) for i in p.index])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out_png = out_dir / f"T16b_heatmap_{vc}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print("=" * 90)
    print("[T16b] Saved:", out_csv)
    print("[T16b] Saved:", out_png)
    print("=" * 90)


if __name__ == "__main__":
    main()
