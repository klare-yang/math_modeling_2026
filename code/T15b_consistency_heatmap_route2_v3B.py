#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T15b_consistency_heatmap_route2_v3B.py

Task point 17: Consistency heatmaps (paper-ready figures) from tidy metrics.

Inputs:
  data/T14b_tidy_metrics_route2_v3B.csv

Outputs:
  - data/fig/T15b_heatmap_hard_hit_set.png
  - data/fig/T15b_heatmap_soft_ppc_p_mean.png
  - data/T15b_heatmap_hard_hit_set.csv   (pivot table)
  - data/T15b_heatmap_soft_ppc_p_mean.csv

Notes:
- Uses matplotlib only; does not set explicit colors (defaults apply).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    need = {"season", "week", value_col}
    if not need.issubset(df.columns):
        missing = sorted(list(need - set(df.columns)))
        raise ValueError(f"[T15b] missing required columns: {missing}")
    d = df.copy()
    d["season"] = pd.to_numeric(d["season"], errors="coerce").astype("Int64")
    d["week"] = pd.to_numeric(d["week"], errors="coerce").astype("Int64")
    p = d.pivot_table(index="season", columns="week", values=value_col, aggfunc="mean")
    # sort indices
    p = p.sort_index()
    p = p.reindex(sorted(p.columns), axis=1)
    return p


def _plot_heatmap(pivot: pd.DataFrame, title: str, out_png: Path) -> None:
    arr = pivot.to_numpy()
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(arr, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("week")
    ax.set_ylabel("season")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy_csv", default="data/T14b_tidy_metrics_route2_v3B.csv")
    ap.add_argument("--out_dir", default="data/fig")
    args = ap.parse_args()

    tidy = Path(args.tidy_csv)
    if not tidy.exists():
        raise FileNotFoundError(f"[T15b] tidy csv not found: {tidy}")
    out_dir = Path(args.out_dir)

    df = pd.read_csv(tidy)

    # Hard
    if "hard_hit_set" in df.columns:
        p_hard = _pivot(df, "hard_hit_set")
        p_hard.to_csv(Path("data/T15b_heatmap_hard_hit_set.csv"))
        _plot_heatmap(p_hard, "Hard consistency: hit_set (mean)", out_dir / "T15b_heatmap_hard_hit_set.png")
    else:
        raise ValueError("[T15b] tidy csv missing hard_hit_set")

    # Soft
    if "ppc_p_mean" in df.columns:
        p_soft = _pivot(df, "ppc_p_mean")
        p_soft.to_csv(Path("data/T15b_heatmap_soft_ppc_p_mean.csv"))
        _plot_heatmap(p_soft, "Soft consistency: PPC P(observed elim) mean", out_dir / "T15b_heatmap_soft_ppc_p_mean.png")
    else:
        raise ValueError("[T15b] tidy csv missing ppc_p_mean")

    print("=" * 90)
    print("[T15b] Saved heatmaps to:", out_dir)
    print("=" * 90)


if __name__ == "__main__":
    main()
