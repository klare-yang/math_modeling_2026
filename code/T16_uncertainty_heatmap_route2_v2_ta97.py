#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T16 — Uncertainty heatmap (contestant × week/event)

Outputs (tag: route2_v2_ta97):
- fig/T16_uncertainty_heatmap_share_ci_width_all_route2_v2_ta97.(png|pdf)
- fig/T16_uncertainty_heatmap_share_ci_width_season34_route2_v2_ta97.(png|pdf)

Default metric: share_ci_width (recommended for Q2).
You can switch to logV_centered_ci_width / share_sd / share_cv, etc.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TAG = "route2_v2_ta97"

# ----- inputs -----
T12_CSV = f"data/T12_uncertainty_{TAG}.csv"  # expected local name
# If you kept the earlier name, use:
# T12_CSV = "data/T12_uncertainty_route2_v2_ta97.csv"

# ----- outputs -----
FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)

def _robust_vmax(values, q=0.99):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.quantile(values, q))

def combined_heatmap(df, metric, fname_prefix, vmax=None):
    """
    Heatmap across all seasons; columns are events ordered by (season, week).
    Rows are contestants ordered by mean(metric) descending (more uncertain at top).
    """
    events = (
        df[["season", "week", "event_id"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )
    event_order = {eid: i for i, eid in enumerate(events["event_id"].tolist())}

    cont_stats = df.groupby(["contestant_idx", "contestant_key"], as_index=False)[metric].mean()
    cont_stats = cont_stats.sort_values(metric, ascending=False).reset_index(drop=True)
    cont_order = cont_stats["contestant_idx"].tolist()
    cont_to_row = {c: i for i, c in enumerate(cont_order)}

    n_rows, n_cols = len(cont_order), len(events)
    mat = np.full((n_rows, n_cols), np.nan, dtype=float)

    # fill matrix
    for r in df.itertuples(index=False):
        row = cont_to_row.get(r.contestant_idx)
        col = event_order.get(r.event_id)
        if row is None or col is None:
            continue
        mat[row, col] = getattr(r, metric)

    if vmax is None:
        vmax = _robust_vmax(mat, q=0.99)

    plt.figure(figsize=(24, 12))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    plt.colorbar(im, fraction=0.02, pad=0.01, label=metric)
    plt.xlabel("event (ordered by season-week)")
    plt.ylabel("contestant (sorted by mean uncertainty)")
    plt.title(f"T16 Uncertainty heatmap ({metric}) — all seasons")

    # draw season boundary lines
    season_starts = events.groupby("season").head(1).index.values
    for x in season_starts:
        plt.axvline(x - 0.5, linewidth=0.5)

    # sparse x tick labels (about 10 season labels)
    seasons = events["season"].unique()
    step = max(1, len(seasons) // 10)
    label_seasons = seasons[::step]
    tick_pos, tick_lab = [], []
    for s in label_seasons:
        idxs = events.index[events["season"] == s].to_numpy()
        tick_pos.append(int(idxs.mean()))
        tick_lab.append(f"S{s}")
    plt.xticks(tick_pos, tick_lab, rotation=0)
    plt.yticks([])  # too many to show

    out_png = os.path.join(FIG_DIR, f"T16_uncertainty_heatmap_{fname_prefix}_all_{TAG}.png")
    out_pdf = os.path.join(FIG_DIR, f"T16_uncertainty_heatmap_{fname_prefix}_all_{TAG}.pdf")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()
    return out_png, out_pdf

def season_heatmap(df, season, metric, fname_prefix, vmax=None, top_labels=25):
    """
    Heatmap within a single season; columns are weeks, rows are contestants.
    Shows y-labels only for top_labels most uncertain contestants (by mean(metric)).
    """
    d = df[df["season"] == season].copy()
    weeks = sorted(d["week"].unique())

    cont_stats = d.groupby(["contestant_idx", "contestant_key"], as_index=False)[metric].mean()
    cont_stats = cont_stats.sort_values(metric, ascending=False).reset_index(drop=True)

    cont_order = cont_stats["contestant_idx"].tolist()
    cont_labels = cont_stats["contestant_key"].tolist()
    cont_to_row = {c: i for i, c in enumerate(cont_order)}
    week_to_col = {w: i for i, w in enumerate(weeks)}

    mat = np.full((len(cont_order), len(weeks)), np.nan, dtype=float)
    for r in d.itertuples(index=False):
        row = cont_to_row.get(r.contestant_idx)
        col = week_to_col.get(r.week)
        if row is None or col is None:
            continue
        mat[row, col] = getattr(r, metric)

    if vmax is None:
        vmax = _robust_vmax(mat, q=0.99)

    plt.figure(figsize=(10, max(6, len(cont_order) * 0.08)))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    plt.colorbar(im, fraction=0.03, pad=0.02, label=metric)
    plt.xlabel("week")
    plt.ylabel("contestant")
    plt.title(f"T16 Uncertainty heatmap ({metric}) — Season {season}")
    plt.xticks(np.arange(len(weeks)), [str(w) for w in weeks])

    show_n = min(top_labels, len(cont_order))
    yticks = np.arange(show_n)
    ylabels = [cont_labels[i].replace(f"S{season}:", "") for i in range(show_n)]
    plt.yticks(yticks, ylabels)
    if len(cont_order) > show_n:
        plt.gca().text(len(weeks) - 0.5, show_n + 1, f"+{len(cont_order)-show_n} more",
                       ha="right", va="top")

    out_png = os.path.join(FIG_DIR, f"T16_uncertainty_heatmap_{fname_prefix}_season{season}_{TAG}.png")
    out_pdf = os.path.join(FIG_DIR, f"T16_uncertainty_heatmap_{fname_prefix}_season{season}_{TAG}.pdf")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()
    return out_png, out_pdf

def main():
    df = pd.read_csv(T12_CSV)

    # Choose one:
    metric = "share_ci_width"           # recommended (interpretable)
    # metric = "share_sd"
    # metric = "share_cv"
    # metric = "logV_centered_ci_width"
    # metric = "logV_centered_sd"

    fname_prefix = metric

    # all-season combined + latest season example
    combined_heatmap(df, metric, fname_prefix)
    latest = int(df["season"].max())
    season_heatmap(df, latest, metric, fname_prefix)

if __name__ == "__main__":
    main()
