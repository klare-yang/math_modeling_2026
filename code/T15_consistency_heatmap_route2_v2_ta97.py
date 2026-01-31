# T15_consistency_heatmap_route2_v2_ta97.py
# Generate event-level consistency heatmaps (season x week).
# Inputs:
#   - data/T10_hard_consistency_by_event_route2_v2_ta97.csv
#   - data/T11_soft_consistency_by_event_route2_v2_ta97.csv
# Outputs:
#   - fig/T15_consistency_heatmap_hard_route2_v2_ta97.png/.pdf
#   - fig/T15_consistency_heatmap_soft_route2_v2_ta97.png/.pdf

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TAG = "route2_v2_ta97"

T10_CSV = f"data/T10_hard_consistency_by_event_{TAG}.csv"
T11_CSV = f"data/T11_soft_consistency_by_event_{TAG}.csv"
OUT_DIR = "fig"

def _heatmap(pivot_df: pd.DataFrame, title: str, out_png: str, out_pdf: str, vmin=0.0, vmax=1.0):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    arr = pivot_df.values.astype(float)
    m_arr = np.ma.masked_invalid(arr)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(m_arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im, shrink=0.8, label=title)

    plt.xticks(np.arange(len(pivot_df.columns)), pivot_df.columns)
    plt.yticks(np.arange(len(pivot_df.index)), pivot_df.index)

    plt.xlabel("week")
    plt.ylabel("season")
    plt.title(title)

    # missing cells (e.g., elim_count==0) show as grey
    plt.gca().set_facecolor("lightgrey")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

def main():
    t10 = pd.read_csv(T10_CSV)
    t11 = pd.read_csv(T11_CSV)

    # Merge to keep a consistent season-week grid
    ev = (
        t10.merge(t11[["event_id", "p_elim_mean"]], on="event_id", how="left")
           .sort_values(["season", "week"])
           .reset_index(drop=True)
    )

    seasons = sorted(ev["season"].unique())
    weeks = sorted(ev["week"].unique())

    hard = (ev.pivot(index="season", columns="week", values="hard_consistency_prob")
              .reindex(index=seasons, columns=weeks))
    soft = (ev.pivot(index="season", columns="week", values="p_elim_mean")
              .reindex(index=seasons, columns=weeks))

    _heatmap(
        hard,
        "T15 hard consistency: Pr(elim in bottom-k | data)",
        os.path.join(OUT_DIR, f"T15_consistency_heatmap_hard_{TAG}.png"),
        os.path.join(OUT_DIR, f"T15_consistency_heatmap_hard_{TAG}.pdf"),
        vmin=0.0, vmax=1.0
    )

    _heatmap(
        soft,
        "T15 soft consistency: E[p_elim(true) | data)",
        os.path.join(OUT_DIR, f"T15_consistency_heatmap_soft_{TAG}.png"),
        os.path.join(OUT_DIR, f"T15_consistency_heatmap_soft_{TAG}.pdf"),
        vmin=0.0, vmax=1.0
    )

    print("Saved heatmaps to:", OUT_DIR)

if __name__ == "__main__":
    main()
