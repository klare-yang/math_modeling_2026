#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T14 — Build tidy metrics table for plotting / paper
Tag: route2_v2_ta97

Inputs:
  - data/T08_logV_posterior_stats_route2_v2_ta97.csv
  - data/T10_hard_consistency_by_event_route2_v2_ta97.csv
  - data/T11_soft_consistency_by_event_route2_v2_ta97.csv
  - data/T13_uncertainty_by_contestant_route2_v2_ta97.csv
  - data/T13_uncertainty_by_week_route2_v2_ta97.csv

Output:
  - data/T14_tidy_metrics_route2_v2_ta97.csv

Notes:
  - This script does NOT require netCDF4/arviz. It merges derived posterior summaries already exported.
  - Event-level metrics (T10/T11) are broadcast to contestant-week rows via (season, week, event_id).
"""

import numpy as np
import pandas as pd

TAG = "route2_v2_ta97"

PATH_T08 = f"data/T08_logV_posterior_stats_{TAG}.csv"
PATH_T10 = f"data/T10_hard_consistency_by_event_{TAG}.csv"
PATH_T11 = f"data/T11_soft_consistency_by_event_{TAG}.csv"
PATH_T13C = f"data/T13_uncertainty_by_contestant_{TAG}.csv"
PATH_T13W = f"data/T13_uncertainty_by_week_{TAG}.csv"

OUT_T14 = f"data/T14_tidy_metrics_{TAG}.csv"

def main():
    t08 = pd.read_csv(PATH_T08)
    t10 = pd.read_csv(PATH_T10)
    t11 = pd.read_csv(PATH_T11)
    t13c = pd.read_csv(PATH_T13C)
    t13w = pd.read_csv(PATH_T13W)

    df = t08.copy()

    # Derived widths/CV (convenient for Q2 & plotting)
    df["logV_centered_ci_width"] = df["logV_centered_ci97_5"] - df["logV_centered_ci2_5"]
    df["share_ci_width"] = df["share_ci97_5"] - df["share_ci2_5"]
    df["share_cv"] = df["share_sd"] / df["share_mean"].replace(0, np.nan)

    # Ranks within event (1 = lowest share)
    df["rank_share_asc"] = df.groupby("event_id")["share_mean"].rank(method="average", ascending=True)
    df["rank_share_desc"] = df.groupby("event_id")["share_mean"].rank(method="average", ascending=False)

    # Merge event-level metrics (hard/soft); keep NA for elim_count==0 events
    event_metrics = t10.merge(
        t11,
        on=["event_id", "season", "week", "n_active", "elim_count"],
        how="outer",
        validate="one_to_one",
    )
    df = df.merge(
        event_metrics[[
            "event_id", "season", "week", "elim_count",
            "hard_consistency_prob",
            "p_elim_mean", "p_elim_hdi_low", "p_elim_hdi_high",
        ]],
        on=["event_id", "season", "week"],
        how="left",
        validate="many_to_one",
    )

    # Convenience indicator: (posterior mean) bottom-k membership by share_mean
    df["is_bottom_k_by_share_mean"] = np.where(
        df["elim_count"] > 0,
        df["rank_share_asc"] <= df["elim_count"],
        np.nan
    )

    # Merge contestant-level heterogeneity summaries (T13)
    df = df.merge(
        t13c[["contestant_idx", "logV_sd_mean", "logV_sd_median", "share_cv_median"]].rename(columns={
            "logV_sd_mean": "contestant_logV_sd_mean",
            "logV_sd_median": "contestant_logV_sd_median",
            "share_cv_median": "contestant_share_cv_median",
        }),
        on="contestant_idx",
        how="left",
        validate="many_to_one",
    )

    # Merge week-level heterogeneity summaries (T13)
    df = df.merge(
        t13w[["season", "week", "n_active_mean", "logV_sd_mean", "share_cv_median"]].rename(columns={
            "n_active_mean": "week_n_active_mean",
            "logV_sd_mean": "week_logV_sd_mean",
            "share_cv_median": "week_share_cv_median",
        }),
        on=["season", "week"],
        how="left",
        validate="many_to_one",
    )

    # Column order (tidy)
    cols = [
        "season","week","event_id","n_active","elim_count",
        "contestant_key","contestant_idx","slot","score_z","eliminated_flag",
        "share_mean","share_sd","share_ci2_5","share_ci97_5","share_ci_width","share_cv",
        "logV_centered_mean","logV_centered_sd","logV_centered_ci2_5","logV_centered_ci97_5","logV_centered_ci_width",
        "rank_share_asc","rank_share_desc","is_bottom_k_by_share_mean",
        "hard_consistency_prob","p_elim_mean","p_elim_hdi_low","p_elim_hdi_high",
        "contestant_logV_sd_mean","contestant_logV_sd_median","contestant_share_cv_median",
        "week_n_active_mean","week_logV_sd_mean","week_share_cv_median",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    # Sanity checks
    assert out.shape[0] == t08.shape[0], "Row count mismatch after merges"
    # event-level fields should be constant within each event (allow all-NA)
    nunq = out.groupby("event_id")[["hard_consistency_prob","p_elim_mean"]].nunique(dropna=False)
    assert (nunq.max().max() == 1), "Event-level metrics vary within an event (unexpected)"

    out.to_csv(OUT_T14, index=False)
    print(f"[OK] Saved: {OUT_T14} | rows={out.shape[0]} cols={out.shape[1]}")

if __name__ == "__main__":
    main()
