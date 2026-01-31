# T13_heteroscedasticity_route2_v2_ta97.py
# Purpose: quantify heteroscedasticity (uncertainty variability) across contestants and weeks.
# Inputs: data/T12_uncertainty_route2_v2_ta97.csv
# Outputs:
#   data/T13_uncertainty_by_contestant_route2_v2_ta97.csv
#   data/T13_uncertainty_by_week_route2_v2_ta97.csv
#   data/T13_heteroscedasticity_summary_route2_v2_ta97.json

import json
import numpy as np
import pandas as pd

TAG = "route2_v2_ta97"
IN_T12 = f"data/T12_uncertainty_{TAG}.csv"

OUT_CONT = f"data/T13_uncertainty_by_contestant_{TAG}.csv"
OUT_WEEK = f"data/T13_uncertainty_by_week_{TAG}.csv"
OUT_SUM  = f"data/T13_heteroscedasticity_summary_{TAG}.json"

def cv(x: pd.Series) -> float:
    x = x.astype(float).to_numpy()
    m = x.mean()
    return float(x.std(ddof=1) / m) if m != 0 else float("nan")

def main():
    t12 = pd.read_csv(IN_T12)

    # -------- (A) aggregate by contestant --------
    by_cont = (
        t12.groupby(["contestant_idx","contestant_key"], as_index=False)
           .agg(
                n_obs=("event_id","count"),
                logV_sd_mean=("logV_centered_sd","mean"),
                logV_sd_median=("logV_centered_sd","median"),
                logV_ciwidth_median=("logV_centered_ci_width","median"),
                share_sd_mean=("share_sd","mean"),
                share_ciwidth_median=("share_ci_width","median"),
                share_cv_median=("share_cv","median"),
           )
    )
    by_cont.to_csv(OUT_CONT, index=False)

    # -------- (B) aggregate by week (season-week) --------
    # first aggregate at event-level (season, week, event_id), then average over events in same week
    by_event = (
        t12.groupby(["season","week","event_id"], as_index=False)
           .agg(
                n_active=("n_active","first"),
                elim_count=("eliminated_flag","sum"),
                logV_sd_mean=("logV_centered_sd","mean"),
                logV_sd_median=("logV_centered_sd","median"),
                logV_ciwidth_median=("logV_centered_ci_width","median"),
                share_sd_mean=("share_sd","mean"),
                share_ciwidth_median=("share_ci_width","median"),
                share_cv_median=("share_cv","median"),
           )
    )

    by_week = (
        by_event.groupby(["season","week"], as_index=False)
                .agg(
                    n_events=("event_id","count"),
                    n_active_mean=("n_active","mean"),
                    logV_sd_mean=("logV_sd_mean","mean"),
                    share_sd_mean=("share_sd_mean","mean"),
                    share_cv_median=("share_cv_median","median"),
                )
    )
    by_week.to_csv(OUT_WEEK, index=False)

    # -------- summary: "certainty is not the same" evidence --------
    summary = {
        "tag": TAG,
        "rows_t12": int(len(t12)),
        "n_events": int(t12["event_id"].nunique()),
        "n_events_with_elim": int((by_event["elim_count"] > 0).sum()),
        "n_contestants": int(t12["contestant_idx"].nunique()),

        # global uncertainty levels (robust medians)
        "global_median_logV_sd": float(t12["logV_centered_sd"].median()),
        "global_median_logV_ci_width": float(t12["logV_centered_ci_width"].median()),
        "global_median_share_sd": float(t12["share_sd"].median()),
        "global_median_share_ci_width": float(t12["share_ci_width"].median()),
        "global_median_share_cv": float(t12["share_cv"].median()),

        # heterogeneity across contestants and events
        "cv_contestant_logV_sd_mean": cv(by_cont["logV_sd_mean"]),
        "cv_event_logV_sd_mean": cv(by_event["logV_sd_mean"]),
        "corr_event_logV_sd_mean_vs_n_active": float(by_event["logV_sd_mean"].corr(by_event["n_active"])),

        # share-space: note absolute sd tends to shrink with more active (simplex constraint),
        # so we additionally report relative uncertainty (share_cv).
        "cv_contestant_share_sd_mean": cv(by_cont["share_sd_mean"]),
        "cv_event_share_sd_mean": cv(by_event["share_sd_mean"]),
        "corr_event_share_sd_mean_vs_n_active": float(by_event["share_sd_mean"].corr(by_event["n_active"])),

        "cv_contestant_share_cv_median": cv(by_cont["share_cv_median"]),
        "cv_event_share_cv_median": cv(by_event["share_cv_median"]),
        "corr_event_share_cv_median_vs_n_active": float(by_event["share_cv_median"].corr(by_event["n_active"])),
    }

    # early vs late within each season (normalize week rank)
    tmp = by_event.copy()
    tmp["week_rank"] = tmp.groupby("season")["week"].rank(method="dense")
    tmp["week_rank_norm"] = tmp["week_rank"] / tmp.groupby("season")["week_rank"].transform("max")
    early = tmp[tmp["week_rank_norm"] <= 0.33]
    late  = tmp[tmp["week_rank_norm"] >= 0.67]
    summary["early_mean_logV_sd_mean"] = float(early["logV_sd_mean"].mean())
    summary["late_mean_logV_sd_mean"] = float(late["logV_sd_mean"].mean())
    summary["early_mean_n_active"] = float(early["n_active"].mean())
    summary["late_mean_n_active"]  = float(late["n_active"].mean())

    # season-level (optional for narrative)
    season_grp = (
        by_event.groupby("season")
               .agg(
                    n_events=("event_id","count"),
                    n_active_mean=("n_active","mean"),
                    logV_sd_mean=("logV_sd_mean","mean"),
                    share_sd_mean=("share_sd_mean","mean"),
               )
    )
    summary["by_season"] = {
        str(k): {col: float(v[col]) for col in season_grp.columns}
        for k, v in season_grp.iterrows()
    }

    with open(OUT_SUM, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(" -", OUT_CONT)
    print(" -", OUT_WEEK)
    print(" -", OUT_SUM)

if __name__ == "__main__":
    main()
