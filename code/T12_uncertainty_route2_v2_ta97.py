# T12_uncertainty_route2_v2_ta97.py
# Purpose: derive uncertainty metrics (SD & CI width) for each contestant-week (event_id, contestant)
#
# Inputs:
#   data/T08_logV_posterior_stats_route2_v2_ta97.csv
#
# Outputs:
#   data/T12_uncertainty_route2_v2_ta97.csv
#
# Notes:
# - Prefer logV_centered_* and share_* for comparability across events (identifiability issue in raw logV).
# - CI here is the 2.5% and 97.5% interval already exported in T08.

import numpy as np
import pandas as pd
from pathlib import Path

TAG = "route2_v2_ta97"
IN_CSV  = Path("data") / f"T08_logV_posterior_stats_{TAG}.csv"
OUT_CSV = Path("data") / f"T12_uncertainty_{TAG}.csv"

def main():
    df = pd.read_csv(IN_CSV)

    required = [
        "season","week","event_id","contestant_key","contestant_idx","slot","n_active","eliminated_flag",
        "logV_centered_sd","logV_centered_ci2_5","logV_centered_ci97_5",
        "share_sd","share_mean","share_ci2_5","share_ci97_5",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["logV_centered_ci_width"] = df["logV_centered_ci97_5"] - df["logV_centered_ci2_5"]
    df["share_ci_width"] = df["share_ci97_5"] - df["share_ci2_5"]
    df["share_cv"] = df["share_sd"] / df["share_mean"].replace(0, np.nan)

    t12 = df[[
        "season","week","event_id","contestant_key","contestant_idx","slot","n_active","eliminated_flag",
        "logV_centered_sd","logV_centered_ci_width",
        "share_sd","share_ci_width","share_cv"
    ]].copy()

    # Sanity checks
    if not (t12["logV_centered_ci_width"] > 0).all():
        bad = t12.loc[~(t12["logV_centered_ci_width"] > 0), ["event_id","contestant_key","logV_centered_ci_width"]].head(10)
        raise ValueError("Found non-positive logV_centered_ci_width. Examples:\n" + bad.to_string(index=False))

    if not (t12["share_ci_width"] > 0).all():
        bad = t12.loc[~(t12["share_ci_width"] > 0), ["event_id","contestant_key","share_ci_width"]].head(10)
        raise ValueError("Found non-positive share_ci_width. Examples:\n" + bad.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    t12.to_csv(OUT_CSV, index=False)

    # Minimal summary print (optional)
    print("Saved:", OUT_CSV)
    print("Rows:", len(t12), "Events:", t12["event_id"].nunique())
    print("median(logV_centered_sd) =", float(t12["logV_centered_sd"].median()))
    print("median(logV_centered_ci_width) =", float(t12["logV_centered_ci_width"].median()))
    print("median(share_sd) =", float(t12["share_sd"].median()))
    print("median(share_ci_width) =", float(t12["share_ci_width"].median()))

if __name__ == "__main__":
    main()
