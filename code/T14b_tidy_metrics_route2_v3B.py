#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T14b_tidy_metrics_route2_v3B.py

Task point 16: Build a tidy, analysis-ready metrics table by merging:
  - T10b hard consistency (event-level)
  - T11b soft consistency (event-level; from t09 PPC analytic)
  - T08b pi posterior stats (slot-level -> aggregated to event-level uncertainty)
  - T05 event metadata (season/week/rule_mode/elim_count/use_two_stage)

Outputs:
  data/T14b_tidy_metrics_route2_v3B.csv
  data/T14b_tidy_metrics_route2_v3B.json

Notes:
- Event-level uncertainty is computed as:
    pi_hdi_width_mean = mean over active slots of pi_hdi_width
    pi_hdi_width_median, pi_sd_mean also exported when available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t05", default="data/T05_model_ready_events_route2_v3B.csv")
    ap.add_argument("--t08b_csv", default="data/T08b_votes_posterior_stats_route2_v3B.csv")
    ap.add_argument("--t10b_csv", default="data/T10b_hard_consistency_S_route2_v3B.csv")
    ap.add_argument("--t11b_csv", default="data/T11b_soft_consistency_2stage_route2_v3B.csv")
    ap.add_argument("--out_csv", default="data/T14b_tidy_metrics_route2_v3B.csv")
    ap.add_argument("--out_json", default="data/T14b_tidy_metrics_route2_v3B.json")
    args = ap.parse_args()

    t05 = Path(args.t05)
    t08b = Path(args.t08b_csv)
    t10b = Path(args.t10b_csv)
    t11b = Path(args.t11b_csv)
    for p in [t05, t08b, t10b, t11b]:
        if not p.exists():
            raise FileNotFoundError(f"[T14b] missing input: {p}")

    df05 = pd.read_csv(t05)
    if "event_id" not in df05.columns:
        raise ValueError("[T14b] T05 must include event_id.")
    # keep selected meta columns if they exist
    meta_cols = ["event_id"]
    for c in ["season", "week", "rule_mode", "elim_count"]:
        if c in df05.columns:
            meta_cols.append(c)
    meta = df05[meta_cols].drop_duplicates("event_id")

    # T08b slot-level -> event-level uncertainty
    df08 = pd.read_csv(t08b)
    if "event_id" not in df08.columns:
        raise ValueError("[T14b] T08b csv must include event_id.")
    agg = {
        "pi_hdi_width": ["mean", "median"],
        "pi_sd": ["mean", "median"],
        "pi_mean": ["max", "min"],
    }
    cols_exist = [c for c in agg.keys() if c in df08.columns]
    if not cols_exist:
        raise ValueError("[T14b] T08b is missing expected columns (pi_hdi_width/pi_sd/pi_mean).")
    agg2 = {c: agg[c] for c in cols_exist}
    df08g = df08.groupby("event_id", as_index=False).agg(agg2)
    # flatten
    df08g.columns = ["event_id"] + [f"{a}_{b}" for a, b in df08g.columns.tolist()[1:]]
    # normalize names
    ren = {}
    if "pi_hdi_width_mean" in df08g.columns: ren["pi_hdi_width_mean"] = "pi_hdi_width_mean"
    if "pi_hdi_width_median" in df08g.columns: ren["pi_hdi_width_median"] = "pi_hdi_width_median"
    if "pi_sd_mean" in df08g.columns: ren["pi_sd_mean"] = "pi_sd_mean"
    if "pi_sd_median" in df08g.columns: ren["pi_sd_median"] = "pi_sd_median"
    if "pi_mean_max" in df08g.columns: ren["pi_mean_max"] = "pi_mean_max"
    if "pi_mean_min" in df08g.columns: ren["pi_mean_min"] = "pi_mean_min"
    df08g = df08g.rename(columns=ren)

    # T10b hard
    df10 = pd.read_csv(t10b)
    if "event_id" not in df10.columns:
        raise ValueError("[T14b] T10b csv must include event_id.")
    keep10 = ["event_id"]
    for c in ["hard_hit_set", "hard_hit_all_in_bottomk", "hard_hit_k1_exact",
              "hard_stage1_hit_bottom2", "hard_stage2_hit_pred",
              "use_two_stage", "elim_count", "rule_mode", "season", "week"]:
        if c in df10.columns:
            keep10.append(c)
    df10 = df10[keep10].copy()

    # T11b soft
    df11 = pd.read_csv(t11b)
    if "event_id" not in df11.columns:
        raise ValueError("[T14b] T11b csv must include event_id.")
    keep11 = ["event_id"]
    for c in ["ppc_p_mean", "ppc_p_q025", "ppc_p_q975", "ppc_logp_mean", "ppc_logp_q025", "ppc_logp_q975",
              "use_two_stage", "elim_count", "rule_mode", "season", "week"]:
        if c in df11.columns:
            keep11.append(c)
    df11 = df11[keep11].copy()

    # Merge priority: meta -> hard -> soft -> uncertainty
    out = meta.merge(df10, on="event_id", how="left", suffixes=("", "_t10b"))
    out = out.merge(df11, on="event_id", how="left", suffixes=("", "_t11b"))
    out = out.merge(df08g, on="event_id", how="left")

    # Coalesce duplicated meta fields
    for c in ["season", "week", "rule_mode", "elim_count", "use_two_stage"]:
        c10 = f"{c}_t10b"
        c11 = f"{c}_t11b"
        if c in out.columns:
            base = out[c]
        else:
            base = pd.Series([np.nan]*len(out))
            out[c] = base
        if c10 in out.columns:
            out[c] = out[c].fillna(out[c10])
            out.drop(columns=[c10], inplace=True)
        if c11 in out.columns:
            out[c] = out[c].fillna(out[c11])
            out.drop(columns=[c11], inplace=True)

    # Sort for readability
    for c in ["season", "week"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    sort_cols = [c for c in ["season", "week", "event_id"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    # Summary JSON
    summary: Dict[str, Any] = {
        "inputs": {
            "t05": str(t05),
            "t08b_csv": str(t08b),
            "t10b_csv": str(t10b),
            "t11b_csv": str(t11b),
        },
        "n_events": int(len(out)),
        "missing_rates": {},
    }
    for c in ["ppc_p_mean", "hard_hit_set", "pi_hdi_width_mean"]:
        if c in out.columns:
            summary["missing_rates"][c] = float(out[c].isna().mean())

    # Quick stratified checks
    if "rule_mode" in out.columns and "ppc_p_mean" in out.columns:
        summary["ppc_p_mean_by_rule_mode"] = out.groupby("rule_mode", dropna=False)["ppc_p_mean"].mean().to_dict()
    if "rule_mode" in out.columns and "hard_hit_set" in out.columns:
        summary["hard_hit_set_by_rule_mode"] = out.groupby("rule_mode", dropna=False)["hard_hit_set"].mean().to_dict()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 90)
    print("[T14b] Saved:", out_csv)
    print("[T14b] Summary:", out_json)
    print("[T14b] Missing rates:", summary["missing_rates"])
    print("=" * 90)


if __name__ == "__main__":
    main()
