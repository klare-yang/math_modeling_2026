#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T11b_soft_consistency_2stage_route2_v3B.py

Task point 15: Soft consistency (likelihood-consistent probability) for observed eliminations.
This script is a lightweight aggregator over the PPC analytic output from t09 (formerly T19).

Inputs:
  --ppc_csv   data/T19_ppc_replay_summary_route2_v3B.csv  (or T09 alias)
  --t05_path  data/T05_model_ready_events_route2_v3B.csv  (for rule_mode merge)
  --mask_path data/T04_judges_save_event_mask_route2_v3B.csv (optional, for sanity/consistency)

Outputs:
  data/T11b_soft_consistency_2stage_route2_v3B.csv
  data/T11b_soft_consistency_2stage_route2_v3B.json

Event-level fields kept:
  ppc_p_mean, ppc_p_q025/q975, ppc_logp_mean, ppc_logp_q025/q975
Plus:
  rule_mode (rank/percent) merged from T05 (if present)

Summary JSON includes:
  overall means/medians and stratified summaries by rule_mode / use_two_stage / elim_count
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
    ap.add_argument("--ppc_csv", default="data/T19_ppc_replay_summary_route2_v3B.csv")
    ap.add_argument("--t05_path", default="data/T05_model_ready_events_route2_v3B.csv")
    ap.add_argument("--mask_path", default="data/T04_judges_save_event_mask_route2_v3B.csv")
    ap.add_argument("--out_csv", default="data/T11b_soft_consistency_2stage_route2_v3B.csv")
    ap.add_argument("--out_json", default="data/T11b_soft_consistency_2stage_route2_v3B.json")
    args = ap.parse_args()

    ppc_path = Path(args.ppc_csv)
    t05_path = Path(args.t05_path)
    mask_path = Path(args.mask_path)

    if not ppc_path.exists():
        raise FileNotFoundError(f"[T11b] ppc_csv not found: {ppc_path}")
    if not t05_path.exists():
        raise FileNotFoundError(f"[T11b] T05 not found: {t05_path}")
    if not mask_path.exists():
        # mask is optional for this aggregator; warn but continue
        mask_path = None

    ppc = pd.read_csv(ppc_path)
    if "event_id" not in ppc.columns:
        raise ValueError("[T11b] ppc_csv must include event_id.")

    t05 = pd.read_csv(t05_path)
    keep_cols = ["event_id"]
    for c in ["rule_mode", "season", "week"]:
        if c in t05.columns and c not in keep_cols:
            keep_cols.append(c)
    meta = t05[keep_cols].drop_duplicates("event_id")

    out = ppc.merge(meta, on="event_id", how="left", suffixes=("", "_t05"))

    # Select/rename final columns
    cols = ["event_id"]
    for c in ["season", "week", "elim_count", "use_two_stage", "rule_mode",
              "ppc_p_mean", "ppc_p_q025", "ppc_p_q975",
              "ppc_logp_mean", "ppc_logp_q025", "ppc_logp_q975"]:
        if c in out.columns:
            cols.append(c)
    out = out[cols].copy()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    # Summaries
    def summarize(df: pd.DataFrame) -> Dict[str, Any]:
        p = df["ppc_p_mean"].astype(float)
        lp = df["ppc_logp_mean"].astype(float)
        return {
            "n": int(len(df)),
            "p_mean": float(p.mean()),
            "p_median": float(p.median()),
            "p_q25": float(p.quantile(0.25)),
            "p_q75": float(p.quantile(0.75)),
            "logp_mean": float(lp.mean()),
            "logp_median": float(lp.median()),
            "share_p_gt_0p5": float((p > 0.5).mean()),
            "share_p_gt_0p2": float((p > 0.2).mean()),
            "share_p_gt_0p1": float((p > 0.1).mean()),
        }

    summary: Dict[str, Any] = {
        "ppc_csv": str(ppc_path),
        "t05": str(t05_path),
        "n_events": int(len(out)),
        "overall": summarize(out),
        "by_rule_mode": {},
        "by_use_two_stage": {},
        "by_elim_count": {},
    }

    if "rule_mode" in out.columns:
        for key, df in out.groupby("rule_mode", dropna=False):
            summary["by_rule_mode"][str(key)] = summarize(df)

    if "use_two_stage" in out.columns:
        for key, df in out.groupby("use_two_stage", dropna=False):
            summary["by_use_two_stage"][str(key)] = summarize(df)

    if "elim_count" in out.columns:
        for key, df in out.groupby("elim_count", dropna=False):
            summary["by_elim_count"][str(key)] = summarize(df)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 90)
    print("[T11b] Saved:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print("[T11b] Overall:", summary["overall"])
    print("=" * 90)


if __name__ == "__main__":
    main()
