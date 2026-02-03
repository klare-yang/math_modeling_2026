#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M11: Build elimination (hazard) dataset from M01 panel, optionally merging M10 latent fan posterior means.

Outputs:
- data/M11_elim_panel.csv
- data/M11_elim_event_summary.csv
- data/M11_elim_meta.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


DEFAULT_PANEL = Path("data/M01_panel_m3.csv")
DEFAULT_MAP = Path("data/M01_keymap_m3.json")
DEFAULT_FANS_LATENT_OBS = Path("data/M10_fans_latent_obs_posterior.csv")

OUT_PANEL = Path("data/M11_elim_panel.csv")
OUT_EVENT = Path("data/M11_elim_event_summary.csv")
OUT_META = Path("data/M11_elim_meta.json")


def main():
    parser = argparse.ArgumentParser(description="M11: Build elimination dataset for hazard model.")
    parser.add_argument("--panel", type=str, default=str(DEFAULT_PANEL))
    parser.add_argument("--map", type=str, default=str(DEFAULT_MAP))
    parser.add_argument("--fans_latent_obs", type=str, default=str(DEFAULT_FANS_LATENT_OBS),
                        help="Optional obs-level latent fan posterior means (from M10). If missing, fallback to y_f_logit_share_mean.")
    parser.add_argument("--only_elim_events", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep only events with at least one elimination (default True).")
    args = parser.parse_args()

    panel_path = Path(args.panel)
    map_path = Path(args.map)
    fans_obs_path = Path(args.fans_latent_obs)

    for p in [panel_path, map_path]:
        if not p.exists():
            raise FileNotFoundError(f"[M11] Missing input: {p}")

    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_EVENT.parent.mkdir(parents=True, exist_ok=True)
    OUT_META.parent.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    meta = json.load(open(map_path, "r", encoding="utf-8"))

    required = [
        "event_id", "season_id", "week", "week_z",
        "pro_id", "celeb_id", "industry_id", "age_z",
        "eliminated_flag", "y_j_score_z", "y_f_logit_share_mean",
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"[M11] Missing required columns in panel: {missing}")

    use_fan = "y_f_logit_share_mean"
    if fans_obs_path.exists():
        fans_obs = pd.read_csv(fans_obs_path)
        if "y_latent_mean" in fans_obs.columns:
            merge_cols = ["event_id", "contestant_key", "y_latent_mean", "share_latent_mean"]
            key_cols = ["event_id", "contestant_key"]
            panel = panel.merge(fans_obs[merge_cols], on=key_cols, how="left", validate="one_to_one")
            if panel["y_latent_mean"].notna().mean() > 0.95:
                use_fan = "y_latent_mean"
        # else silently ignore

    # event summary
    ev = panel.groupby("event_id").agg(
        season_id=("season_id", "first"),
        week=("week", "first"),
        n_active=("n_active", "first"),
        n_elim=("eliminated_flag", "sum"),
    ).reset_index()
    ev["n_elim"] = ev["n_elim"].astype(int)
    ev.to_csv(OUT_EVENT, index=False)

    # filter eligible events
    panel2 = panel.copy()
    before_rows = len(panel2)
    before_events = panel2["event_id"].nunique()

    if args.only_elim_events:
        keep_events = set(ev.loc[ev["n_elim"] > 0, "event_id"].tolist())
        panel2 = panel2[panel2["event_id"].isin(keep_events)].copy()

    after_rows = len(panel2)
    after_events = panel2["event_id"].nunique()

    # build centered performance covariates for potential model variants
    panel2["score_c"] = panel2["y_j_score_z"] - panel2.groupby("event_id")["y_j_score_z"].transform("mean")
    panel2["fan_c"] = panel2[use_fan] - panel2.groupby("event_id")[use_fan].transform("mean")

    # keep modeling columns
    keep_cols = [
        "event_id", "season_id", "week", "week_z", "n_active",
        "contestant_key", "celeb_id", "pro_id", "industry_id", "age_z",
        "eliminated_flag",
        "y_j_score_z", "score_c",
        "y_f_logit_share_mean",
        "fan_c",
    ]
    if "y_latent_mean" in panel2.columns:
        keep_cols += ["y_latent_mean", "share_latent_mean"]
    panel_out = panel2[keep_cols].copy()

    panel_out.to_csv(OUT_PANEL, index=False)

    meta_out: Dict[str, Any] = {
        "inputs": {"panel": str(panel_path), "map": str(map_path), "fans_latent_obs": str(fans_obs_path)},
        "outputs": {"elim_panel": str(OUT_PANEL), "event_summary": str(OUT_EVENT)},
        "filters": {
            "only_elim_events": bool(args.only_elim_events),
            "rows_before": int(before_rows),
            "rows_after": int(after_rows),
            "events_before": int(before_events),
            "events_after": int(after_events),
        },
        "fan_predictor_used": use_fan,
        "dims": {
            "n_pro": len(meta["id_maps"]["pro"]["names"]),
            "n_industry": len(meta["id_maps"]["industry"]["names"]),
            "n_season": len(meta["id_maps"]["season"]["values"]),
            "n_celeb": len(meta["id_maps"]["celebrity"]["names"]),
        },
        "event_elim_distribution": ev["n_elim"].value_counts().sort_index().to_dict(),
    }

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    # CHECK BLOCK
    print("\n" + "=" * 90)
    print("M11 CHECK BLOCK START")
    print("=" * 90)
    print("[M11] Inputs:")
    print(f"  panel = {panel_path}")
    print(f"  map   = {map_path}")
    print(f"  fans_latent_obs = {fans_obs_path} (exists={fans_obs_path.exists()})")
    print("\n[M11] Outputs:")
    print(f"  elim_panel     = {OUT_PANEL}")
    print(f"  event_summary  = {OUT_EVENT}")
    print(f"  meta_json      = {OUT_META}")
    print("\n[M11] Fan predictor used:")
    print(f"  use_fan = {use_fan}")
    print("\n[M11] Filters:")
    print(json.dumps(meta_out["filters"], indent=2))
    print("\n[M11] Event elim distribution (n_elim -> count of events):")
    print(json.dumps(meta_out["event_elim_distribution"], indent=2))
    print("\nM11 CHECK BLOCK END")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
