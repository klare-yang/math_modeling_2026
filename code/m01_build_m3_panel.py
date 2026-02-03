# code/m01_build_m3_panel.py
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_T08 = Path("data/T08_logV_posterior_stats_route2_v2_ta97.csv")
DEFAULT_RAW = Path("data/2026_MCM_Problem_C_Data.csv")
DEFAULT_OUT_PANEL = Path("data/M01_panel_m3.csv")
DEFAULT_OUT_MAP = Path("data/M01_keymap_m3.json")


def parse_contestant_key(contestant_key: str) -> Tuple[int, str]:
    """
    contestant_key format: "S{season}:{celebrity_name}"
    Example: "S1:Evander Holyfield"
    """
    m = re.match(r"^S(\d+):(.*)$", str(contestant_key))
    if not m:
        return -1, str(contestant_key)
    return int(m.group(1)), m.group(2).strip()


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def build_id_map(values) -> Dict:
    """Stable id mapping: sort unique values as strings."""
    uniq = sorted(pd.Series(values).dropna().astype(str).unique().tolist())
    name_to_id = {name: i for i, name in enumerate(uniq)}
    return {"names": uniq, "name_to_id": name_to_id}


def main():
    parser = argparse.ArgumentParser(description="M01: Build Model 3 panel (Bayesian LMM) from T08 + raw wide table.")
    parser.add_argument("--t08", type=str, default=str(DEFAULT_T08), help="Path to T08 posterior stats csv.")
    parser.add_argument("--raw", type=str, default=str(DEFAULT_RAW), help="Path to raw wide DWTS csv.")
    parser.add_argument("--out_panel", type=str, default=str(DEFAULT_OUT_PANEL), help="Output panel csv path.")
    parser.add_argument("--out_map", type=str, default=str(DEFAULT_OUT_MAP), help="Output mapping json path.")
    parser.add_argument("--industry_min_count", type=int, default=5, help="Industries with < min_count will be grouped into 'Other'.")
    parser.add_argument("--clip_eps", type=float, default=1e-6, help="Clip epsilon for share_mean before logit transform.")
    parser.add_argument("--se_logit_cap", type=float, default=10.0, help="Cap for y_f_logit_share_se to avoid infinite weights.")
    args = parser.parse_args()

    t08_path = Path(args.t08)
    raw_path = Path(args.raw)
    out_panel = Path(args.out_panel)
    out_map = Path(args.out_map)

    if not t08_path.exists():
        raise FileNotFoundError(f"T08 not found: {t08_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    out_panel.parent.mkdir(parents=True, exist_ok=True)
    out_map.parent.mkdir(parents=True, exist_ok=True)

    print(f"[M01] Loading T08: {t08_path}")
    t08 = pd.read_csv(t08_path)

    required_t08_cols = {
        "season", "week", "contestant_key", "score_z",
        "share_mean", "share_sd",
        "logV_centered_mean", "logV_centered_sd",
    }
    missing = sorted(list(required_t08_cols - set(t08.columns)))
    if missing:
        raise ValueError(f"[M01] T08 missing required columns: {missing}")

    print(f"[M01] Loading RAW: {raw_path}")
    raw = pd.read_csv(raw_path)

    required_raw_cols = {
        "season", "celebrity_name", "ballroom_partner",
        "celebrity_industry", "celebrity_age_during_season",
    }
    missing_raw = sorted(list(required_raw_cols - set(raw.columns)))
    if missing_raw:
        raise ValueError(f"[M01] RAW missing required columns: {missing_raw}")

    # Build a season-celebrity dimension table from RAW (unique contestants per season)
    dim = (
        raw[list(required_raw_cols)]
        .drop_duplicates(subset=["season", "celebrity_name"])
        .copy()
    )
    print(f"[M01] RAW dim rows (season-celebrity unique): {dim.shape[0]}")

    # Group rare industries into "Other" based on RAW contestant-level counts (not week-level)
    ind_counts = dim["celebrity_industry"].value_counts(dropna=False)
    rare_inds = set(ind_counts[ind_counts < args.industry_min_count].index.astype(str).tolist())

    dim["industry_group"] = dim["celebrity_industry"].astype(str)
    dim.loc[dim["industry_group"].isin(rare_inds), "industry_group"] = "Other"

    # Parse contestant_key into parsed season + celebrity_name (cross-check)
    parsed = t08["contestant_key"].apply(parse_contestant_key)
    t08["season_from_key"] = parsed.apply(lambda x: x[0])
    t08["celebrity_name_from_key"] = parsed.apply(lambda x: x[1])

    # Cross-check season consistency
    bad_season = (t08["season_from_key"] != t08["season"]).sum()
    if bad_season > 0:
        print(f"[M01][WARN] {bad_season} rows where season_from_key != season column. Using season column as truth.")

    # Merge: (season, celebrity_name) from key with RAW dim
    panel = t08.merge(
        dim,
        left_on=["season", "celebrity_name_from_key"],
        right_on=["season", "celebrity_name"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_raw")
    )

    n_total = panel.shape[0]
    n_unmatched = panel["ballroom_partner"].isna().sum()
    if n_unmatched > 0:
        print(f"[M01][WARN] Unmatched rows after merge: {n_unmatched}/{n_total}")
        # Show a few examples for debugging
        ex = panel.loc[panel["ballroom_partner"].isna(), ["season", "contestant_key", "celebrity_name_from_key"]].head(10)
        print("[M01][WARN] Unmatched examples:\n", ex.to_string(index=False))

    # Drop unmatched rows (cannot run Model 3 without pro/industry/age)
    before_drop = panel.shape[0]
    panel = panel.dropna(subset=["ballroom_partner", "industry_group", "celebrity_age_during_season"]).copy()
    dropped = before_drop - panel.shape[0]
    print(f"[M01] Dropped rows with missing covariates: {dropped}")

    # Build fan outcome candidates
    eps = float(args.clip_eps)
    p = panel["share_mean"].astype(float).to_numpy()
    p = np.clip(p, eps, 1.0 - eps)
    panel["share_mean_clipped"] = p

    panel["y_f_logit_share_mean"] = logit(p)

    # Delta method for se in logit space: se_y ≈ se_p / (p*(1-p))
    se_p = panel["share_sd"].astype(float).fillna(0.0).to_numpy()
    denom = p * (1.0 - p)
    se_y = np.zeros_like(se_p)
    # avoid division by zero (after clipping denom should be >0, but keep safe)
    denom_safe = np.where(denom <= 0.0, np.nan, denom)
    se_y = se_p / denom_safe
    # cap extreme se
    se_cap = float(args.se_logit_cap)
    se_y = np.where(np.isnan(se_y), se_cap, se_y)
    se_y = np.clip(se_y, 0.0, se_cap)
    panel["y_f_logit_share_se"] = se_y

    # Judges outcome is already standardized in T08: score_z
    panel["y_j_score_z"] = panel["score_z"].astype(float)

    # Standardize age and week on the PANEL (week-level observations)
    panel["age"] = panel["celebrity_age_during_season"].astype(float)
    age_mean = float(panel["age"].mean())
    age_sd = float(panel["age"].std(ddof=0)) if float(panel["age"].std(ddof=0)) > 0 else 1.0
    panel["age_z"] = (panel["age"] - age_mean) / age_sd

    wk = panel["week"].astype(float)
    wk_mean = float(wk.mean())
    wk_sd = float(wk.std(ddof=0)) if float(wk.std(ddof=0)) > 0 else 1.0
    panel["week_z"] = (wk - wk_mean) / wk_sd

    # Build stable ID maps (string-based)
    # celeb_id: unique across season? For LMM, celebrity should be unique within season (same name may appear in different seasons? Rare but possible).
    # We define celeb_key = "S{season}:{celebrity_name}" consistent with T08 contestant_key.
    panel["celeb_key"] = panel["contestant_key"].astype(str)

    pro_map = build_id_map(panel["ballroom_partner"].astype(str))
    ind_map = build_id_map(panel["industry_group"].astype(str))
    celeb_map = build_id_map(panel["celeb_key"].astype(str))
    season_vals = sorted(panel["season"].dropna().astype(int).unique().tolist())
    season_to_id = {int(s): i for i, s in enumerate(season_vals)}

    panel["pro_id"] = panel["ballroom_partner"].astype(str).map(pro_map["name_to_id"]).astype(int)
    panel["industry_id"] = panel["industry_group"].astype(str).map(ind_map["name_to_id"]).astype(int)
    panel["celeb_id"] = panel["celeb_key"].astype(str).map(celeb_map["name_to_id"]).astype(int)
    panel["season_id"] = panel["season"].astype(int).map(season_to_id).astype(int)

    # Reorder columns for readability / downstream use
    cols_front = [
        "season", "season_id", "week", "week_z", "event_id", "slot", "n_active",
        "contestant_key", "celeb_key", "celeb_id",
        "ballroom_partner", "pro_id",
        "industry_group", "industry_id",
        "age", "age_z",
        "y_j_score_z",
        "share_mean", "share_sd", "share_ci2_5", "share_ci97_5",
        "share_mean_clipped",
        "y_f_logit_share_mean", "y_f_logit_share_se",
        "logV_centered_mean", "logV_centered_sd", "logV_centered_ci2_5", "logV_centered_ci97_5",
        "eliminated_flag",
    ]
    cols_front = [c for c in cols_front if c in panel.columns]
    cols_rest = [c for c in panel.columns if c not in cols_front]
    panel = panel[cols_front + cols_rest].copy()

    # Save panel
    panel.to_csv(out_panel, index=False)
    print(f"[M01] Saved panel: {out_panel}  shape={panel.shape}")

    # Save mapping / config
    meta = {
        "inputs": {
            "t08": str(t08_path),
            "raw": str(raw_path),
        },
        "outputs": {
            "panel": str(out_panel),
            "map": str(out_map),
        },
        "config": {
            "industry_min_count": int(args.industry_min_count),
            "clip_eps": float(args.clip_eps),
            "se_logit_cap": float(args.se_logit_cap),
        },
        "panel_summary": {
            "n_rows": int(panel.shape[0]),
            "n_seasons": int(panel["season"].nunique()),
            "n_unique_celeb": int(panel["celeb_key"].nunique()),
            "n_unique_pro": int(panel["ballroom_partner"].nunique()),
            "n_unique_industry": int(panel["industry_group"].nunique()),
            "dropped_missing_covariates": int(dropped),
            "unmatched_after_merge": int(n_unmatched),
        },
        "standardization": {
            "age_mean": age_mean,
            "age_sd": age_sd,
            "week_mean": wk_mean,
            "week_sd": wk_sd,
        },
        "id_maps": {
            "season": {"values": season_vals, "value_to_id": season_to_id},
            "pro": pro_map,
            "industry": ind_map,
            "celebrity": celeb_map,
        },
    }

    with open(out_map, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[M01] Saved id map + meta: {out_map}")

    # Quick QC prints
    print("\n[M01] QC snapshot:")
    print(panel[[
        "season", "week", "contestant_key",
        "ballroom_partner", "industry_group", "age",
        "y_j_score_z", "share_mean", "y_f_logit_share_mean", "y_f_logit_share_se"
    ]].head(10).to_string(index=False))

    print("\n[M01] Distribution checks:")
    print("  y_j_score_z: mean/std =", float(panel["y_j_score_z"].mean()), float(panel["y_j_score_z"].std()))
    print("  y_f_logit_share_mean: mean/std =", float(panel["y_f_logit_share_mean"].mean()), float(panel["y_f_logit_share_mean"].std()))
    print("  y_f_logit_share_se: min/median/max =", float(panel["y_f_logit_share_se"].min()),
          float(panel["y_f_logit_share_se"].median()), float(panel["y_f_logit_share_se"].max()))
    print("\n[M01] Done.")


if __name__ == "__main__":
    main()
