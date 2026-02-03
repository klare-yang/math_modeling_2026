#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T02 (route2_v3B): Raw wide table -> long panel aggregated to (season, week, celebrity).

What this script does (single change):
- Produce a stable, versioned per-(celebrity, season, week) panel with:
    judge_score_total, judge_score_count, judge_score_avg, raw_zero_count
  plus key meta columns (industry/age/etc.) copied via 'first'.

Key robustness:
- Handles "N/A" / non-numeric scores by coercing to NaN, then dropping.
- Only columns matching ^week\\d+_judge\\d+_score$ are treated as scores.

Outputs:
- data/T02_long_format_panel_route2_v3B.csv
- data/T02_qc_report_route2_v3B.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


RAW_PATH = Path("data/2026_MCM_Problem_C_Data.csv")
OUT_PANEL = Path("data/T02_long_format_panel_route2_v3B.csv")
OUT_QC = Path("data/T02_qc_report_route2_v3B.json")


SCORE_COL_RE = re.compile(r"^week(\d+)_judge(\d+)_score$")


def _detect_score_cols(columns: List[str]) -> List[str]:
    return [c for c in columns if SCORE_COL_RE.match(c) is not None]


def _detect_meta_cols(columns: List[str]) -> List[str]:
    # meta cols are all columns except score columns
    score_cols = set(_detect_score_cols(columns))
    return [c for c in columns if c not in score_cols]


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"[T02_v3B] raw file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    print(f"[T02_v3B] Raw shape: {df.shape}")

    score_cols = _detect_score_cols(df.columns.tolist())
    meta_cols = _detect_meta_cols(df.columns.tolist())

    if len(score_cols) == 0:
        raise ValueError("[T02_v3B] No score columns detected. Expected week{n}_judge{m}_score.")

    req_meta = {"celebrity_name", "season"}
    miss = req_meta - set(meta_cols)
    if miss:
        raise ValueError(f"[T02_v3B] Missing required meta columns: {sorted(miss)}")

    print(f"[T02_v3B] Detected meta cols: {len(meta_cols)}")
    print(f"[T02_v3B] Detected score cols: {len(score_cols)}")

    # Melt to long
    df_long = df[meta_cols + score_cols].melt(
        id_vars=meta_cols,
        value_vars=score_cols,
        var_name="week_judge",
        value_name="score_raw",
    )

    # Parse week, judge_id
    extracted = df_long["week_judge"].str.extract(r"^week(\d+)_judge(\d+)_score$")
    df_long["week"] = extracted[0].astype("Int64")
    df_long["judge_id"] = extracted[1].astype("Int64")

    # Coerce score to numeric (handles strings like "N/A")
    df_long["score"] = pd.to_numeric(df_long["score_raw"], errors="coerce")

    n_total = len(df_long)
    n_nan_score = int(df_long["score"].isna().sum())
    n_nonnull = int(df_long["score"].notna().sum())

    # Drop missing scores (true missing or coerced non-numeric)
    df_long = df_long.dropna(subset=["score", "week", "judge_id"]).copy()

    # basic range checks (scores should be 0..10 per prompt, but we won't hard-fail)
    score_min = float(df_long["score"].min()) if len(df_long) else float("nan")
    score_max = float(df_long["score"].max()) if len(df_long) else float("nan")

    if score_min < 0 or score_max > 100:
        print(f"[T02_v3B][WARN] Unusual score range: min={score_min}, max={score_max}")

    # Aggregate to panel: (celebrity_name, season, week)
    # Keep meta columns via 'first' (they are season-level descriptors in the prompt data)
    group_keys = ["celebrity_name", "season", "week"]

    # ensure types
    df_long["season"] = df_long["season"].astype(int)
    df_long["week"] = df_long["week"].astype(int)

    meta_keep = [c for c in meta_cols if c not in {"week"}]  # week parsed from score column
    # 'season' and 'celebrity_name' already in keys; keep the rest
    meta_keep = [c for c in meta_keep if c not in {"celebrity_name", "season"}]

    agg_spec: Dict[str, tuple] = {
        "judge_score_total": ("score", "sum"),
        "judge_score_count": ("score", "count"),
        "raw_zero_count": ("score", lambda x: int((x == 0).sum())),
    }
    for c in meta_keep:
        agg_spec[c] = (c, "first")

    panel = (
        df_long.groupby(group_keys, as_index=False)
        .agg(**agg_spec)
        .copy()
    )

    # avg score
    panel["judge_score_avg"] = panel["judge_score_total"] / panel["judge_score_count"].replace(0, np.nan)

    # Sort and save
    panel["season"] = panel["season"].astype(int)
    panel["week"] = panel["week"].astype(int)
    panel = panel.sort_values(["season", "week", "celebrity_name"]).reset_index(drop=True)

    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PANEL, index=False)
    print(f"[T02_v3B] Saved panel: {OUT_PANEL} (rows={len(panel)})")

    # QC report
    qc = {
        "raw": {
            "path": str(RAW_PATH),
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "n_score_cols": int(len(score_cols)),
            "n_meta_cols": int(len(meta_cols)),
        },
        "long": {
            "n_rows_total": int(n_total),
            "n_score_nonnull": int(n_nonnull),
            "n_score_nan_or_non_numeric": int(n_nan_score),
            "score_min": score_min,
            "score_max": score_max,
        },
        "panel": {
            "path": str(OUT_PANEL),
            "n_rows": int(len(panel)),
            "n_seasons": int(panel["season"].nunique()),
            "season_min": int(panel["season"].min()) if len(panel) else None,
            "season_max": int(panel["season"].max()) if len(panel) else None,
            "week_max_global": int(panel["week"].max()) if len(panel) else None,
            "pct_rows_total_eq_0": float((panel["judge_score_total"] == 0).mean()) if len(panel) else None,
            "pct_rows_count_eq_0": float((panel["judge_score_count"] == 0).mean()) if len(panel) else None,
        },
    }
    OUT_QC.parent.mkdir(parents=True, exist_ok=True)
    OUT_QC.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[T02_v3B] Saved QC report: {OUT_QC}")

    # Quick interpretive prints
    print("[T02_v3B] Interpretation hints:")
    print("  - judge_score_total==0 rows are typically placeholders after elimination (depends on dataset encoding).")
    print("  - High n_score_nan_or_non_numeric often indicates 'N/A' strings in raw scores; coerced and dropped.")


if __name__ == "__main__":
    main()
