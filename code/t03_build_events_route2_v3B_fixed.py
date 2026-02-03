#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T03 (route2_v3B): Build per-(season, week) event structure with rule metadata and judge arrays.

Single change relative to v2:
- Augment each event with:
    rule_mode (rank/percent), is_judges_save (bool),
    J_total_by_active (aligned with active_ids),
    J_pct_by_active (event-normalized, aligned with active_ids)

Active definition (kept close to v2 for comparability):
- active := (judge_score_total > 0) AND (judge_score_count > 0)

Outputs:
- data/T03_active_elimination_structure_route2_v3B.json
- data/T05_model_ready_events_route2_v3B.csv

Requires:
- code/t01_route2_v3B.py providing rule_for_season()
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from t01_route2_v3B import rule_for_season


def contestant_key(season: int, celebrity_name: str) -> str:
    return f"S{int(season)}:{str(celebrity_name)}"


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def build_events(panel: pd.DataFrame, only_consecutive_weeks: bool = True) -> Dict[str, Any]:
    req = {"celebrity_name", "season", "week", "judge_score_total", "judge_score_count"}
    miss = req - set(panel.columns)
    if miss:
        raise ValueError(f"[T03_v3B] Panel missing required columns: {sorted(miss)}")

    df = panel.copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["judge_score_total"] = df["judge_score_total"].fillna(0.0).astype(float)
    df["judge_score_count"] = df["judge_score_count"].fillna(0).astype(int)

    # active: keep near v2 but guard against weird zero-count rows
    df["active"] = (df["judge_score_total"] > 0) & (df["judge_score_count"] > 0)
    df["contestant_key"] = df.apply(lambda r: contestant_key(r["season"], r["celebrity_name"]), axis=1)

    events: Dict[str, Any] = {}
    max_active = 0
    max_elim = 0
    n_events = 0
    n_events_with_elim = 0
    nonconsecutive_skipped = 0

    # per (season, week, contestant_key) -> judge_total
    # we'll query per event by filtering (season, week)
    for season, gS in df.groupby("season", sort=True):
        weeks = sorted(gS["week"].unique().tolist())
        for i, w in enumerate(weeks):
            n_events += 1
            next_week = weeks[i + 1] if i + 1 < len(weeks) else None
            delta_week = (next_week - w) if next_week is not None else None
            terminal = next_week is None

            gW = gS[gS["week"] == w]
            active_now = sorted(gW.loc[gW["active"], "contestant_key"].unique().tolist())
            max_active = max(max_active, len(active_now))

            eliminated_ids: List[str] = []
            if not terminal:
                gNext = gS[gS["week"] == next_week]
                active_next = set(gNext.loc[gNext["active"], "contestant_key"].unique().tolist())
                active_now_set = set(active_now)

                if (not only_consecutive_weeks) or (delta_week == 1):
                    eliminated_ids = sorted(list(active_now_set - active_next))
                else:
                    nonconsecutive_skipped += 1
                    eliminated_ids = []

            elim_count = len(eliminated_ids)
            if elim_count > 0:
                n_events_with_elim += 1
            max_elim = max(max_elim, elim_count)

            # eliminated_indices aligned to active_now ordering
            idx_map = {cid: j for j, cid in enumerate(active_now)}
            eliminated_indices = [idx_map[cid] for cid in eliminated_ids if cid in idx_map]

            # rule metadata
            rule = rule_for_season(season)
            rule_mode = rule['mode']
            is_judges_save = bool(rule['judges_save'])

            # J arrays aligned with active_now
            # Use judge_score_total for mechanism; percent-normalize within active set
            gW_active = gW.set_index("contestant_key")
            J_total = []
            for cid in active_now:
                if cid in gW_active.index:
                    J_total.append(float(gW_active.loc[cid, "judge_score_total"]))
                else:
                    # should not happen; keep consistent length
                    J_total.append(0.0)

            J_sum = float(np.sum(J_total))
            if J_sum > 0:
                J_pct = [float(x / J_sum) for x in J_total]
            else:
                # pathological (should not for active), but keep safe
                J_pct = [0.0 for _ in J_total]

            event_id = f"{season}-{w}"
            events[event_id] = {
                "event_id": event_id,
                "season": int(season),
                "week": int(w),
                "next_week": int(next_week) if next_week is not None else None,
                "delta_week": int(delta_week) if delta_week is not None else None,
                "terminal": bool(terminal),
                "active_ids": active_now,
                "n_active": int(len(active_now)),
                "eliminated_ids": eliminated_ids,
                "elim_count": int(elim_count),
                "eliminated_indices": eliminated_indices,
                "rule_mode": rule_mode,
                "is_judges_save": is_judges_save,
                "J_total_by_active": J_total,
                "J_pct_by_active": J_pct,
            }

    payload = {
        "dimensions": {
            "n_events": int(len(events)),
            "max_active": int(max_active),
            "max_elim": int(max_elim),
            "n_events_with_elim": int(n_events_with_elim),
            "n_events": int(n_events),
            "nonconsecutive_skipped": int(nonconsecutive_skipped),
        },
        "rules": {
            "note": "season->rule mapping is defined in code/t01_route2_v3B.py (rule_for_season).",
        },
        "events": events,
    }
    return payload


def export_t05(payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for event_id, ev in payload["events"].items():
        rows.append(
            {
                "event_id": ev["event_id"],
                "season": ev["season"],
                "week": ev["week"],
                "next_week": ev["next_week"],
                "delta_week": ev["delta_week"],
                "terminal": ev["terminal"],
                "n_active": ev["n_active"],
                "elim_count": ev["elim_count"],
                "rule_mode": ev["rule_mode"],
                "is_judges_save": ev["is_judges_save"],
                "active_ids": _safe_json(ev["active_ids"]),
                "eliminated_ids": _safe_json(ev["eliminated_ids"]),
                "eliminated_indices": _safe_json(ev["eliminated_indices"]),
                "J_total_by_active": _safe_json(ev["J_total_by_active"]),
                "J_pct_by_active": _safe_json(ev["J_pct_by_active"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_panel", default="data/T02_long_format_panel_route2_v3B.csv")
    ap.add_argument("--out_t03", default="data/T03_active_elimination_structure_route2_v3B.json")
    ap.add_argument("--out_t05", default="data/T05_model_ready_events_route2_v3B.csv")
    ap.add_argument("--allow_nonconsecutive", action="store_true", default=False)
    args = ap.parse_args()

    panel_path = Path(args.input_panel)
    if not panel_path.exists():
        raise FileNotFoundError(f"[T03_v3B] input_panel not found: {panel_path}")

    panel = pd.read_csv(panel_path)
    print(f"[T03_v3B] panel rows: {len(panel)}")

    payload = build_events(panel, only_consecutive_weeks=(not args.allow_nonconsecutive))

    out_t03 = Path(args.out_t03)
    out_t03.parent.mkdir(parents=True, exist_ok=True)
    out_t03.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[T03_v3B] Saved: {out_t03}")

    out_t05 = Path(args.out_t05)
    out_t05.parent.mkdir(parents=True, exist_ok=True)
    t05 = export_t05(payload)
    t05.to_csv(out_t05, index=False)
    print(f"[T03_v3B] Saved: {out_t05} (rows={len(t05)})")

    dims = payload["dimensions"]
    print("[T03_v3B] dimensions:", json.dumps(dims, ensure_ascii=False))
    print("[T03_v3B] Interpretation hints:")
    print("  - rule_mode / is_judges_save are per-season metadata for mechanism-aligned modeling.")
    print("  - J_pct_by_active is event-normalized judge share; percent-rule uses it directly.")
    print("  - eliminated_indices are indices into active_ids (ordering defines alignment).")


if __name__ == "__main__":
    main()
