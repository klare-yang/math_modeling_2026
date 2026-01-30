#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T03 (Route2 v2): Build per-(season,week) active set and eliminated set from the long panel table.

Key fix vs v1:
  active := (judge_score_total > 0)   # treat all-zero placeholder rows as INACTIVE

Outputs:
  - data/T03_active_elimination_structure_route2_v2.json
  - data/T05_model_ready_events_route2_v2.csv

Schema contract (frozen):
  contestant_key := f"S{season}:{celebrity_name}"
  event_id := f"{season}-{week}"
  event fields:
    active_ids, eliminated_ids, eliminated_indices, n_active, elim_count, next_week, delta_week, terminal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def contestant_key(season: int, celebrity_name: str) -> str:
    return f"S{int(season)}:{str(celebrity_name)}"


def build_events(panel: pd.DataFrame, only_consecutive_weeks: bool = True) -> Dict[str, Any]:
    req = {"celebrity_name", "season", "week", "judge_score_total"}
    miss = req - set(panel.columns)
    if miss:
        raise ValueError(f"[T03] Panel missing required columns: {sorted(miss)}")

    df = panel.copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    # ✅ FIX: active is score_total > 0 (placeholder rows are 0)
    df["active"] = df["judge_score_total"].fillna(0) > 0
    df["contestant_key"] = df.apply(lambda r: contestant_key(r["season"], r["celebrity_name"]), axis=1)

    events: Dict[str, Any] = {}
    elim_counts: List[int] = []
    max_active = 0
    max_elim = 0
    n_events = 0
    n_events_with_elim = 0

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
                    eliminated_ids = []

            eliminated_indices = [active_now.index(k) for k in eliminated_ids]
            elim_count = len(eliminated_ids)

            if elim_count > 0:
                n_events_with_elim += 1

            max_elim = max(max_elim, elim_count)
            elim_counts.append(elim_count)

            event_id = f"{season}-{w}"
            events[event_id] = {
                "season": int(season),
                "week": int(w),
                "next_week": int(next_week) if next_week is not None else None,
                "delta_week": int(delta_week) if delta_week is not None else None,
                "active_ids": active_now,
                "eliminated_ids": eliminated_ids,
                "eliminated_indices": eliminated_indices,
                "n_active": len(active_now),
                "elim_count": elim_count,
                "terminal": bool(terminal),
            }

    payload = {
        "schema_version": "route2.v2",
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "config": {
            "only_consecutive_weeks": bool(only_consecutive_weeks),
            "active_definition": "judge_score_total > 0",
            "contestant_key": "S{season}:{celebrity_name}",
            "event_id": "season-week",
        },
        "stats": {
            "n_events": int(n_events),
            "n_events_with_elim": int(n_events_with_elim),
            "max_active": int(max_active),
            "max_elim": int(max_elim),
            "elim_count_hist": dict(pd.Series(elim_counts).value_counts().sort_index().to_dict()),
        },
        "events": events,
    }
    return payload


def export_t05(payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for event_id, ev in payload["events"].items():
        rows.append(
            {
                "season": ev["season"],
                "week": ev["week"],
                "next_week": ev["next_week"],
                "delta_week": ev["delta_week"],
                "n_active": ev["n_active"],
                "elim_count": ev["elim_count"],
                "active_ids": json.dumps(ev["active_ids"], ensure_ascii=False),
                "eliminated_ids": json.dumps(ev["eliminated_ids"], ensure_ascii=False),
                "eliminated_indices": json.dumps(ev["eliminated_indices"], ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_panel", default="data/T02_long_format_panel.csv")
    ap.add_argument("--out_t03", default="data/T03_active_elimination_structure_route2_v2.json")
    ap.add_argument("--out_t05", default="data/T05_model_ready_events_route2_v2.csv")
    ap.add_argument("--allow_nonconsecutive", action="store_true", default=False)
    args = ap.parse_args()

    panel_path = Path(args.input_panel)
    if not panel_path.exists():
        raise FileNotFoundError(f"[T03] input_panel not found: {panel_path}")

    panel = pd.read_csv(panel_path)
    payload = build_events(panel, only_consecutive_weeks=(not args.allow_nonconsecutive))

    out_t03 = Path(args.out_t03)
    out_t03.parent.mkdir(parents=True, exist_ok=True)
    out_t03.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[T03] Saved: {out_t03}")
    print(f"[T03] Stats: {json.dumps(payload['stats'], ensure_ascii=False)}")

    t05 = export_t05(payload)
    out_t05 = Path(args.out_t05)
    out_t05.parent.mkdir(parents=True, exist_ok=True)
    t05.to_csv(out_t05, index=False, encoding="utf-8")
    print(f"[T05] Saved: {out_t05} (rows={len(t05)})")


if __name__ == "__main__":
    main()
