#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T03 (Route2 v1): Build per-(season,week) active set and eliminated set from the long panel table,
then export:
  - data/T03_active_elimination_structure_route2.json
  - data/T05_model_ready_events_route2.csv

Schema contract:
  contestant_key := f"S{season}:{celebrity_name}"
  events[event_id] contains active_ids, eliminated_ids, eliminated_indices (all stable, sorted).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd


def contestant_key(season: int, celebrity_name: str) -> str:
    return f"S{int(season)}:{str(celebrity_name)}"


def build_events(
    panel: pd.DataFrame,
    min_judges: int = 1,
    only_consecutive_weeks: bool = True,
) -> Dict[str, Any]:
    # required columns
    req = {"celebrity_name", "season", "week", "judge_score_total", "judge_score_count"}
    missing = req - set(panel.columns)
    if missing:
        raise ValueError(f"[T03] Panel missing required columns: {sorted(missing)}")

    df = panel.copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["contestant_key"] = df.apply(lambda r: contestant_key(r["season"], r["celebrity_name"]), axis=1)

    # active definition
    df["active"] = (df["judge_score_count"].fillna(0).astype(int) >= int(min_judges)) & df["judge_score_total"].notna()

    events: Dict[str, Any] = {}
    stats_elim_counts: List[int] = []
    max_active = 0
    max_elim = 0
    n_events = 0

    for season, gS in df.groupby("season", sort=True):
        weeks = sorted(gS["week"].unique().tolist())
        for idx_w, w in enumerate(weeks):
            n_events += 1
            next_week = weeks[idx_w + 1] if idx_w + 1 < len(weeks) else None
            delta_week = (next_week - w) if next_week is not None else None

            gW = gS[gS["week"] == w]
            active_ids = sorted(gW.loc[gW["active"], "contestant_key"].unique().tolist())
            max_active = max(max_active, len(active_ids))

            eliminated_ids: List[str] = []
            terminal = next_week is None

            if not terminal:
                gNext = gS[gS["week"] == next_week]
                active_next = set(sorted(gNext.loc[gNext["active"], "contestant_key"].unique().tolist()))
                active_now = set(active_ids)

                if (not only_consecutive_weeks) or (delta_week == 1):
                    eliminated_ids = sorted(list(active_now - active_next))
                else:
                    eliminated_ids = []

            eliminated_indices = [active_ids.index(x) for x in eliminated_ids]  # stable by sorting
            elim_count = len(eliminated_ids)

            max_elim = max(max_elim, elim_count)
            stats_elim_counts.append(elim_count)

            event_id = f"{season}-{w}"
            events[event_id] = {
                "season": int(season),
                "week": int(w),
                "next_week": int(next_week) if next_week is not None else None,
                "delta_week": int(delta_week) if delta_week is not None else None,
                "active_ids": active_ids,
                "eliminated_ids": eliminated_ids,
                "eliminated_indices": eliminated_indices,
                "n_active": len(active_ids),
                "elim_count": elim_count,
                "terminal": bool(terminal),
            }

    payload = {
        "schema_version": "route2.v1",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config": {
            "min_judges": int(min_judges),
            "only_consecutive_weeks": bool(only_consecutive_weeks),
            "active_definition": "judge_score_count>=min_judges AND judge_score_total notna",
            "contestant_key": "S{season}:{celebrity_name}",
            "event_id": "season-week",
        },
        "stats": {
            "n_events": int(n_events),
            "max_active": int(max_active),
            "max_elim": int(max_elim),
            "elim_count_hist": dict(pd.Series(stats_elim_counts).value_counts().sort_index().to_dict()),
        },
        "events": events,
    }
    return payload


def export_t05(events_payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for event_id, ev in events_payload["events"].items():
        rows.append(
            {
                "season": ev["season"],
                "week": ev["week"],
                "next_week": ev["next_week"],
                "delta_week": ev["delta_week"],
                "n_active": ev["n_active"],
                "elim_count": ev["elim_count"],
                # JSON strings (stable, parseable)
                "active_ids": json.dumps(ev["active_ids"], ensure_ascii=False),
                "eliminated_ids": json.dumps(ev["eliminated_ids"], ensure_ascii=False),
                "eliminated_indices": json.dumps(ev["eliminated_indices"], ensure_ascii=False),
            }
        )
    df = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_panel", default="data/T02_long_format_panel.csv")
    ap.add_argument("--out_t03", default="data/T03_active_elimination_structure_route2.json")
    ap.add_argument("--out_t05", default="data/T05_model_ready_events_route2.csv")
    ap.add_argument("--min_judges", type=int, default=1)
    ap.add_argument("--only_consecutive_weeks", action="store_true", default=True)
    ap.add_argument("--allow_nonconsecutive", action="store_true", default=False)
    args = ap.parse_args()

    only_consecutive = args.only_consecutive_weeks and (not args.allow_nonconsecutive)

    in_path = Path(args.input_panel)
    if not in_path.exists():
        raise FileNotFoundError(f"[T03] input_panel not found: {in_path}")

    panel = pd.read_csv(in_path)
    payload = build_events(panel, min_judges=args.min_judges, only_consecutive_weeks=only_consecutive)

    out_t03 = Path(args.out_t03)
    out_t03.parent.mkdir(parents=True, exist_ok=True)
    out_t03.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[T03] Saved: {out_t03}  (n_events={payload['stats']['n_events']}, max_active={payload['stats']['max_active']}, max_elim={payload['stats']['max_elim']})")

    t05 = export_t05(payload)
    out_t05 = Path(args.out_t05)
    out_t05.parent.mkdir(parents=True, exist_ok=True)
    t05.to_csv(out_t05, index=False, encoding="utf-8")
    print(f"[T05] Saved: {out_t05}  (rows={len(t05)})")


if __name__ == "__main__":
    main()
