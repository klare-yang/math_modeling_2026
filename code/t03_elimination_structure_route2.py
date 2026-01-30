# code/t03_elimination_structure_route2.py
# -*- coding: utf-8 -*-

"""
T03 (Route2, Normalized) — Build season-week event structure with multi-elimination support.

Canonical output schema (per event_key = "{season}-{week}"):

{
  "schema_version": "route2_norm_v1",
  "generated_at": "...ISO8601...",
  "events": {
      "1-1": {
          "season": 1,
          "week": 1,
          "next_week": 2,
          "delta_week": 1,
          "active_ids": ["S1:NameA", ...],            # sorted, unique
          "eliminated_ids": ["S1:NameX", ...],        # sorted, unique, subset of active_ids
          "eliminated_indices": [int, ...],           # positions in active_ids
          "n_active": int,
          "elim_count": int,
          "terminal": bool
      },
      ...
  }
}

Elimination definition:
- "eliminated in week t" = active(t) \ active(next_observed_week).

Active rule (kept consistent with your pipeline):
- judge_score_count > 0 AND judge_score_total > 0
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_PANEL = Path("data/T02_long_format_panel.csv")
OUTPUT_JSON = Path("data/T03_active_elimination_structure_route2.json")
OUTPUT_CSV  = Path("data/T03_active_elimination_structure_route2.csv")  # optional snapshot

ONLY_CONSECUTIVE_WEEKS = False
ACTIVE_RULE = "count_and_total_positive"  # supported: "count_and_total_positive"
SCHEMA_VERSION = "route2_norm_v1"


def normalize_name(x: str) -> str:
    return " ".join(str(x).strip().split())


def make_contestant_key(season: int, celebrity_name: str) -> str:
    return f"S{int(season)}:{normalize_name(celebrity_name)}"


def is_active_row(row) -> bool:
    if ACTIVE_RULE == "count_and_total_positive":
        return (row["judge_score_count"] > 0) and (row["judge_score_total"] > 0)
    raise ValueError(f"Unknown ACTIVE_RULE={ACTIVE_RULE}")


def main():
    df = pd.read_csv(INPUT_PANEL)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["celebrity_name"] = df["celebrity_name"].astype(str).map(normalize_name)

    active_df = df[df.apply(is_active_row, axis=1)].copy()
    active_df["contestant_key"] = active_df.apply(
        lambda r: make_contestant_key(r["season"], r["celebrity_name"]), axis=1
    )

    active_sets = {
        (int(season), int(week)): sorted(set(g["contestant_key"].tolist()))
        for (season, week), g in active_df.groupby(["season", "week"])
    }

    seasons = sorted(active_df["season"].unique().astype(int).tolist())

    events = {}
    stats = {
        "num_seasons": len(seasons),
        "num_events_total": 0,
        "num_events_kept": 0,
        "num_terminal": 0,
        "elim_count_hist": {},
        "anomalies": {
            "non_consecutive_skipped": 0,
            "elim_not_subset_active": 0,
            "next_active_mismatch": 0,
        }
    }

    for season in seasons:
        weeks = sorted(active_df.loc[active_df["season"] == season, "week"].unique().astype(int).tolist())
        for i, w_now in enumerate(weeks):
            key = f"{season}-{w_now}"
            A = active_sets.get((season, w_now), [])
            A_set = set(A)

            stats["num_events_total"] += 1

            if i == len(weeks) - 1:
                events[key] = {
                    "season": int(season),
                    "week": int(w_now),
                    "next_week": None,
                    "delta_week": None,
                    "active_ids": A,
                    "eliminated_ids": [],
                    "eliminated_indices": [],
                    "n_active": int(len(A)),
                    "elim_count": 0,
                    "terminal": True,
                }
                stats["num_terminal"] += 1
                stats["num_events_kept"] += 1
                continue

            w_next = int(weeks[i + 1])
            delta = int(w_next - w_now)

            if ONLY_CONSECUTIVE_WEEKS and delta != 1:
                stats["anomalies"]["non_consecutive_skipped"] += 1
                continue

            B = active_sets.get((season, w_next), [])
            B_set = set(B)

            eliminated = sorted(list(A_set - B_set))
            elim_count = int(len(eliminated))

            if not set(eliminated).issubset(A_set):
                stats["anomalies"]["elim_not_subset_active"] += 1

            idx_map = {cid: j for j, cid in enumerate(A)}
            eliminated_indices = [int(idx_map[e]) for e in eliminated]

            expected_B = sorted(list(A_set - set(eliminated)))
            if expected_B != sorted(list(B_set)):
                stats["anomalies"]["next_active_mismatch"] += 1

            events[key] = {
                "season": int(season),
                "week": int(w_now),
                "next_week": int(w_next),
                "delta_week": int(delta),
                "active_ids": A,
                "eliminated_ids": eliminated,
                "eliminated_indices": eliminated_indices,
                "n_active": int(len(A)),
                "elim_count": elim_count,
                "terminal": False,
            }

            stats["elim_count_hist"][elim_count] = stats["elim_count_hist"].get(elim_count, 0) + 1
            stats["num_events_kept"] += 1

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "ONLY_CONSECUTIVE_WEEKS": ONLY_CONSECUTIVE_WEEKS,
            "ACTIVE_RULE": ACTIVE_RULE,
            "INPUT_PANEL": str(INPUT_PANEL),
        },
        "stats": stats,
        "events": events,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    rows = []
    for ek, e in events.items():
        rows.append({
            "event_key": ek,
            "season": e["season"],
            "week": e["week"],
            "next_week": e["next_week"],
            "delta_week": e["delta_week"],
            "n_active": e["n_active"],
            "elim_count": e["elim_count"],
            "terminal": e["terminal"],
            "active_ids": json.dumps(e["active_ids"], ensure_ascii=False),
            "eliminated_ids": json.dumps(e["eliminated_ids"], ensure_ascii=False),
            "eliminated_indices": json.dumps(e["eliminated_indices"], ensure_ascii=False),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print("\n=== T03 (Route2, Normalized) SUMMARY ===")
    print(f"Output JSON: {OUTPUT_JSON}")
    print(f"Output CSV : {OUTPUT_CSV}")
    print(f"Events kept: {stats['num_events_kept']} (terminal={stats['num_terminal']})")
    print("elim_count histogram:")
    for k in sorted(stats["elim_count_hist"].keys()):
        print(f"  elim_count={k}: {stats['elim_count_hist'][k]}")
    print("Anomalies:")
    for k, v in stats["anomalies"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
