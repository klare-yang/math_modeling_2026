# code/t05_prepare_model_data_route2.py
# -*- coding: utf-8 -*-

"""
T05 (Route2, Normalized) — Materialize model-ready event table from canonical T03 payload.

- One row per event.
- List fields are stored as JSON strings (stable round-trip in pandas).

Input:
- data/T03_active_elimination_structure_route2.json

Output:
- data/T05_model_ready_events_route2.csv
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

INPUT_T03 = Path("data/T03_active_elimination_structure_route2.json")
OUTPUT_CSV = Path("data/T05_model_ready_events_route2.csv")

DROP_TERMINAL = True
DROP_ZERO_ELIM = False
SCHEMA_VERSION_REQUIRED = "route2_norm_v1"


def load_t03(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "events" in obj and "schema_version" in obj:
        return obj
    raise ValueError("T03 is not normalized payload. Re-run T03.")


def main():
    payload = load_t03(INPUT_T03)
    if payload.get("schema_version") != SCHEMA_VERSION_REQUIRED:
        raise ValueError(f"schema_version mismatch: {payload.get('schema_version')}")

    events = payload["events"]
    rows = []
    for ek, e in events.items():
        if DROP_TERMINAL and e.get("terminal", False):
            continue
        if DROP_ZERO_ELIM and int(e.get("elim_count", 0)) == 0:
            continue

        rows.append({
            "event_key": ek,
            "season": int(e["season"]),
            "week": int(e["week"]),
            "next_week": e["next_week"] if e["next_week"] is None else int(e["next_week"]),
            "delta_week": e["delta_week"] if e["delta_week"] is None else int(e["delta_week"]),
            "n_active": int(e["n_active"]),
            "elim_count": int(e["elim_count"]),
            "terminal": bool(e["terminal"]),
            "active_ids": json.dumps(e["active_ids"], ensure_ascii=False),
            "eliminated_ids": json.dumps(e["eliminated_ids"], ensure_ascii=False),
            "eliminated_indices": json.dumps(e["eliminated_indices"], ensure_ascii=False),
        })

    df = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n=== T05 (Route2, Normalized) SUMMARY ===")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Rows: {len(df)}")
    print(df["elim_count"].value_counts().sort_index().to_string())
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
