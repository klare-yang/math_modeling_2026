#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T04 (route2_v3B): Flag judges-save exceptions and produce a per-event mask for two-stage likelihood.

Single change:
- Produce:
    use_two_stage = (is_judges_save==True) AND (elim_count==1) AND (n_active>=2) AND (not terminal)
  else fallback_one_stage with an explicit reason.

Outputs:
- data/T04_judges_save_event_mask_route2_v3B.csv
- data/T04_judges_save_exceptions_route2_v3B.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _jload(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_t05", default="data/T05_model_ready_events_route2_v3B.csv")
    ap.add_argument("--out_mask", default="data/T04_judges_save_event_mask_route2_v3B.csv")
    ap.add_argument("--out_exceptions", default="data/T04_judges_save_exceptions_route2_v3B.csv")
    args = ap.parse_args()

    in_path = Path(args.input_t05)
    if not in_path.exists():
        raise FileNotFoundError(f"[T04_v3B] input_t05 not found: {in_path}")

    t05 = pd.read_csv(in_path)
    req = {"event_id", "season", "week", "terminal", "n_active", "elim_count", "is_judges_save"}
    miss = req - set(t05.columns)
    if miss:
        raise ValueError(f"[T04_v3B] T05 missing columns: {sorted(miss)}")

    # ensure types
    t05["season"] = t05["season"].astype(int)
    t05["week"] = t05["week"].astype(int)
    t05["terminal"] = t05["terminal"].astype(bool)
    t05["n_active"] = t05["n_active"].astype(int)
    t05["elim_count"] = t05["elim_count"].astype(int)
    t05["is_judges_save"] = t05["is_judges_save"].astype(bool)

    rows: List[Dict[str, Any]] = []
    exc_rows: List[Dict[str, Any]] = []

    n_total = len(t05)
    n_js = int(t05["is_judges_save"].sum())
    n_two_stage = 0

    for _, r in t05.iterrows():
        use_two_stage = False
        reason = ""

        if not r["is_judges_save"]:
            reason = "not_judges_save"
        elif r["terminal"]:
            reason = "terminal_week"
        elif r["n_active"] < 2:
            reason = "n_active_lt_2"
        elif r["elim_count"] != 1:
            reason = f"elim_count_ne_1({int(r['elim_count'])})"
        else:
            use_two_stage = True
            reason = "ok"

        if use_two_stage:
            n_two_stage += 1

        out_row = {
            "event_id": r["event_id"],
            "season": int(r["season"]),
            "week": int(r["week"]),
            "is_judges_save": bool(r["is_judges_save"]),
            "terminal": bool(r["terminal"]),
            "n_active": int(r["n_active"]),
            "elim_count": int(r["elim_count"]),
            "use_two_stage": bool(use_two_stage),
            "fallback_one_stage": bool(not use_two_stage),
            "reason": reason,
        }
        rows.append(out_row)

        if r["is_judges_save"] and (not use_two_stage):
            exc_rows.append(out_row)

    mask_df = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
    exc_df = pd.DataFrame(exc_rows).sort_values(["season", "week"]).reset_index(drop=True)

    out_mask = Path(args.out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    mask_df.to_csv(out_mask, index=False)

    out_exc = Path(args.out_exceptions)
    out_exc.parent.mkdir(parents=True, exist_ok=True)
    exc_df.to_csv(out_exc, index=False)

    print(f"[T04_v3B] total events: {n_total}")
    print(f"[T04_v3B] judges-save events: {n_js}")
    print(f"[T04_v3B] use_two_stage events: {n_two_stage}")
    print(f"[T04_v3B] exceptions among judges-save: {len(exc_df)}")
    print(f"[T04_v3B] Saved mask: {out_mask}")
    print(f"[T04_v3B] Saved exceptions: {out_exc}")

    print("[T04_v3B] Interpretation hints:")
    print("  - use_two_stage selects events where the two-stage judges-save likelihood is valid.")
    print("  - exceptions typically correspond to double-elimination / special weeks; they must fall back to one-stage.")


if __name__ == "__main__":
    main()
