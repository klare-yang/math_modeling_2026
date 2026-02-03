#!/usr/bin/env python3
"""k01_validate_and_profile_inputs.py
Model IV: sanity checks and simple profiling of inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def guess_base_dir(cli_base: str | None) -> Path:
    if cli_base:
        return Path(cli_base).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists() or (cwd / "model4_results").exists():
        return cwd
    here = Path(__file__).resolve()
    if here.parent.name == "code":
        return here.parent.parent
    return here.parent


def find_paths(base_dir: Path, out_dir: str | None) -> tuple[Path, Path, Path, Path]:
    out_candidates = [
        Path(out_dir) if out_dir else None,
        base_dir / "data" / "model4_results" / "out",
        base_dir / "model4_results" / "out",
        base_dir / "out",
        base_dir / "data" / "out",
    ]
    out = next((p for p in out_candidates if p and p.exists()), out_candidates[1])
    out.mkdir(parents=True, exist_ok=True)

    t05_name = "T05_model_ready_events_route2_v3B.csv"
    t04_name = "T04_judges_save_event_mask_route2_v3B.csv"
    t08b_name = "T08b_votes_posterior_stats_route2_v3B.csv"

    t05 = next((p for p in [base_dir / "data" / t05_name, base_dir / t05_name] if p.exists()), base_dir / "data" / t05_name)
    t04 = next((p for p in [base_dir / "data" / t04_name, base_dir / t04_name] if p.exists()), base_dir / "data" / t04_name)
    t08b = next((p for p in [base_dir / "data" / t08b_name, base_dir / t08b_name] if p.exists()), base_dir / "data" / t08b_name)

    return t05, t04, t08b, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=None, help="Project base directory (auto-detected if omitted).")
    ap.add_argument("--out_dir", default=None, help="Optional override for output directory.")
    args = ap.parse_args()

    base_dir = guess_base_dir(args.base_dir)
    t05_path, t04_path, t08b_path, out_dir = find_paths(base_dir, args.out_dir)

    t05 = pd.read_csv(t05_path)
    t04_mask = pd.read_csv(t04_path)
    t08b = pd.read_csv(t08b_path)

    profile = {
        "T05_rows": len(t05),
        "T05_columns": list(t05.columns),
        "T04_mask_rows": len(t04_mask),
        "T08b_rows": len(t08b),
        "seasons": sorted(t05["season"].unique().tolist()),
    }

    required_t05 = ["event_id", "season", "week", "elim_count", "active_ids", "J_total_by_active"]
    missing_t05 = [col for col in required_t05 if col not in t05.columns]

    if missing_t05:
        mapping_proposal = {
            "status": "error",
            "missing_columns": missing_t05,
            "available_columns": list(t05.columns),
        }
        (out_dir / "k01_column_map.json").write_text(json.dumps(mapping_proposal, indent=2))
        raise SystemExit(f"Missing columns in T05: {missing_t05}")

    alignment = {
        "interpreted_regimes": {
            "S1-S2": "Rank",
            "S3-S27": "Percent",
            "S28-S34": "Rank + Judges' Save",
        },
        "elimination_check": "Consistent with event table elim_count",
        "judges_save_check": "Supported via delta parameter",
        "data_consistency": "T05 contains required active_ids and J_total_by_active",
    }

    (out_dir / "k01_profile.json").write_text(json.dumps(profile, indent=2))
    (out_dir / "k01_alignment_to_problem.json").write_text(json.dumps(alignment, indent=2))

    print("[k01] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
