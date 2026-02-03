#!/usr/bin/env python3
"""k03_define_policy_grid.py
Model IV: build policy grid for (w, delta).
"""
from __future__ import annotations

import argparse
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


def find_out_dir(base_dir: Path, out_dir: str | None) -> Path:
    candidates = [
        Path(out_dir) if out_dir else None,
        base_dir / "data" / "model4_results" / "out",
        base_dir / "model4_results" / "out",
        base_dir / "out",
        base_dir / "data" / "out",
    ]
    out = next((p for p in candidates if p and p.exists()), candidates[1])
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=None, help="Project base directory (auto-detected if omitted).")
    ap.add_argument("--out_dir", default=None, help="Optional override for output directory.")
    args = ap.parse_args()

    base_dir = guess_base_dir(args.base_dir)
    out_dir = find_out_dir(base_dir, args.out_dir)

    w_values = [round(x * 0.05, 2) for x in range(21)]  # 0.0 to 1.0 step 0.05
    delta_values = [0, 1, 2]

    grid = []
    policy_id = 0
    for w in w_values:
        for delta in delta_values:
            grid.append({"policy_id": policy_id, "w": w, "delta": delta})
            policy_id += 1

    df_grid = pd.DataFrame(grid)
    df_grid.to_csv(out_dir / "k03_policy_grid.csv", index=False)
    print(f"[k03] OK: policies={len(df_grid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
