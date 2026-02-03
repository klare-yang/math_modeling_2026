#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k06_pareto_and_selection.py
Model IV: build Pareto frontier and select a single policy (Utopia-distance).

Inputs:
  - k05_policy_objectives_summary.csv
  - k03_policy_grid.csv

Outputs:
  - k06_pareto_frontier.csv
  - k06_selected_policy.json
  - k06_compare_baselines.csv

Objectives:
  maximize f1_mean, maximize f2_mean, minimize f3_mean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def guess_base_dir(cli_base: str | None) -> Path:
    if cli_base:
        return Path(cli_base).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists() or (cwd / "model4_results").exists():
        return cwd
    return Path(__file__).resolve().parent


def find_out_dir(base_dir: Path) -> Path:
    candidates = [
        base_dir / "data" / "model4_results" / "out",
        base_dir / "model4_results" / "out",
        base_dir / "out",
        base_dir / "data" / "out",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-efficient points for MINIMIZATION costs."""
    n = costs.shape[0]
    is_eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_eff[i]:
            continue
        dominated = np.any(np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1))
        if dominated:
            is_eff[i] = False
    return is_eff


def norm01(x: np.ndarray) -> np.ndarray:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if np.isclose(xmax, xmin):
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=None, help="Project base directory (auto-detected if omitted).")
    args = ap.parse_args()

    base_dir = guess_base_dir(args.base_dir)
    out_dir = find_out_dir(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "k05_policy_objectives_summary.csv"
    grid_path = out_dir / "k03_policy_grid.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path} (run k05 first).")
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing {grid_path} (run k03 first).")

    summary = pd.read_csv(summary_path)
    grid = pd.read_csv(grid_path)
    df = summary.merge(grid, on="policy_id", how="inner")

    costs = np.column_stack(
        [-df["f1_mean"].to_numpy(float), -df["f2_mean"].to_numpy(float), df["f3_mean"].to_numpy(float)]
    )
    mask = pareto_efficient(costs)
    pareto = df.loc[mask].copy()
    pareto_out = out_dir / "k06_pareto_frontier.csv"
    pareto.to_csv(pareto_out, index=False)

    # Utopia selection on normalized benefit space, restricted to Pareto set
    nf1 = norm01(df["f1_mean"].to_numpy(float))
    nf2 = norm01(df["f2_mean"].to_numpy(float))
    nf3 = 1.0 - norm01(df["f3_mean"].to_numpy(float))
    dist = np.sqrt((1.0 - nf1) ** 2 + (1.0 - nf2) ** 2 + (1.0 - nf3) ** 2)

    pareto_idx = pareto.index.to_numpy()
    best_idx = int(pareto_idx[np.argmin(dist[pareto_idx])])
    sel = df.loc[best_idx]

    selected_json = {
        "policy_id": int(sel["policy_id"]),
        "w": float(sel["w"]),
        "delta": int(sel["delta"]),
        "objectives": {"f1": float(sel["f1_mean"]), "f2": float(sel["f2_mean"]), "f3": float(sel["f3_mean"])},
        "notes": {"selection": "min distance to utopia in normalized benefit space (Pareto set)"},
    }
    (out_dir / "k06_selected_policy.json").write_text(json.dumps(selected_json, indent=2))

    baselines = [
        {"label": "current", "w": 0.5, "delta": 0},
        {"label": "plus_delta1", "w": 0.5, "delta": 1},
        {"label": "plus_delta2", "w": 0.5, "delta": 2},
        {"label": "selected", "w": float(sel["w"]), "delta": int(sel["delta"])},
    ]
    rows = []
    for b in baselines:
        sub = df[(np.isclose(df["w"], b["w"])) & (df["delta"] == b["delta"])]
        if not sub.empty:
            r = sub.iloc[0]
            rows.append(
                {
                    "label": b["label"],
                    "w": float(r["w"]),
                    "delta": int(r["delta"]),
                    "f1": float(r["f1_mean"]),
                    "f2": float(r["f2_mean"]),
                    "f3": float(r["f3_mean"]),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "k06_compare_baselines.csv", index=False)

    print(f"[k06] pareto_points={len(pareto)}")
    print(f"[k06] wrote: {pareto_out.name}, k06_selected_policy.json, k06_compare_baselines.csv to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
