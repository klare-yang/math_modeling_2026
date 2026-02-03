#!/usr/bin/env python3
"""k07_make_figures.py
Model IV: generate summary plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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


def find_paths(base_dir: Path, out_dir: str | None, fig_dir: str | None) -> tuple[Path, Path]:
    out_candidates = [
        Path(out_dir) if out_dir else None,
        base_dir / "data" / "model4_results" / "out",
        base_dir / "model4_results" / "out",
        base_dir / "out",
        base_dir / "data" / "out",
    ]
    out = next((p for p in out_candidates if p and p.exists()), out_candidates[1])

    fig_candidates = [
        Path(fig_dir) if fig_dir else None,
        base_dir / "data" / "model4_results" / "fig",
        base_dir / "model4_results" / "fig",
        base_dir / "fig",
        base_dir / "data" / "fig",
    ]
    fig = next((p for p in fig_candidates if p and p.exists()), fig_candidates[1])
    fig.mkdir(parents=True, exist_ok=True)
    return out, fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=None, help="Project base directory (auto-detected if omitted).")
    ap.add_argument("--out_dir", default=None, help="Optional override for output directory.")
    ap.add_argument("--fig_dir", default=None, help="Optional override for figure directory.")
    args = ap.parse_args()

    base_dir = guess_base_dir(args.base_dir)
    out_dir, fig_dir = find_paths(base_dir, args.out_dir, args.fig_dir)

    summary = pd.read_csv(out_dir / "k05_policy_objectives_summary.csv")
    grid = pd.read_csv(out_dir / "k03_policy_grid.csv")
    df = summary.merge(grid, on="policy_id")
    pareto = pd.read_csv(out_dir / "k06_pareto_frontier.csv")
    baseline = pd.read_csv(out_dir / "k06_compare_baselines.csv")

    # 1. Pareto Scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(df["f1_mean"], df["f2_mean"], c=df["f3_mean"], cmap="viridis", alpha=0.5, label="All Policies")
    plt.scatter(pareto["f1_mean"], pareto["f2_mean"], color="red", marker="x", label="Pareto Frontier")
    plt.xlabel("Meritocracy (f1)")
    plt.ylabel("Excitement (f2)")
    plt.title("Pareto Frontier: f1 vs f2 (Color by f3)")
    plt.colorbar(label="Catastrophe Risk (f3)")
    plt.legend()
    plt.savefig(fig_dir / "k07_pareto_scatter.png")
    plt.close()

    # 2. Tradeoff Curves
    plt.figure(figsize=(10, 6))
    for d in [0, 1, 2]:
        sub = df[df["delta"] == d].sort_values("w")
        plt.plot(sub["w"], sub["f1_mean"], label=f"f1 (delta={d})", linestyle="-")
        plt.plot(sub["w"], sub["f3_mean"], label=f"f3 (delta={d})", linestyle="--")
    plt.xlabel("Judge Weight (w)")
    plt.ylabel("Objective Value")
    plt.title("Tradeoff Curves across w and delta")
    plt.legend()
    plt.savefig(fig_dir / "k07_tradeoff_curves.png")
    plt.close()

    # 3. Baseline vs Optimal Bar
    baseline.plot(kind="bar", x="label", y=["f1", "f2", "f3"], figsize=(10, 6))
    plt.title("Baseline vs Optimal Policy Objectives")
    plt.ylabel("Value")
    plt.xticks(rotation=0)
    plt.savefig(fig_dir / "k07_baseline_vs_opt_bar.png")
    plt.close()

    print("[k07] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
