#!/usr/bin/env python3
"""
k02_load_posterior_samples.py
Model IV: construct Monte Carlo posterior draws WITHOUT .nc.

Source:
- Uses T08b_votes_posterior_stats_route2_v3B.csv (pi_mean, pi_sd) to build
  Dirichlet draws per event (moment-matched via alpha0). This preserves
  event-level vote share structure without requiring netCDF4/xarray.

Outputs:
- k02_posterior_samples.npz (pi_draws, event_id, contestant_id)
- k02_posterior_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

MIN_DRAWS = 100
DEFAULT_SEED = 20260202


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


def find_paths(base_dir: Path, t08b_path: str | None, out_dir: str | None) -> tuple[Path, Path]:
    t08b_name = "T08b_votes_posterior_stats_route2_v3B.csv"
    t08b_candidates = [
        Path(t08b_path) if t08b_path else None,
        base_dir / "data" / t08b_name,
        base_dir / t08b_name,
        base_dir / "data" / "inputs" / t08b_name,
    ]
    t08b = next((p for p in t08b_candidates if p and p.exists()), t08b_candidates[1])

    out_candidates = [
        Path(out_dir) if out_dir else None,
        base_dir / "data" / "model4_results" / "out",
        base_dir / "model4_results" / "out",
        base_dir / "out",
        base_dir / "data" / "out",
    ]
    out = next((p for p in out_candidates if p and p.exists()), out_candidates[1])
    out.mkdir(parents=True, exist_ok=True)
    return t08b, out


def estimate_alpha0(m: np.ndarray, sd: np.ndarray) -> np.ndarray:
    v = sd * sd
    denom = m * (1.0 - m)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha0 = denom / v - 1.0
    alpha0 = alpha0[np.isfinite(alpha0) & (alpha0 > 0)]
    return alpha0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=None, help="Project base directory (auto-detected if omitted).")
    ap.add_argument("--t08b_path", default=None, help="Optional override for T08b stats CSV path.")
    ap.add_argument("--out_dir", default=None, help="Optional override for output directory.")
    ap.add_argument("--n_draws", type=int, default=200, help="Number of Monte Carlo draws (min 100).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for reproducibility.")
    args = ap.parse_args()

    base_dir = guess_base_dir(args.base_dir)
    t08b_path, out_dir = find_paths(base_dir, args.t08b_path, args.out_dir)

    if not t08b_path.exists():
        raise FileNotFoundError(f"Cannot find T08b stats file: {t08b_path}")

    n_draws = max(int(args.n_draws), MIN_DRAWS)
    rng = np.random.default_rng(int(args.seed))

    df = pd.read_csv(t08b_path)
    need_cols = {"event_id", "contestant_id", "pi_mean", "pi_sd"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"T08b missing columns: {sorted(missing)}")

    df = df.reset_index(drop=True)
    m_all = df["pi_mean"].to_numpy(float)
    sd_all = df["pi_sd"].to_numpy(float)
    alpha0_all = estimate_alpha0(m_all, sd_all)
    global_alpha0 = float(np.median(alpha0_all)) if alpha0_all.size else 50.0

    # Per-event alpha0 (median of component-wise estimates)
    alpha0_by_event: Dict[str, float] = {}
    fallback_events = 0
    for event_id, sub in df.groupby("event_id", sort=False):
        alpha0 = estimate_alpha0(sub["pi_mean"].to_numpy(float), sub["pi_sd"].to_numpy(float))
        if alpha0.size == 0:
            alpha0_by_event[str(event_id)] = global_alpha0
            fallback_events += 1
        else:
            alpha0_by_event[str(event_id)] = float(np.median(alpha0))

    min_alpha = 1.0e-4
    pi_draws = np.empty((n_draws, len(df)), dtype=np.float32)

    for event_id, sub in df.groupby("event_id", sort=False):
        idx = sub.index.to_numpy()
        m = sub["pi_mean"].to_numpy(float)
        alpha0 = alpha0_by_event[str(event_id)]
        alpha = np.maximum(m * alpha0, min_alpha)
        gamma = rng.gamma(shape=alpha, scale=1.0, size=(n_draws, len(alpha)))
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        pi = gamma / row_sums
        pi_draws[:, idx] = pi.astype(np.float32)

    event_id_arr = np.asarray(df["event_id"].astype(str).to_numpy(), dtype="U")
    contestant_id_arr = np.asarray(df["contestant_id"].astype(str).to_numpy(), dtype="U")
    method_arr = np.asarray(["dirichlet_moment_match"], dtype="U")

    np.savez_compressed(
        out_dir / "k02_posterior_samples.npz",
        pi_draws=pi_draws,
        event_id=event_id_arr,
        contestant_id=contestant_id_arr,
        seed=np.array([int(args.seed)], dtype=np.int64),
        n_draws=np.array([int(n_draws)], dtype=np.int64),
        method=method_arr,
    )

    manifest = {
        "base_dir": str(base_dir),
        "t08b_path": str(t08b_path),
        "out_dir": str(out_dir),
        "n_draws": int(n_draws),
        "seed": int(args.seed),
        "method": "dirichlet_moment_match",
        "alpha0_global_median": float(global_alpha0),
        "fallback_events": int(fallback_events),
        "rows": int(len(df)),
        "events": int(df["event_id"].nunique()),
    }
    (out_dir / "k02_posterior_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[k02] OK: draws={n_draws}, events={manifest['events']}, rows={manifest['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
