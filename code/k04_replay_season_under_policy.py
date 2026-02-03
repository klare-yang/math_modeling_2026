#!/usr/bin/env python3
"""
k04_replay_season_under_policy.py
Model IV: replay seasons under each policy using posterior draws from k02.

Inputs:
- T05_model_ready_events_route2_v3B.csv (event structure, judge scores)
- k03_policy_grid.csv (policy grid)
- k02_posterior_samples.npz (pi_draws, event_id, contestant_id)

Output:
- k04_replay_paths.csv
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MIN_DRAWS = 100


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


def find_paths(base_dir: Path, out_dir: str | None) -> Tuple[Path, Path, Path]:
    out_candidates = [
        Path(out_dir) if out_dir else None,
        base_dir / "data" / "model4_results" / "out",
        base_dir / "model4_results" / "out",
        base_dir / "out",
        base_dir / "data" / "out",
    ]
    out = next((p for p in out_candidates if p and p.exists()), out_candidates[1])

    t05_name = "T05_model_ready_events_route2_v3B.csv"
    t05_candidates = [
        base_dir / "data" / t05_name,
        base_dir / t05_name,
        base_dir / "data" / "inputs" / t05_name,
    ]
    t05 = next((p for p in t05_candidates if p.exists()), t05_candidates[0])

    grid = out / "k03_policy_grid.csv"
    posterior = out / "k02_posterior_samples.npz"

    out.mkdir(parents=True, exist_ok=True)
    return t05, grid, posterior


def _as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [t.strip() for t in x.strip("[](){}").split(",") if t.strip()]
    return [x]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=None, help="Project base directory (auto-detected if omitted).")
    ap.add_argument("--out_dir", default=None, help="Optional override for output directory.")
    ap.add_argument("--n_draws", type=int, default=None, help="Optional cap on number of draws.")
    args = ap.parse_args()

    base_dir = guess_base_dir(args.base_dir)
    t05_path, grid_path, posterior_path = find_paths(base_dir, args.out_dir)

    if not t05_path.exists():
        raise FileNotFoundError(f"Cannot find T05 file: {t05_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing policy grid: {grid_path}")
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior draws: {posterior_path}")

    t05 = pd.read_csv(t05_path).reset_index(drop=True)
    grid = pd.read_csv(grid_path)

    # Parse list columns
    t05["active_ids"] = t05["active_ids"].apply(_as_list)
    t05["J_total_by_active"] = t05["J_total_by_active"].apply(_as_list)
    if "J_pct_by_active" in t05.columns:
        t05["J_pct_by_active"] = t05["J_pct_by_active"].apply(_as_list)

    # Load posterior draws
    post = np.load(posterior_path, allow_pickle=False)
    if "pi_draws" not in post:
        raise ValueError("k02_posterior_samples.npz missing pi_draws")
    pi_draws = post["pi_draws"]
    event_ids = post["event_id"].astype(str)
    contestant_ids = post["contestant_id"].astype(str)

    if pi_draws.ndim != 2:
        raise ValueError("pi_draws must be 2D (draws x rows)")

    n_draws_avail = int(pi_draws.shape[0])
    if n_draws_avail < MIN_DRAWS:
        raise ValueError(f"Posterior draws ({n_draws_avail}) < MIN_DRAWS ({MIN_DRAWS})")

    if args.n_draws is None:
        n_draws = n_draws_avail
    else:
        n_draws = min(int(args.n_draws), n_draws_avail)
        n_draws = max(n_draws, MIN_DRAWS)

    pi_draws = pi_draws[:n_draws, :]

    # Map (event_id, contestant_id) -> column index
    event_map: Dict[str, Dict[str, int]] = {}
    for idx, (eid, cid) in enumerate(zip(event_ids, contestant_ids)):
        event_map.setdefault(str(eid), {})[str(cid)] = int(idx)

    # Precompute event-level pi matrices (draws x n_active)
    event_pi_slices: List[np.ndarray] = []
    missing_total = 0
    for _, row in t05.iterrows():
        eid = str(row["event_id"]) if "event_id" in row else f"{row['season']}-{row['week']}"
        active_ids = [str(c) for c in row["active_ids"]]
        n_active = len(active_ids)
        idxs = np.array([event_map.get(eid, {}).get(cid, -1) for cid in active_ids], dtype=int)
        miss_mask = idxs < 0
        missing_total += int(miss_mask.sum())

        # Fallback probabilities for missing contestants
        if "J_pct_by_active" in row and row["J_pct_by_active"] is not None:
            fallback_raw = np.asarray(row["J_pct_by_active"], dtype=float)
        else:
            fallback_raw = np.asarray(row["J_total_by_active"], dtype=float)
        if fallback_raw.size != n_active:
            fallback_raw = np.ones(n_active, dtype=float)
        fallback_sum = fallback_raw.sum()
        if fallback_sum <= 0:
            fallback = np.ones(n_active, dtype=float) / max(n_active, 1)
        else:
            fallback = fallback_raw / fallback_sum

        block = np.zeros((n_draws, n_active), dtype=float)
        if (~miss_mask).any():
            block[:, ~miss_mask] = pi_draws[:, idxs[~miss_mask]]
        if miss_mask.any():
            block[:, miss_mask] = fallback[miss_mask]

        row_sums = block.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        block = block / row_sums
        event_pi_slices.append(block.astype(np.float32))

    results = []

    for _, p_row in grid.iterrows():
        pid = int(p_row["policy_id"])
        w = float(p_row["w"])
        delta = int(p_row["delta"])

        for d_idx in range(n_draws):
            for season in t05["season"].unique():
                season_events = t05[t05["season"] == season].sort_values("week")

                for _, e_row in season_events.iterrows():
                    if int(e_row["elim_count"]) == 0:
                        continue

                    active_ids = e_row["active_ids"]
                    J = np.array(e_row["J_total_by_active"], dtype=float)
                    pi = event_pi_slices[int(e_row.name)][d_idx]

                    sum_j = J.sum() if J.sum() > 0 else 1e-6
                    pj = J / sum_j

                    sum_pi = pi.sum() if pi.sum() > 0 else 1e-6
                    pv = pi / sum_pi

                    S = w * pj + (1.0 - w) * pv

                    elim_count = int(e_row["elim_count"])
                    eliminated_in_event = []
                    temp_active_indices = list(range(len(active_ids)))

                    for _ in range(elim_count):
                        sub_J = J[temp_active_indices]
                        sub_pi = pi[temp_active_indices]
                        sub_sum_j = sub_J.sum() if sub_J.sum() > 0 else 1e-6
                        sub_sum_pi = sub_pi.sum() if sub_pi.sum() > 0 else 1e-6
                        sub_S = w * (sub_J / sub_sum_j) + (1.0 - w) * (sub_pi / sub_sum_pi)

                        target_idx_in_sub = -1
                        if elim_count == 1 and delta > 0:
                            k_save = delta + 1
                            if len(sub_S) >= k_save:
                                sort_keys = []
                                for i in range(len(temp_active_indices)):
                                    idx_orig = temp_active_indices[i]
                                    sort_keys.append((sub_S[i], sub_J[i], active_ids[idx_orig], i))
                                sort_keys.sort()
                                bottom_k_meta = sort_keys[:k_save]
                                bottom_k_meta.sort(key=lambda x: (x[1], x[0], x[2]))
                                target_idx_in_sub = bottom_k_meta[0][3]

                        if target_idx_in_sub == -1:
                            sort_keys = []
                            for i in range(len(temp_active_indices)):
                                idx_orig = temp_active_indices[i]
                                sort_keys.append((sub_S[i], sub_J[i], active_ids[idx_orig], i))
                            sort_keys.sort()
                            target_idx_in_sub = sort_keys[0][3]

                        elim_idx_orig = temp_active_indices.pop(target_idx_in_sub)
                        eliminated_in_event.append(active_ids[elim_idx_orig])

                    results.append(
                        {
                            "season": int(season),
                            "week": int(e_row["week"]),
                            "draw_id": int(d_idx),
                            "policy_id": int(pid),
                            "eliminated_ids": "|".join([str(x) for x in eliminated_in_event]),
                        }
                    )

    out_dir = grid_path.parent
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_dir / "k04_replay_paths.csv", index=False)

    print(f"[k04] OK: draws={n_draws}, policies={len(grid)}, missing_map={missing_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
