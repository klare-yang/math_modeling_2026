#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k05_compute_objectives.py
Model IV: compute policy objectives from replay paths (k04) and event data (T05).

Outputs:
  - k05_policy_objectives_by_draw.csv
  - k05_policy_objectives_summary.csv

Objectives:
  f1 (Meritocracy): Spearman rank correlation between judge ranking and final placement.
  f2 (Excitement): Average bottom-50% survival fraction within elimination weeks.
  f3 (Catastrophe): Worst-by-judges contestant reaching finale probability.

Notes:
- No SciPy dependency; Spearman is computed via Pearson correlation of ranks.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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


def spearman_rho(x: List[float], y: List[float]) -> float:
    if len(x) <= 1:
        return float("nan")
    xr = pd.Series(np.asarray(x)).rank(method="average")
    yr = pd.Series(np.asarray(y)).rank(method="average")
    return float(xr.corr(yr))


def guess_base_dir(cli_base: str | None) -> Path:
    if cli_base:
        return Path(cli_base).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists() or (cwd / "model4_results").exists():
        return cwd
    return Path(__file__).resolve().parent


def find_paths(base_dir: Path) -> Tuple[Path, Path, Path]:
    out_candidates = [
        base_dir / "data" / "model4_results" / "out",
        base_dir / "model4_results" / "out",
        base_dir / "out",
        base_dir / "data" / "out",
    ]
    out_dir = next((p for p in out_candidates if p.exists()), out_candidates[0])

    t05_name = "T05_model_ready_events_route2_v3B.csv"
    t05_candidates = [
        base_dir / "data" / t05_name,
        base_dir / t05_name,
        base_dir / "data" / "inputs" / t05_name,
    ]
    t05_path = next((p for p in t05_candidates if p.exists()), t05_candidates[0])

    replay_candidates = [
        out_dir / "k04_replay_paths.csv",
        base_dir / "k04_replay_paths.csv",
        base_dir / "data" / "k04_replay_paths.csv",
    ]
    replay_path = next((p for p in replay_candidates if p.exists()), replay_candidates[0])

    return t05_path, replay_path, out_dir


def compute_jbar(t05: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in t05.iterrows():
        active_ids = r["active_ids"]
        j_vec = r["J_total_by_active"]
        for i, cid in enumerate(active_ids):
            rows.append({"season": int(r["season"]), "contestant_id": str(cid), "J": float(j_vec[i])})
    df = pd.DataFrame(rows)
    j_bar = (
        df.groupby(["season", "contestant_id"], as_index=False)["J"].mean().rename(columns={"J": "J_bar"})
    )
    return j_bar


def bottom_half_survival_fraction(active_ids: List[str], j_scores: np.ndarray, eliminated: List[str]) -> float:
    n = len(active_ids)
    if n == 0:
        return float("nan")
    k = int(np.ceil(0.5 * n))
    order = np.argsort(j_scores)  # ascending: worst first
    bottom_ids = {str(active_ids[i]) for i in order[:k]}
    elim_set = {str(c) for c in eliminated if c != ""}
    survive = bottom_ids - elim_set
    return len(survive) / max(len(bottom_ids), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=None, help="Project base directory (auto-detected if omitted).")
    ap.add_argument("--max_draws", type=int, default=None, help="Optional cap on number of draw_id values.")
    args = ap.parse_args()

    base_dir = guess_base_dir(args.base_dir)
    t05_path, replay_path, out_dir = find_paths(base_dir)

    if not t05_path.exists():
        raise FileNotFoundError(f"Cannot find T05 file: {t05_path}")
    if not replay_path.exists():
        raise FileNotFoundError(f"Cannot find replay file: {replay_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    t05 = pd.read_csv(t05_path)
    replay = pd.read_csv(replay_path)

    if "active_ids" not in t05.columns or "J_total_by_active" not in t05.columns:
        raise ValueError("T05 must contain columns: active_ids, J_total_by_active")

    t05["active_ids"] = t05["active_ids"].apply(_as_list)
    t05["J_total_by_active"] = t05["J_total_by_active"].apply(_as_list)
    t05["season"] = t05["season"].astype(int)
    t05["week"] = t05["week"].astype(int)

    need_cols = {"season", "week", "draw_id", "policy_id", "eliminated_ids"}
    missing = need_cols - set(replay.columns)
    if missing:
        raise ValueError(f"Replay file missing columns: {sorted(missing)}")

    replay["season"] = replay["season"].astype(int)
    replay["week"] = replay["week"].astype(int)
    replay["draw_id"] = replay["draw_id"].astype(int)
    replay["policy_id"] = replay["policy_id"].astype(int)

    if args.max_draws is not None:
        draw_ids = sorted(replay["draw_id"].unique())[: args.max_draws]
        replay = replay[replay["draw_id"].isin(draw_ids)].copy()

    j_bar = compute_jbar(t05)
    worst_judge = j_bar.loc[j_bar.groupby("season")["J_bar"].idxmin()].copy()
    worst_map = worst_judge.set_index("season")["contestant_id"].to_dict()

    sw_lookup: Dict[Tuple[int, int], Tuple[List[str], np.ndarray]] = {}
    for _, r in t05.iterrows():
        key = (int(r["season"]), int(r["week"]))
        active_ids = [str(c) for c in r["active_ids"]]
        j_vec = np.asarray([float(v) for v in r["J_total_by_active"]], dtype=float)
        sw_lookup[key] = (active_ids, j_vec)

    metrics = []
    for (pid, did), g in replay.groupby(["policy_id", "draw_id"]):
        season_f1, season_f2, season_f3 = [], [], []
        for season, sg in g.groupby("season"):
            sg = sg.sort_values("week")
            elim_week: Dict[str, int] = {}
            elim_all: List[str] = []
            f2_vals = []

            for _, rr in sg.iterrows():
                key = (int(season), int(rr["week"]))
                if key not in sw_lookup:
                    continue
                active_ids, j_vec = sw_lookup[key]
                eliminated = [c for c in str(rr["eliminated_ids"]).split("|") if c != "nan"]
                eliminated = [c for c in eliminated if c != ""]
                f2_vals.append(bottom_half_survival_fraction(active_ids, j_vec, eliminated))

                for c in eliminated:
                    c = str(c)
                    elim_all.append(c)
                    if c not in elim_week:
                        elim_week[c] = int(rr["week"])

            s_jbar = j_bar[j_bar["season"] == int(season)].copy()
            all_cids = [str(c) for c in s_jbar["contestant_id"].tolist()]
            if len(sg):
                max_week = int(sg["week"].max())
            else:
                max_week = int(t05[t05["season"] == int(season)]["week"].max())

            place_rows = []
            for cid in all_cids:
                surv_t = elim_week.get(cid, max_week + 1)
                jb = float(s_jbar[s_jbar["contestant_id"] == cid]["J_bar"].iloc[0])
                place_rows.append((cid, surv_t, jb))
            place_df = pd.DataFrame(place_rows, columns=["contestant_id", "survival_time", "J_bar"])
            place_df = place_df.sort_values(["survival_time", "J_bar"], ascending=[False, False]).reset_index(drop=True)
            place_df["R_final"] = np.arange(1, len(place_df) + 1, dtype=int)

            s_jbar["R_J"] = s_jbar["J_bar"].rank(ascending=False, method="average")
            rj_map = s_jbar.set_index("contestant_id")["R_J"].to_dict()
            rf_map = place_df.set_index("contestant_id")["R_final"].to_dict()

            cids = [c for c in all_cids if c in rj_map and c in rf_map]
            if len(cids) > 1:
                rho = spearman_rho([rj_map[c] for c in cids], [rf_map[c] for c in cids])
                if not np.isnan(rho):
                    season_f1.append(rho)

            if len(f2_vals):
                season_f2.append(float(np.nanmean(f2_vals)))

            worst_cid = str(worst_map.get(int(season), ""))
            if worst_cid:
                season_f3.append(1.0 if worst_cid not in set(elim_all) else 0.0)

        metrics.append(
            {
                "policy_id": int(pid),
                "draw_id": int(did),
                "f1": float(np.mean(season_f1)) if len(season_f1) else 0.0,
                "f2": float(np.mean(season_f2)) if len(season_f2) else 0.0,
                "f3": float(np.mean(season_f3)) if len(season_f3) else 0.0,
            }
        )

    df = pd.DataFrame(metrics).sort_values(["policy_id", "draw_id"])
    out_by_draw = out_dir / "k05_policy_objectives_by_draw.csv"
    df.to_csv(out_by_draw, index=False)

    summary = df.groupby("policy_id")[["f1", "f2", "f3"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["policy_id", "f1_mean", "f1_std", "f2_mean", "f2_std", "f3_mean", "f3_std"]
    out_summary = out_dir / "k05_policy_objectives_summary.csv"
    summary.to_csv(out_summary, index=False)

    print(f"[k05] draws={df['draw_id'].nunique()}, policies={df['policy_id'].nunique()}")
    print(f"[k05] wrote: {out_by_draw.name}, {out_summary.name} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
