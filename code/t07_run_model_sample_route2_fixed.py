#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T07 (Route2 v1): Build PyMC hierarchical model with multi-elimination likelihood (sequential without replacement),
including:
  - Input consistency checks (saved as JSON report)
  - Score normalization (z-score per season)
  - Sampling (NUTS) and saving inference data

Inputs:
  data/T05_model_ready_events_route2.csv
  data/T02_long_format_panel.csv

Outputs:
  data/T07_input_consistency_report.json
  data/T07_trace_route2.nc
  data/T07_posterior_mean_consistency.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import pymc as pm
import pymc as pm
import pytensor.tensor as pt

logsumexp = pm.math.logsumexp



def _parse_list_cell(x: Any) -> list:
    """Robustly parse list-like cells that may be JSON string or Python repr."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if s == "":
        return []
    # Try JSON first
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fallback: Python literal
    import ast
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Cannot parse list cell: {x}") from e


def contestant_key(season: int, celebrity_name: str) -> str:
    return f"S{int(season)}:{str(celebrity_name)}"


@dataclass
class ModelInput:
    # dimensions
    n_events: int
    n_contestants: int
    max_active: int
    max_elim: int

    # arrays
    active_key_idx: np.ndarray      # (E, A) int32
    score_z: np.ndarray             # (E, A) float32
    active_mask: np.ndarray         # (E, A) bool
    elim_idx: np.ndarray            # (E, K) int32 (padded)
    elim_mask: np.ndarray           # (E, K) bool

    # metadata
    event_ids: List[str]
    contestant_keys: List[str]
    season_of_event: np.ndarray     # (E,) int32
    week_of_event: np.ndarray       # (E,) int32


def load_inputs(t05_path: Path, panel_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not t05_path.exists():
        raise FileNotFoundError(f"[T07] missing: {t05_path}")
    if not panel_path.exists():
        raise FileNotFoundError(f"[T07] missing: {panel_path}")
    ev = pd.read_csv(t05_path)
    panel = pd.read_csv(panel_path)
    return ev, panel


def normalize_scores(panel: pd.DataFrame) -> pd.DataFrame:
    req = {"celebrity_name", "season", "week", "judge_score_total", "judge_score_count"}
    miss = req - set(panel.columns)
    if miss:
        raise ValueError(f"[T07] panel missing columns: {sorted(miss)}")

    df = panel.copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["contestant_key"] = df.apply(lambda r: contestant_key(r["season"], r["celebrity_name"]), axis=1)

    # active proxy in panel: has >=1 judge score and total notna
    df["active_proxy"] = (df["judge_score_count"].fillna(0).astype(int) >= 1) & df["judge_score_total"].notna()

    # z-score by season (only active_proxy rows)
    df["score_z"] = np.nan
    for season, g in df[df["active_proxy"]].groupby("season", sort=True):
        mu = float(g["judge_score_total"].mean())
        sd = float(g["judge_score_total"].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        idx = g.index
        df.loc[idx, "score_z"] = (g["judge_score_total"] - mu) / sd
    return df


def build_model_input(ev: pd.DataFrame, panel_norm: pd.DataFrame) -> Tuple[ModelInput, Dict[str, Any]]:
    # required columns in event table
    req = {"season", "week", "active_ids", "eliminated_indices", "eliminated_ids", "n_active", "elim_count"}
    miss = req - set(ev.columns)
    if miss:
        raise ValueError(f"[T07] T05 missing columns: {sorted(miss)}")

    # parse list cells
    ev = ev.copy()
    ev["active_ids"] = ev["active_ids"].apply(_parse_list_cell)
    ev["eliminated_ids"] = ev["eliminated_ids"].apply(_parse_list_cell)
    ev["eliminated_indices"] = ev["eliminated_indices"].apply(_parse_list_cell)

    # build event_ids
    ev["season"] = ev["season"].astype(int)
    ev["week"] = ev["week"].astype(int)
    ev["event_id"] = ev.apply(lambda r: f"{int(r['season'])}-{int(r['week'])}", axis=1)

    # compute dimensions
    event_ids = ev["event_id"].tolist()
    n_events = len(ev)
    max_active = int(max(len(x) for x in ev["active_ids"])) if n_events > 0 else 0
    max_elim = int(max(len(x) for x in ev["eliminated_indices"])) if n_events > 0 else 0
    max_elim = max(max_elim, 1)  # at least 1 for arrays

    # contestant universe from active_ids
    all_keys = sorted({k for lst in ev["active_ids"] for k in lst})
    key2idx = {k: i for i, k in enumerate(all_keys)}
    n_contestants = len(all_keys)

    # mapping from (season, week, key) -> score_z
    panel_norm = panel_norm.copy()
    panel_norm["season"] = panel_norm["season"].astype(int)
    panel_norm["week"] = panel_norm["week"].astype(int)
    score_map = {
        (int(r.season), int(r.week), str(r.contestant_key)): float(r.score_z)
        for r in panel_norm.dropna(subset=["score_z"]).itertuples(index=False)
    }

    # allocate arrays
    A = max_active
    K = max_elim
    active_key_idx = np.zeros((n_events, A), dtype=np.int32)
    score_z = np.zeros((n_events, A), dtype=np.float32)
    active_mask = np.zeros((n_events, A), dtype=bool)
    elim_idx = np.zeros((n_events, K), dtype=np.int32)
    elim_mask = np.zeros((n_events, K), dtype=bool)

    season_of_event = ev["season"].to_numpy(dtype=np.int32)
    week_of_event = ev["week"].to_numpy(dtype=np.int32)

    anomalies = {
        "dup_event_ids": int(ev["event_id"].duplicated().sum()),
        "elim_not_subset_active": 0,
        "elim_indices_mismatch": 0,
        "elim_count_mismatch": 0,
        "missing_scores": 0,
        "missing_scores_samples": [],
        "max_elim_count": 0,
    }

    for e, row in enumerate(ev.itertuples(index=False)):
        act: List[str] = list(row.active_ids)
        elim_ids: List[str] = list(row.eliminated_ids)
        elim_inds: List[int] = list(row.eliminated_indices)

        # enforce sorted active_ids for stability
        act_sorted = sorted(act)
        if act != act_sorted:
            act = act_sorted  # canonicalize

        # basic
        nA = len(act)
        active_mask[e, :nA] = True
        active_key_idx[e, :nA] = np.array([key2idx[k] for k in act], dtype=np.int32)

        # scores
        missing_local = []
        for j, k in enumerate(act):
            val = score_map.get((int(row.season), int(row.week), k), None)
            if val is None or (isinstance(val, float) and (not np.isfinite(val))):
                missing_local.append(k)
                val = 0.0
            score_z[e, j] = np.float32(val)

        if missing_local:
            anomalies["missing_scores"] += 1
            if len(anomalies["missing_scores_samples"]) < 10:
                anomalies["missing_scores_samples"].append(
                    {"event_id": str(row.event_id), "missing_keys": missing_local[:5]}
                )

        # elimination subset check
        if not set(elim_ids).issubset(set(act)):
            anomalies["elim_not_subset_active"] += 1

        # elim_count check
        if int(row.elim_count) != len(elim_inds) or int(row.elim_count) != len(elim_ids):
            anomalies["elim_count_mismatch"] += 1

        # indices check
        if any((ix < 0 or ix >= nA) for ix in elim_inds):
            anomalies["elim_indices_mismatch"] += 1

        anomalies["max_elim_count"] = max(anomalies["max_elim_count"], len(elim_inds))

        # pad elimination arrays
        m = len(elim_inds)
        if m > 0:
            elim_idx[e, :m] = np.array(elim_inds, dtype=np.int32)
            elim_mask[e, :m] = True
        # if m==0 => keep defaults (idx=0, mask=False)

    mi = ModelInput(
        n_events=n_events,
        n_contestants=n_contestants,
        max_active=max_active,
        max_elim=max_elim,
        active_key_idx=active_key_idx,
        score_z=score_z,
        active_mask=active_mask,
        elim_idx=elim_idx,
        elim_mask=elim_mask,
        event_ids=event_ids,
        contestant_keys=all_keys,
        season_of_event=season_of_event,
        week_of_event=week_of_event,
    )

    dims = {
        "n_events": n_events,
        "n_contestants": n_contestants,
        "max_active": max_active,
        "max_elim": max_elim,
    }
    return mi, {"dimensions": dims, "anomalies": anomalies}


def build_pymc_model(data: ModelInput) -> pm.Model:
    E, A, K = data.n_events, data.max_active, data.max_elim

    with pm.Model() as model:
        # data containers (NO mutable kwarg: version-safe)
        active_key_idx = pm.Data("active_key_idx", data.active_key_idx.astype("int32"))
        score_z = pm.Data("score_z", data.score_z.astype("float32"))
        active_mask = pm.Data("active_mask", data.active_mask.astype("int8"))  # 0/1 for pytensor ops
        elim_idx = pm.Data("elim_idx", data.elim_idx.astype("int32"))
        elim_mask = pm.Data("elim_mask", data.elim_mask.astype("int8"))

        # priors
        beta0 = pm.Normal("beta0", mu=0.0, sigma=1.0)
        beta1 = pm.Normal("beta1", mu=0.0, sigma=1.0)

        sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
        sigma_w = pm.HalfNormal("sigma_w", sigma=1.0)

        u_raw = pm.Normal("u_raw", mu=0.0, sigma=1.0, shape=(data.n_contestants,))
        w_raw = pm.Normal("w_raw", mu=0.0, sigma=1.0, shape=(data.n_events,))

        # sum-to-zero constraints for identifiability (softmax is shift-invariant)
        u = pm.Deterministic("u", sigma_u * (u_raw - pt.mean(u_raw)))
        w = pm.Deterministic("w", sigma_w * (w_raw - pt.mean(w_raw)))

        # utilities for each (event, slot)
        slot_u = u[active_key_idx]  # (E,A)
        logV = beta0 + beta1 * score_z + slot_u + w[:, None]  # (E,A)

        # Multi-elimination likelihood: sequential draws WITHOUT replacement on losers
        # Probability proportional to exp(-logV) among remaining active slots.
        big_neg = np.float32(-1e9)

        # selected mask: (E,A) bool/int8
        selected = pt.zeros((E, A), dtype="int8")
        total_logp = 0.0

        arangeE = pt.arange(E)
        arangeA = pt.arange(A)[None, :]

        for r in range(K):
            mr = elim_mask[:, r].astype("int8")  # (E,)
            idx_r = elim_idx[:, r]  # (E,)

            # safe index to avoid -1 gather (if mr=0 we won't use it)
            idx_safe = pt.where(mr, idx_r, 0)

            # remaining = active_mask & (~selected)
            remaining = pt.and_(
                pt.gt(active_mask, 0),
                pt.eq(selected, 0)
            )

            logits = -logV
            logits_masked = pt.where(remaining, logits, big_neg)  # (E,A)

            denom = logsumexp(logits_masked, axis=1)  # (E,)
            chosen = logits_masked[arangeE, idx_safe]  # (E,)
            step_logp = chosen - denom  # (E,)

            total_logp = total_logp + pt.sum(pt.where(mr, step_logp, 0.0))

            # update selected mask where mr=1: mark idx_r as selected
            onehot = pt.eq(arangeA, idx_safe[:, None]).astype("int8")  # (E,A)
            selected = pt.where(mr[:, None], pt.clip(selected + onehot, 0, 1), selected)

        pm.Potential("elim_like", total_logp)

    return model


def posterior_mean_consistency(idata, data: ModelInput, out_path: Path) -> Dict[str, Any]:
    # Minimal: use posterior means of beta0,beta1,u,w to compute mean-logV consistency
    post = idata.posterior
    beta0 = float(post["beta0"].mean().values)
    beta1 = float(post["beta1"].mean().values)
    u = post["u"].mean(dim=("chain", "draw")).values  # (C,)
    w = post["w"].mean(dim=("chain", "draw")).values  # (E,)

    E, A = data.n_events, data.max_active
    logV = beta0 + beta1 * data.score_z + u[data.active_key_idx] + w[:, None]
    active_mask = data.active_mask
    elim_mask = data.elim_mask
    elim_idx = data.elim_idx

    # hard check: for each event, are eliminated indices among the bottom-k of logV?
    hard_hits = []
    for e in range(E):
        k = int(elim_mask[e].sum())
        if k == 0:
            continue
        nA = int(active_mask[e].sum())
        vals = logV[e, :nA]
        bottom_k = set(np.argsort(vals)[:k].tolist())
        elim_set = set(elim_idx[e, :k].tolist())
        hard_hits.append(int(elim_set.issubset(bottom_k)))

    summary = {
        "n_events_with_elim": int(len(hard_hits)),
        "hard_consistency_rate_mean_logV": float(np.mean(hard_hits)) if hard_hits else None,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_t05", default="data/T05_model_ready_events_route2.csv")
    ap.add_argument("--input_panel", default="data/T02_long_format_panel.csv")
    ap.add_argument("--out_report", default="data/T07_input_consistency_report.json")
    ap.add_argument("--out_trace", default="data/T07_trace_route2.nc")
    ap.add_argument("--out_mean_consistency", default="data/T07_posterior_mean_consistency.json")

    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20260130)

    args = ap.parse_args()

    t05_path = Path(args.input_t05)
    panel_path = Path(args.input_panel)

    print("[T07] Loading inputs...")
    ev, panel = load_inputs(t05_path, panel_path)

    print("[T07] Normalizing judge scores (z-score per season)...")
    panel_norm = normalize_scores(panel)

    print("[T07] Building model arrays + consistency checks...")
    mi, report = build_model_input(ev, panel_norm)

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps(
            {
                "generated_at": pd.Timestamp.utcnow().isoformat(),
                "inputs": {"INPUT_T05": str(t05_path), "INPUT_PANEL": str(panel_path)},
                "dimensions": report["dimensions"],
                "anomalies": report["anomalies"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[T07] Consistency report saved: {out_report}")
    print(json.dumps({"dimensions": report["dimensions"], "anomalies": report["anomalies"]}, ensure_ascii=False, indent=2))

    if report["anomalies"]["missing_scores"] > 0:
        raise RuntimeError("[T07] Missing scores detected. Fix panel/event alignment before sampling.")

    print("[T07] Building model...")
    model = build_pymc_model(mi)

    print("[T07] Sampling...")
    with model:
        # Compatibility: some versions default to InferenceData already
        try:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                target_accept=args.target_accept,
                random_seed=args.seed,
                return_inferencedata=True,
            )
        except TypeError:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                target_accept=args.target_accept,
                random_seed=args.seed,
            )

    out_trace = Path(args.out_trace)
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    # InferenceData has to_netcdf; MultiTrace doesn't. Guard it.
    if hasattr(idata, "to_netcdf"):
        idata.to_netcdf(out_trace)
        print(f"[T07] Trace saved: {out_trace}")
    else:
        print("[T07] WARNING: idata has no to_netcdf(); you are likely on an older backend returning MultiTrace.")

    # quick posterior mean consistency (optional)
    if hasattr(idata, "posterior"):
        out_mean = Path(args.out_mean_consistency)
        out_mean.parent.mkdir(parents=True, exist_ok=True)
        s = posterior_mean_consistency(idata, mi, out_mean)
        print(f"[T07] Posterior-mean hard consistency saved: {out_mean}")
        print(json.dumps(s, ensure_ascii=False, indent=2))

    print("[T07] Done.")


if __name__ == "__main__":
    main()
