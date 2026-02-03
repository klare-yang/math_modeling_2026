#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t09_ppc_replay_2stage_route2_v3B.py

Task point 13 (renamed from T19): posterior predictive replay / match probability.

Implements Scheme A (analytic, likelihood-consistent):
- One-stage weeks (elim_count=k<=3):
    P(E_t = observed_set) under unordered set probability with weights
      w_i ∝ exp(-kappa_S * S_i)
- Two-stage judges-save weeks (use_two_stage=1, elim_count=1):
    P(elim=e) = Σ_{j≠e} P({e,j} as bottom-2) * P(elim=e | {e,j})
    with
      P({e,j}) = (w_e*w_j/W) * (1/(W-w_e) + 1/(W-w_j))
      judge weights u_i ∝ exp(-kappa_J * softRank(J_total)_i)
      P(elim=e | {e,j}) = u_e / (u_e + u_j)

Inputs:
  - --trace: data/T07f_trace_route2_v3B.nc
  - --t05_path: data/T05_model_ready_events_route2_v3B.csv
  - --mask_path: data/T04_judges_save_event_mask_route2_v3B.csv

Outputs:
  - --out_csv: data/T19_ppc_replay_summary_route2_v3B.csv (kept for downstream compatibility)
  - --out_csv_alias: data/T09_ppc_replay_summary_route2_v3B.csv (alias)

Acceptance goal:
- Replay probability uses the same formulas as the T07f likelihood.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

try:
    import arviz as az
except Exception as e:
    raise RuntimeError("arviz is required. Install: pip install arviz") from e


EPS = 1e-30


def _parse_json_list(x: Any) -> List[Any]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, np.ndarray)):
        return list(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    try:
        return json.loads(s)
    except Exception:
        s2 = s.replace("'", '"')
        return json.loads(s2)


def soft_rank_np(x: np.ndarray, mask: np.ndarray, tau: float = 0.15) -> np.ndarray:
    """Numpy replica of soft_rank_pt (v3B). Inactive slots return 0."""
    x = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=float)
    A = x.shape[0]
    diff = (x[None, :] - x[:, None]) / float(tau)  # x_j - x_i
    sig = 1.0 / (1.0 + np.exp(-diff))
    pair_mask = (m[:, None] * m[None, :]) * (1.0 - np.eye(A, dtype=float))
    r = 1.0 + np.sum(sig * pair_mask, axis=1)
    return r * m


def unordered_pair_prob(wi: np.ndarray, wj: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Same unordered pair probability as T07f."""
    wi = np.maximum(wi, EPS)
    wj = np.maximum(wj, EPS)
    W = np.maximum(W, EPS)
    term = (wi * wj / W) * (1.0 / np.maximum(W - wi, EPS) + 1.0 / np.maximum(W - wj, EPS))
    return np.maximum(term, 0.0)


def unordered_k_set_prob(weights: np.ndarray, idxs: List[int]) -> np.ndarray:
    """weights: (samples, A), idxs length k in {1,2,3}. returns p: (samples,)"""
    k = len(idxs)
    w = np.maximum(weights, 0.0)
    W = np.maximum(np.sum(w, axis=1), EPS)

    if k == 1:
        wi = np.maximum(w[:, idxs[0]], EPS)
        return wi / W

    if k == 2:
        wi = w[:, idxs[0]]
        wj = w[:, idxs[1]]
        return unordered_pair_prob(wi, wj, W)

    if k == 3:
        a, b, c = idxs
        wa = np.maximum(w[:, a], EPS)
        wb = np.maximum(w[:, b], EPS)
        wc = np.maximum(w[:, c], EPS)

        def order_prob(first, second, third):
            denom1 = np.maximum(W - first, EPS)
            denom2 = np.maximum(W - first - second, EPS)
            return (first / W) * (second / denom1) * (third / denom2)

        p = (
            order_prob(wa, wb, wc)
            + order_prob(wa, wc, wb)
            + order_prob(wb, wa, wc)
            + order_prob(wb, wc, wa)
            + order_prob(wc, wa, wb)
            + order_prob(wc, wb, wa)
        )
        return np.maximum(p, 0.0)

    raise ValueError("unordered_k_set_prob supports k in {1,2,3}")


def _stack_samples(da):
    if "chain" in da.dims and "draw" in da.dims:
        return da.stack(sample=("chain", "draw"))
    if "sample" in da.dims:
        return da
    return da


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="data/T07f_trace_route2_v3B.nc")
    ap.add_argument("--t05_path", default="data/T05_model_ready_events_route2_v3B.csv")
    ap.add_argument("--mask_path", default="data/T04_judges_save_event_mask_route2_v3B.csv")
    ap.add_argument("--out_csv", default="data/T19_ppc_replay_summary_route2_v3B.csv")
    ap.add_argument("--out_csv_alias", default="data/T09_ppc_replay_summary_route2_v3B.csv")
    ap.add_argument("--tau_rank", type=float, default=0.15)
    ap.add_argument("--max_draws", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260202)
    args = ap.parse_args()

    trace_path = Path(args.trace)
    t05_path = Path(args.t05_path)
    mask_path = Path(args.mask_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"[t09] trace not found: {trace_path}")
    if not t05_path.exists():
        raise FileNotFoundError(f"[t09] T05 not found: {t05_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"[t09] mask not found: {mask_path}")

    t05 = pd.read_csv(t05_path)
    if "event_id" not in t05.columns:
        raise ValueError("[t09] T05 must include event_id.")
    n_events = len(t05)

    mask_df = pd.read_csv(mask_path)
    if "event_id" not in mask_df.columns:
        raise ValueError("[t09] mask must include event_id.")
    for col in ["use_two_stage", "fallback_one_stage"]:
        if col not in mask_df.columns:
            raise ValueError(f"[t09] mask missing required column: {col}")

    mm = t05[["event_id"]].merge(
        mask_df[["event_id", "use_two_stage", "fallback_one_stage"]],
        on="event_id",
        how="left",
    )
    use_two_stage = (
        (mm["use_two_stage"].fillna(0).astype(int).to_numpy() == 1)
        & (mm["fallback_one_stage"].fillna(0).astype(int).to_numpy() == 0)
    ).astype(np.int32)

    # active ids (slot alignment)
    active_ids_col = None
    for c in ["active_ids", "active_ids_by_event", "active_ids_json"]:
        if c in t05.columns:
            active_ids_col = c
            break
    if active_ids_col is None:
        raise ValueError("[t09] T05 missing active_ids (JSON list aligned to slots).")
    active_ids = [_parse_json_list(v) for v in t05[active_ids_col].tolist()]

    # eliminated slots (preferred)
    elim_idx_col = None
    for c in ["eliminated_indices", "elim_indices", "eliminated_slots"]:
        if c in t05.columns:
            elim_idx_col = c
            break
    elim_ids_col = None
    for c in ["eliminated_ids", "elim_ids"]:
        if c in t05.columns:
            elim_ids_col = c
            break

    elim_slots: List[List[int]] = []
    elim_count = np.zeros((n_events,), dtype=np.int32)
    for e in range(n_events):
        if elim_idx_col is not None:
            idxs = _parse_json_list(t05.loc[e, elim_idx_col])
            idxs = [int(i) for i in idxs]
        elif elim_ids_col is not None:
            eids = _parse_json_list(t05.loc[e, elim_ids_col])
            aids = active_ids[e]
            aid_to_slot = {aids[s]: s for s in range(len(aids))}
            idxs = [int(aid_to_slot[cid]) for cid in eids if cid in aid_to_slot]
        else:
            raise ValueError("[t09] T05 must include eliminated_indices or eliminated_ids.")
        idxs = [i for i in idxs if i is not None]
        elim_slots.append(idxs)
        elim_count[e] = int(len(idxs))

    # judge totals for rankJ
    jtotal_col = None
    for c in ["J_total_by_active", "J_total_active", "judge_total_by_active", "J_total"]:
        if c in t05.columns:
            jtotal_col = c
            break
    if jtotal_col is None:
        raise ValueError(
            "[t09] T05 missing J_total_by_active (JSON list aligned to slots). Required for two-stage judge weights."
        )
    J_total_by_active = [_parse_json_list(v) for v in t05[jtotal_col].tolist()]

    # Load trace
    idata = az.from_netcdf(trace_path)
    post = idata.posterior
    if "S" not in post:
        raise ValueError("[t09] trace missing posterior variable 'S'. Use T07f/T07g.")
    if "kappa_S" not in post:
        raise ValueError("[t09] trace missing kappa_S.")
    if "kappa_J" not in post and int(use_two_stage.sum()) > 0:
        raise ValueError("[t09] trace missing kappa_J but use_two_stage events exist.")

    S_da = _stack_samples(post["S"])
    kS_da = _stack_samples(post["kappa_S"])
    kJ_da = _stack_samples(post["kappa_J"]) if "kappa_J" in post else None

    if S_da.ndim != 3:
        raise ValueError(f"[t09] Unexpected S dims={S_da.dims} shape={tuple(S_da.shape)}")
    dims = list(S_da.dims)
    if "sample" not in dims:
        raise ValueError("[t09] Failed to stack S to include 'sample' dim.")
    other = [d for d in dims if d != "sample"]
    if len(other) != 2:
        raise ValueError(f"[t09] Unexpected S dims={dims}")
    if S_da.sizes[other[0]] == n_events:
        event_dim, slot_dim = other[0], other[1]
    elif S_da.sizes[other[1]] == n_events:
        event_dim, slot_dim = other[1], other[0]
    else:
        raise ValueError(f"[t09] Could not match S event dim to T05 length={n_events}. S sizes={S_da.sizes}")
    S_da = S_da.transpose("sample", event_dim, slot_dim)

    # subsample
    rng = np.random.default_rng(args.seed)
    n_samp = int(S_da.sizes["sample"])
    if args.max_draws and args.max_draws > 0 and args.max_draws < n_samp:
        idx = rng.choice(n_samp, size=args.max_draws, replace=False)
        S_da = S_da.isel(sample=idx)
        kS_da = kS_da.isel(sample=idx)
        if kJ_da is not None:
            kJ_da = kJ_da.isel(sample=idx)
        n_samp = int(S_da.sizes["sample"])

    S = np.asarray(S_da.values, dtype=float)  # (sample,E,A)
    kappa_S = np.asarray(kS_da.values, dtype=float).reshape(-1)
    kappa_J = np.asarray(kJ_da.values, dtype=float).reshape(-1) if kJ_da is not None else None

    E = n_events
    A = S.shape[2]

    # active mask
    active_mask = np.zeros((E, A), dtype=float)
    for e in range(E):
        aids = active_ids[e]
        if len(aids) < A:
            aids = aids + [None] * (A - len(aids))
        else:
            aids = aids[:A]
        for s in range(A):
            if aids[s] is not None:
                active_mask[e, s] = 1.0

    # rankJ (E,A)
    rankJ = np.zeros((E, A), dtype=float)
    for e in range(E):
        jt = J_total_by_active[e]
        if len(jt) < A:
            jt = jt + [0.0] * (A - len(jt))
        else:
            jt = jt[:A]
        rankJ[e, :] = soft_rank_np(np.asarray(jt, dtype=float), active_mask[e, :], tau=float(args.tau_rank))

    # stage1 weights
    weights = np.exp(-kappa_S[:, None, None] * S) * active_mask[None, :, :]

    # judge weights
    judge_weights = None
    if kappa_J is not None:
        judge_weights = np.exp(-kappa_J[:, None, None] * rankJ[None, :, :]) * active_mask[None, :, :]

    p = np.ones((n_samp, E), dtype=float)

    for e in range(E):
        k = int(elim_count[e])
        if k == 0:
            p[:, e] = 1.0
            continue

        idxs = [int(i) for i in elim_slots[e] if i is not None and 0 <= int(i) < A]
        if len(idxs) == 0:
            p[:, e] = 1.0
            continue

        w_e = weights[:, e, :]  # (sample,A)
        W = np.maximum(np.sum(w_e, axis=1), EPS)

        if use_two_stage[e] == 1 and k == 1:
            if judge_weights is None:
                raise RuntimeError("[t09] judge_weights unavailable for two-stage events.")
            i = idxs[0]
            wi = w_e[:, i]
            wj = w_e
            term = (wi[:, None] * wj / W[:, None]) * (
                1.0 / np.maximum(W[:, None] - wi[:, None], EPS) + 1.0 / np.maximum(W[:, None] - wj, EPS)
            )
            term[:, i] = 0.0
            term *= active_mask[e, :][None, :]

            jw_e = judge_weights[:, e, :]
            jw_i = jw_e[:, i]
            p_elim_i = jw_i[:, None] / np.maximum(jw_i[:, None] + jw_e, EPS)
            p_elim_i[:, i] = 0.0
            p_elim_i *= active_mask[e, :][None, :]

            p[:, e] = np.maximum(np.sum(term * p_elim_i, axis=1), 0.0)
        else:
            if len(idxs) > 3:
                p[:, e] = 0.0
            else:
                p[:, e] = unordered_k_set_prob(w_e, idxs)

    p = np.clip(p, EPS, 1.0)
    logp = np.log(p)

    out = pd.DataFrame(
        {
            "event_id": t05["event_id"].tolist(),
            "season": t05["season"].astype(int).tolist() if "season" in t05.columns else [None] * E,
            "week": t05["week"].astype(int).tolist() if "week" in t05.columns else [None] * E,
            "elim_count": elim_count.astype(int).tolist(),
            "use_two_stage": use_two_stage.astype(int).tolist(),
            "ppc_p_mean": p.mean(axis=0).tolist(),
            "ppc_p_q025": np.quantile(p, 0.025, axis=0).tolist(),
            "ppc_p_q975": np.quantile(p, 0.975, axis=0).tolist(),
            "ppc_logp_mean": logp.mean(axis=0).tolist(),
            "ppc_logp_q025": np.quantile(logp, 0.025, axis=0).tolist(),
            "ppc_logp_q975": np.quantile(logp, 0.975, axis=0).tolist(),
        }
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    out_alias = Path(args.out_csv_alias)
    out_alias.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_csv, out_alias)

    print("=" * 90)
    print("[t09] Saved:")
    print(f"  {out_csv}")
    print(f"  {out_alias}")
    print(f"[t09] n_events={E}, n_samples_used={n_samp}, two_stage_events={int(use_two_stage.sum())}")
    print("=" * 90)


if __name__ == "__main__":
    main()
