#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T07c (route2_v3B): Latent vote share (pi) inside the model, using baseline one-stage elimination likelihood.

This task point ONLY introduces:
  - latent theta -> pi (logistic-normal) inside the PyMC model.

It intentionally DOES NOT yet:
  - use S(J, pi) mechanism score,
  - use unordered set likelihood,
  - implement two-stage judges-save.

Compatibility note
------------------
Some PyTensor versions do not expose `pt.nnet.softmax`. This script implements a small
masked-softmax utility using core PyTensor ops.

Inputs
------
- data/T05_model_ready_events_route2_v3B.csv  (from T03)
  Required columns:
    event_id, season, week, terminal, n_active, elim_count,
    active_ids (json list), eliminated_indices (json list)

Outputs
-------
- data/T07c_trace_route2_v3B.nc
- data/T07c_input_consistency_report_route2_v3B.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import pymc as pm
import pytensor.tensor as pt
import arviz as az


# ----------------------------
# Utilities
# ----------------------------

def jload_list(s: Any) -> List[Any]:
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    return json.loads(s)


def masked_softmax(theta_2d, mask_2d, axis: int = 1, eps: float = 1e-30):
    """
    Masked softmax implemented with core PyTensor ops.

    Parameters
    ----------
    theta_2d : pt tensor, shape (n_events, max_active)
    mask_2d : pt tensor, same shape, float {0,1} or bool
    axis : int
        Softmax axis (default 1, over active slots).
    eps : float
        Denominator clamp.

    Returns
    -------
    pt tensor, same shape, rows sum to 1 over active positions; inactive positions = 0.
    """
    m = pt.cast(mask_2d, "float64")
    big_neg = -1.0e9
    t_masked = pt.switch(m > 0.5, theta_2d, big_neg)

    # subtract max for stability (max over masked values)
    t_shift = t_masked - pt.max(t_masked, axis=axis, keepdims=True)
    ex = pt.exp(t_shift) * m
    denom = pt.sum(ex, axis=axis, keepdims=True)
    return ex / pt.maximum(denom, eps)


@dataclass
class ModelInput:
    n_events: int
    n_contestants: int
    max_active: int
    max_elim: int
    active_idx: np.ndarray        # (n_events, max_active) int32, -1 for inactive
    active_mask: np.ndarray       # (n_events, max_active) bool
    elim_slots: np.ndarray        # (n_events, max_elim) int32, slot index into active positions; -1 for none
    elim_mask: np.ndarray         # (n_events, max_elim) bool
    elim_count: np.ndarray        # (n_events,) int32
    event_id: List[str]
    season: np.ndarray
    week: np.ndarray


def build_model_input(t05: pd.DataFrame) -> ModelInput:
    req = {"event_id", "season", "week", "terminal", "n_active", "elim_count", "active_ids", "eliminated_indices"}
    miss = req - set(t05.columns)
    if miss:
        raise ValueError(f"[T07c] T05 missing columns: {sorted(miss)}")

    t05 = t05.sort_values(["season", "week"]).reset_index(drop=True).copy()
    n_events = len(t05)

    active_lists: List[List[str]] = [list(map(str, jload_list(s))) for s in t05["active_ids"].tolist()]
    elim_idx_lists: List[List[int]] = [list(map(int, jload_list(s))) for s in t05["eliminated_indices"].tolist()]

    max_active = int(max((len(x) for x in active_lists), default=0))
    max_elim = int(max((len(x) for x in elim_idx_lists), default=0))
    if max_elim < 1:
        max_elim = 1
    if max_elim > 3:
        print(f"[T07c][WARN] max_elim={max_elim} > 3, clipping to 3")
        max_elim = 3

    all_ids = sorted({cid for ids in active_lists for cid in ids})
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    n_contestants = len(all_ids)

    active_idx = np.full((n_events, max_active), -1, dtype=np.int32)
    active_mask = np.zeros((n_events, max_active), dtype=bool)

    for e, ids in enumerate(active_lists):
        for j, cid in enumerate(ids):
            active_idx[e, j] = id_to_idx[cid]
            active_mask[e, j] = True

    elim_slots = np.full((n_events, max_elim), -1, dtype=np.int32)
    elim_mask = np.zeros((n_events, max_elim), dtype=bool)
    elim_count = np.zeros((n_events,), dtype=np.int32)

    for e, elim_slots_list in enumerate(elim_idx_lists):
        k = len(elim_slots_list)
        elim_count[e] = k
        if k == 0:
            continue
        nA = len(active_lists[e])
        for s in elim_slots_list[:max_elim]:
            if not (0 <= int(s) < nA):
                raise ValueError(f"[T07c] event {t05.loc[e,'event_id']} invalid elim slot {s} for n_active={nA}")
        for r, s in enumerate(elim_slots_list[:max_elim]):
            elim_slots[e, r] = int(s)
            elim_mask[e, r] = True
        if len(set(elim_slots_list)) != len(elim_slots_list):
            raise ValueError(f"[T07c] event {t05.loc[e,'event_id']} duplicate eliminated_indices: {elim_slots_list}")

    return ModelInput(
        n_events=n_events,
        n_contestants=n_contestants,
        max_active=max_active,
        max_elim=max_elim,
        active_idx=active_idx,
        active_mask=active_mask,
        elim_slots=elim_slots,
        elim_mask=elim_mask,
        elim_count=elim_count,
        event_id=t05["event_id"].astype(str).tolist(),
        season=t05["season"].astype(int).to_numpy(),
        week=t05["week"].astype(int).to_numpy(),
    )


def save_report(mi: ModelInput, out_path: Path) -> None:
    report = {
        "dimensions": {
            "n_events": int(mi.n_events),
            "n_contestants": int(mi.n_contestants),
            "max_active": int(mi.max_active),
            "max_elim": int(mi.max_elim),
        },
        "sanity": {
            "pct_events_elim0": float(np.mean(mi.elim_count == 0)),
            "pct_events_elim1": float(np.mean(mi.elim_count == 1)),
            "pct_events_elim2": float(np.mean(mi.elim_count == 2)),
            "pct_events_elim3": float(np.mean(mi.elim_count == 3)),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t05_path", default="data/T05_model_ready_events_route2_v3B.csv")
    ap.add_argument("--out_trace", default="data/T07c_trace_route2_v3B.nc")
    ap.add_argument("--out_report", default="data/T07c_input_consistency_report_route2_v3B.json")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20260201)
    args = ap.parse_args()

    t05_path = Path(args.t05_path)
    if not t05_path.exists():
        raise FileNotFoundError(f"[T07c] T05 not found: {t05_path}")

    print("=" * 90)
    print("[T07c] Loading inputs...")
    print("=" * 90)
    t05 = pd.read_csv(t05_path)
    mi = build_model_input(t05)

    print(f"[T07c] t05_path = {t05_path}")
    print(f"[T07c] n_events={mi.n_events}, n_contestants={mi.n_contestants}, max_active={mi.max_active}, max_elim={mi.max_elim}")
    print(f"[T07c] elim_count distribution: "
          f"0:{np.sum(mi.elim_count==0)} "
          f"1:{np.sum(mi.elim_count==1)} "
          f"2:{np.sum(mi.elim_count==2)} "
          f"3:{np.sum(mi.elim_count==3)}")

    out_report = Path(args.out_report)
    save_report(mi, out_report)
    print(f"[T07c] Saved report: {out_report}")

    # tensors
    active_idx_t = pt.as_tensor_variable(mi.active_idx)
    active_mask_t = pt.as_tensor_variable(mi.active_mask.astype("float64"))
    elim_slots_t = pt.as_tensor_variable(mi.elim_slots)
    elim_mask_t = pt.as_tensor_variable(mi.elim_mask.astype("float64"))

    elim_slots_safe_t = pt.maximum(elim_slots_t, 0)
    ev_idx = pt.arange(mi.n_events)

    print("=" * 90)
    print("[T07c] Building PyMC model (latent theta -> pi)...")
    print("=" * 90)

    with pm.Model() as model:
        sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
        sigma_w = pm.HalfNormal("sigma_w", sigma=1.0)
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=1.0)
        kappa = pm.HalfNormal("kappa", sigma=1.0)

        u_raw = pm.Normal("u_raw", mu=0.0, sigma=1.0, shape=(mi.n_contestants,))
        w_raw = pm.Normal("w_raw", mu=0.0, sigma=1.0, shape=(mi.n_events,))

        u = pm.Deterministic("u", (u_raw - pt.mean(u_raw)) * sigma_u)
        w = pm.Deterministic("w", (w_raw - pt.mean(w_raw)) * sigma_w)

        idx_safe = pt.maximum(active_idx_t, 0)
        u_es = u[idx_safe] * active_mask_t

        eps = pm.Normal("eps", mu=0.0, sigma=sigma_eps, shape=(mi.n_events, mi.max_active))
        theta = pm.Deterministic("theta", u_es + w[:, None] + eps * active_mask_t)

        # pi: masked softmax over slots
        pi = pm.Deterministic("pi", masked_softmax(theta, active_mask_t, axis=1))

        # elimination weights: favor SMALL theta
        # IMPORTANT: use unmasked theta and apply mask after exp to avoid overflow.
        weights = pt.exp(-kappa * theta) * active_mask_t

        sum0 = pt.maximum(weights.sum(axis=1), 1e-30)

        slot1 = elim_slots_safe_t[:, 0]
        w1 = weights[ev_idx, slot1]
        p1 = w1 / sum0
        sum1 = pt.maximum(sum0 - w1, 1e-30)

        lp = elim_mask_t[:, 0] * pt.log(pt.maximum(p1, 1e-30))

        if mi.max_elim >= 2:
            slot2 = elim_slots_safe_t[:, 1]
            w2 = weights[ev_idx, slot2]
            p2 = w2 / sum1
            sum2 = pt.maximum(sum1 - w2, 1e-30)
            lp = lp + elim_mask_t[:, 1] * pt.log(pt.maximum(p2, 1e-30))
        else:
            sum2 = sum1

        if mi.max_elim >= 3:
            slot3 = elim_slots_safe_t[:, 2]
            w3 = weights[ev_idx, slot3]
            p3 = w3 / sum2
            lp = lp + elim_mask_t[:, 2] * pt.log(pt.maximum(p3, 1e-30))

        pm.Potential("elim_loglik", pt.sum(lp))

        print("[T07c] Sampling config:")
        print(f"  draws={args.draws}, tune={args.tune}, chains={args.chains}, cores={args.cores}, target_accept={args.target_accept}, seed={args.seed}")

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.seed,
            return_inferencedata=True,
        )

    out_trace = Path(args.out_trace)
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(out_trace)
    print(f"[T07c] Saved trace: {out_trace}")

    try:
        summ = az.summary(idata, var_names=["sigma_u", "sigma_w", "sigma_eps", "kappa"], kind="stats")
        print("[T07c] Posterior summary (key scales):")
        print(summ.to_string())
    except Exception as e:
        print(f"[T07c][WARN] az.summary failed: {e}")

    print("=" * 90)
    print("[T07c] Done.")
    print("=" * 90)
    print("[T07c] Interpretation hints:")
    print("  - pi is now an internal model quantity (vote share per event).")
    print("  - This is still a baseline elimination likelihood on theta; S(J,pi) comes in the next task point.")


if __name__ == "__main__":
    main()
