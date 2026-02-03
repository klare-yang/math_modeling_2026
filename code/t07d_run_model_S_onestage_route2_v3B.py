#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T07d (route2_v3B): Mechanism score S(J, pi) with ONE-STAGE elimination likelihood (still order-sensitive).

Task point scope
----------------
- Introduce mechanism-aligned composite score S(J, pi):
    * percent-rule: S = J_pct + pi
    * rank-rule:    S = softRank(J_total) + softRank(pi)
- Use S (instead of theta) to drive elimination: weights ∝ exp(-kappa_S * S)

Not included yet
----------------
- Unordered set likelihood for multi-elimination events (T07e).
- Two-stage judges-save likelihood + marginalization (T07f).

Inputs
------
- data/T05_model_ready_events_route2_v3B.csv
  Required columns:
    event_id, season, week, terminal, n_active, elim_count,
    rule_mode,
    active_ids (json list), eliminated_indices (json list),
    J_total_by_active (json list), J_pct_by_active (json list)

Outputs
-------
- data/T07d_trace_route2_v3B.nc
- data/T07d_input_consistency_report_route2_v3B.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az

from t01_route2_v3B import soft_rank_pt


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
    m = pt.cast(mask_2d, "float64")
    big_neg = -1.0e9
    t_masked = pt.switch(m > 0.5, theta_2d, big_neg)
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
    active_idx: np.ndarray
    active_mask: np.ndarray
    elim_slots: np.ndarray
    elim_mask: np.ndarray
    elim_count: np.ndarray
    rule_is_rank: np.ndarray
    J_total: np.ndarray
    J_pct: np.ndarray
    event_id: List[str]
    season: np.ndarray
    week: np.ndarray


def build_model_input(t05: pd.DataFrame) -> ModelInput:
    req = {
        "event_id", "season", "week", "terminal", "n_active", "elim_count",
        "rule_mode", "active_ids", "eliminated_indices", "J_total_by_active", "J_pct_by_active",
    }
    miss = req - set(t05.columns)
    if miss:
        raise ValueError(f"[T07d] T05 missing columns: {sorted(miss)}")

    t05 = t05.sort_values(["season", "week"]).reset_index(drop=True).copy()
    E = len(t05)

    active_lists = [list(map(str, jload_list(s))) for s in t05["active_ids"].tolist()]
    elim_lists = [list(map(int, jload_list(s))) for s in t05["eliminated_indices"].tolist()]
    J_total_lists = [list(map(float, jload_list(s))) for s in t05["J_total_by_active"].tolist()]
    J_pct_lists = [list(map(float, jload_list(s))) for s in t05["J_pct_by_active"].tolist()]

    max_active = int(max((len(x) for x in active_lists), default=0))
    max_elim = int(max((len(x) for x in elim_lists), default=1))
    max_elim = max(1, min(max_elim, 3))

    all_ids = sorted({cid for ids in active_lists for cid in ids})
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    C = len(all_ids)

    active_idx = np.full((E, max_active), -1, dtype=np.int32)
    active_mask = np.zeros((E, max_active), dtype=bool)
    J_total = np.zeros((E, max_active), dtype=np.float64)
    J_pct = np.zeros((E, max_active), dtype=np.float64)

    for e in range(E):
        ids = active_lists[e]
        jt = J_total_lists[e]
        jp = J_pct_lists[e]
        if not (len(ids) == len(jt) == len(jp)):
            raise ValueError(
                f"[T07d] event {t05.loc[e,'event_id']} length mismatch: ids={len(ids)} jt={len(jt)} jp={len(jp)}"
            )
        for j, cid in enumerate(ids):
            active_idx[e, j] = id_to_idx[cid]
            active_mask[e, j] = True
            J_total[e, j] = float(jt[j])
            J_pct[e, j] = float(jp[j])

    elim_slots = np.full((E, max_elim), -1, dtype=np.int32)
    elim_mask = np.zeros((E, max_elim), dtype=bool)
    elim_count = np.zeros((E,), dtype=np.int32)

    for e, slots in enumerate(elim_lists):
        k = len(slots)
        elim_count[e] = k
        if k == 0:
            continue
        nA = len(active_lists[e])
        for s in slots[:max_elim]:
            if not (0 <= int(s) < nA):
                raise ValueError(f"[T07d] event {t05.loc[e,'event_id']} invalid elim slot {s} for n_active={nA}")
        if len(set(slots)) != len(slots):
            raise ValueError(f"[T07d] event {t05.loc[e,'event_id']} duplicate eliminated_indices: {slots}")
        for r, s in enumerate(slots[:max_elim]):
            elim_slots[e, r] = int(s)
            elim_mask[e, r] = True

    rule_is_rank = (t05["rule_mode"].astype(str).str.lower() == "rank").to_numpy()

    return ModelInput(
        n_events=E,
        n_contestants=C,
        max_active=max_active,
        max_elim=max_elim,
        active_idx=active_idx,
        active_mask=active_mask,
        elim_slots=elim_slots,
        elim_mask=elim_mask,
        elim_count=elim_count,
        rule_is_rank=rule_is_rank,
        J_total=J_total,
        J_pct=J_pct,
        event_id=t05["event_id"].astype(str).tolist(),
        season=t05["season"].astype(int).to_numpy(),
        week=t05["week"].astype(int).to_numpy(),
    )


def save_report(mi: ModelInput, out_path: Path) -> None:
    rep = {
        "dimensions": {
            "n_events": int(mi.n_events),
            "n_contestants": int(mi.n_contestants),
            "max_active": int(mi.max_active),
            "max_elim": int(mi.max_elim),
        },
        "events": {
            "n_rank": int(np.sum(mi.rule_is_rank)),
            "n_percent": int(np.sum(~mi.rule_is_rank)),
        },
        "elim_count": {
            "n0": int(np.sum(mi.elim_count == 0)),
            "n1": int(np.sum(mi.elim_count == 1)),
            "n2": int(np.sum(mi.elim_count == 2)),
            "n3": int(np.sum(mi.elim_count == 3)),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")


def soft_rank_batch(x_2d, mask_2d, tau: float):
    def _row_fn(x_row, m_row):
        return soft_rank_pt(x_row, tau=tau, mask=m_row)

    res = pytensor.scan(fn=_row_fn, sequences=[x_2d, mask_2d], return_updates=False)
    return res[0] if isinstance(res, (tuple, list)) else res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t05_path", default="data/T05_model_ready_events_route2_v3B.csv")
    ap.add_argument("--out_trace", default="data/T07d_trace_route2_v3B.nc")
    ap.add_argument("--out_report", default="data/T07d_input_consistency_report_route2_v3B.json")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.97)
    ap.add_argument("--seed", type=int, default=20260201)
    ap.add_argument("--tau_rank", type=float, default=0.15)
    args = ap.parse_args()

    t05_path = Path(args.t05_path)
    if not t05_path.exists():
        raise FileNotFoundError(f"[T07d] T05 not found: {t05_path}")

    print("=" * 90)
    print("[T07d] Loading inputs...")
    print("=" * 90)
    t05 = pd.read_csv(t05_path)
    mi = build_model_input(t05)
    save_report(mi, Path(args.out_report))

    print(f"[T07d] n_events={mi.n_events}, n_contestants={mi.n_contestants}, max_active={mi.max_active}, max_elim={mi.max_elim}")
    print(f"[T07d] rule counts: rank={int(np.sum(mi.rule_is_rank))}, percent={int(np.sum(~mi.rule_is_rank))}")
    print(f"[T07d] elim_count: 0:{np.sum(mi.elim_count==0)} 1:{np.sum(mi.elim_count==1)} 2:{np.sum(mi.elim_count==2)} 3:{np.sum(mi.elim_count==3)}")
    print(f"[T07d] Saved report: {args.out_report}")

    active_idx_t = pt.as_tensor_variable(mi.active_idx)
    active_mask_t = pt.as_tensor_variable(mi.active_mask.astype("float64"))
    elim_slots_t = pt.as_tensor_variable(mi.elim_slots)
    elim_mask_t = pt.as_tensor_variable(mi.elim_mask.astype("float64"))
    elim_slots_safe_t = pt.maximum(elim_slots_t, 0)
    ev_idx = pt.arange(mi.n_events)

    J_total_t = pt.as_tensor_variable(mi.J_total.astype("float64"))
    J_pct_t = pt.as_tensor_variable(mi.J_pct.astype("float64"))
    is_rank_t = pt.as_tensor_variable(mi.rule_is_rank.astype("float64"))

    print("=" * 90)
    print("[T07d] Building PyMC model (S(J,pi) one-stage likelihood)...")
    print("=" * 90)

    with pm.Model() as model:
        sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
        sigma_w = pm.HalfNormal("sigma_w", sigma=1.0)
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=1.0)

        kappa_S = pm.HalfNormal("kappa_S", sigma=1.0)

        u_raw = pm.Normal("u_raw", 0.0, 1.0, shape=(mi.n_contestants,))
        w_raw = pm.Normal("w_raw", 0.0, 1.0, shape=(mi.n_events,))

        u = pm.Deterministic("u", (u_raw - pt.mean(u_raw)) * sigma_u)
        w = pm.Deterministic("w", (w_raw - pt.mean(w_raw)) * sigma_w)

        idx_safe = pt.maximum(active_idx_t, 0)
        u_es = u[idx_safe] * active_mask_t

        eps_raw = pm.Normal("eps_raw", 0.0, 1.0, shape=(mi.n_events, mi.max_active))
        eps = pm.Deterministic("eps", eps_raw * sigma_eps)

        theta = pm.Deterministic("theta", u_es + w[:, None] + eps * active_mask_t)
        pi = pm.Deterministic("pi", masked_softmax(theta, active_mask_t, axis=1))

        S_percent = J_pct_t + pi
        rankJ = soft_rank_batch(J_total_t, active_mask_t, tau=float(args.tau_rank))
        rankPi = soft_rank_batch(pi, active_mask_t, tau=float(args.tau_rank))
        S_rank = rankJ + rankPi
        S = pm.Deterministic("S", is_rank_t[:, None] * S_rank + (1.0 - is_rank_t)[:, None] * S_percent)

        weights = pt.exp(-kappa_S * S) * active_mask_t

        # ordered sequential (temporary)
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

        print("[T07d] Sampling config:")
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
    print(f"[T07d] Saved trace: {out_trace}")

    try:
        summ = az.summary(idata, var_names=["sigma_u", "sigma_w", "sigma_eps", "kappa_S"], kind="stats")
        print("[T07d] Posterior summary (key scales):")
        print(summ.to_string())
    except Exception as e:
        print(f"[T07d][WARN] az.summary failed: {e}")

    print("=" * 90)
    print("[T07d] Done.")
    print("=" * 90)
    print("[T07d] Interpretation hints:")
    print("  - S is the mechanism-aligned composite score used to drive elimination.")
    print("  - Multi-elimination is still order-sensitive here; fix comes in T07e.")


if __name__ == "__main__":
    main()
