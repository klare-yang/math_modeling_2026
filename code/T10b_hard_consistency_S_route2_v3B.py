#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T10b_hard_consistency_S_route2_v3B.py

Task point 14: Hard consistency in S-space (mechanism-aligned).
- Uses posterior mean of S (stored as Deterministic('S', ...) in T07f trace)
- Computes bottom-k by S_mean within active set and compares with observed eliminations.
- For judges-save events (use_two_stage==1 and elim_count==1), reports:
    * stage1_hit: eliminated is in bottom-2 (by S_mean)
    * stage2_pred: among bottom-2, predicted eliminated is the one with worse judge rank (softRank(J_total))
    * stage2_hit: eliminated equals that predicted (deterministic proxy for judge choice)

Inputs:
  --trace      data/T07f_trace_route2_v3B.nc
  --t05_path   data/T05_model_ready_events_route2_v3B.csv
  --mask_path  data/T04_judges_save_event_mask_route2_v3B.csv

Outputs:
  data/T10b_hard_consistency_S_route2_v3B.csv
  data/T10b_hard_consistency_S_route2_v3B.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import arviz as az
except Exception as e:
    raise RuntimeError("arviz is required. Install: pip install arviz xarray") from e


EPS = 1e-12


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

def _json_dumps_intlist(xs: List[int]) -> str:
    """json.dumps that is robust to numpy integer types."""
    return json.dumps([int(x) for x in xs])


def soft_rank_np(x: np.ndarray, mask: np.ndarray, tau: float = 0.15) -> np.ndarray:
    """
    Numpy replica of v3B soft_rank:
      r_i = 1 + Σ_{j≠i} σ((x_j - x_i)/tau) * 1{active_j} * 1{active_i}
    Inactive slots return 0.
    """
    x = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=float)
    A = x.shape[0]
    diff = (x[None, :] - x[:, None]) / float(tau)
    sig = 1.0 / (1.0 + np.exp(-diff))
    pair_mask = (m[:, None] * m[None, :]) * (1.0 - np.eye(A, dtype=float))
    r = 1.0 + np.sum(sig * pair_mask, axis=1)
    return r * m


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
    ap.add_argument("--tau_rank", type=float, default=0.15)
    ap.add_argument("--out_csv", default="data/T10b_hard_consistency_S_route2_v3B.csv")
    ap.add_argument("--out_json", default="data/T10b_hard_consistency_S_route2_v3B.json")
    args = ap.parse_args()

    trace_path = Path(args.trace)
    t05_path = Path(args.t05_path)
    mask_path = Path(args.mask_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"[T10b] trace not found: {trace_path}")
    if not t05_path.exists():
        raise FileNotFoundError(f"[T10b] T05 not found: {t05_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"[T10b] mask not found: {mask_path}")

    t05 = pd.read_csv(t05_path)
    if "event_id" not in t05.columns:
        raise ValueError("[T10b] T05 must include event_id.")
    n_events = len(t05)

    mask_df = pd.read_csv(mask_path)
    for col in ["event_id", "use_two_stage", "fallback_one_stage"]:
        if col not in mask_df.columns:
            raise ValueError(f"[T10b] mask missing column: {col}")

    mm = t05[["event_id"]].merge(mask_df[["event_id", "use_two_stage", "fallback_one_stage"]], on="event_id", how="left")
    use_two_stage = (mm["use_two_stage"].fillna(0).astype(int).to_numpy() == 1) & (mm["fallback_one_stage"].fillna(0).astype(int).to_numpy() == 0)
    use_two_stage = use_two_stage.astype(int)

    # active ids
    active_ids_col = None
    for c in ["active_ids", "active_ids_by_event", "active_ids_json"]:
        if c in t05.columns:
            active_ids_col = c
            break
    if active_ids_col is None:
        raise ValueError("[T10b] T05 missing active_ids (JSON list aligned to slots).")
    active_ids = [_parse_json_list(v) for v in t05[active_ids_col].tolist()]

    # eliminated indices
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
    if elim_idx_col is None and elim_ids_col is None:
        raise ValueError("[T10b] T05 must include eliminated_indices or eliminated_ids.")

    elim_slots: List[List[int]] = []
    elim_count = np.zeros((n_events,), dtype=int)
    for e in range(n_events):
        if elim_idx_col is not None:
            idxs = [int(i) for i in _parse_json_list(t05.loc[e, elim_idx_col])]
        else:
            eids = _parse_json_list(t05.loc[e, elim_ids_col])
            aids = active_ids[e]
            aid_to_slot = {aids[s]: s for s in range(len(aids))}
            idxs = [int(aid_to_slot[cid]) for cid in eids if cid in aid_to_slot]
        idxs = [i for i in idxs if i is not None]
        elim_slots.append(sorted(list(set(idxs))))
        elim_count[e] = len(elim_slots[e])

    # Judge totals (for stage2 proxy)
    jtotal_col = None
    for c in ["J_total_by_active", "J_total_active", "judge_total_by_active", "J_total"]:
        if c in t05.columns:
            jtotal_col = c
            break
    if jtotal_col is None:
        raise ValueError("[T10b] T05 missing J_total_by_active (needed for judges-save stage2 proxy).")
    J_total_by_active = [_parse_json_list(v) for v in t05[jtotal_col].tolist()]

    # Read trace and compute S_mean
    idata = az.from_netcdf(trace_path)
    post = idata.posterior
    if "S" not in post:
        raise ValueError("[T10b] trace missing posterior variable 'S'. Run T07f which stores Deterministic('S', ...).")
    S_da = _stack_samples(post["S"])
    if S_da.ndim != 3:
        raise ValueError(f"[T10b] Unexpected S dims={S_da.dims} shape={tuple(S_da.shape)}; expected 3D.")

    dims = list(S_da.dims)
    if "sample" not in dims:
        raise ValueError("[T10b] Failed to create 'sample' dim from posterior.")
    other = [d for d in dims if d != "sample"]
    if len(other) != 2:
        raise ValueError(f"[T10b] Unexpected S dims: {dims}")

    if S_da.sizes[other[0]] == n_events:
        event_dim, slot_dim = other[0], other[1]
    elif S_da.sizes[other[1]] == n_events:
        event_dim, slot_dim = other[1], other[0]
    else:
        raise ValueError(f"[T10b] Could not match S event dimension to T05 length={n_events}. S sizes={S_da.sizes}")

    S_da = S_da.transpose("sample", event_dim, slot_dim)
    S_mean = np.asarray(S_da.mean(dim="sample").values, dtype=float)  # (E,A)
    E, A = S_mean.shape

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

    # compute rankJ for stage2 proxy
    rankJ = np.zeros((E, A), dtype=float)
    for e in range(E):
        jt = J_total_by_active[e]
        if len(jt) < A:
            jt = jt + [0.0] * (A - len(jt))
        else:
            jt = jt[:A]
        rankJ[e, :] = soft_rank_np(np.asarray(jt, dtype=float), active_mask[e, :], tau=float(args.tau_rank))

    rows = []
    for e in range(E):
        k = int(elim_count[e])
        elim = elim_slots[e]
        # inactive slots have +inf score for sorting
        scores = S_mean[e, :].copy()
        scores[active_mask[e, :] < 0.5] = np.inf

        if k == 0:
            bottomk = []
            hit_set = True
            hit_all_in_bottomk = True
            hit_k1 = None
            stage1_hit = None
            stage2_pred = None
            stage2_hit = None
        else:
            # bottom-k indices (smallest scores)
            bottomk = list(np.argsort(scores)[:k])
            hit_set = (set(elim) == set(bottomk))
            hit_all_in_bottomk = all([i in set(bottomk) for i in elim])
            hit_k1 = (elim[0] == bottomk[0]) if (k == 1 and len(elim) == 1) else None

            stage1_hit = None
            stage2_pred = None
            stage2_hit = None

            if use_two_stage[e] == 1 and k == 1 and len(elim) == 1:
                bottom2 = list(np.argsort(scores)[:2])
                stage1_hit = (elim[0] in set(bottom2))
                # stage2 proxy: judges eliminate worse judge rank (larger rankJ)
                r2 = rankJ[e, bottom2]
                stage2_pred = int(bottom2[int(np.argmax(r2))])
                stage2_hit = (elim[0] == stage2_pred)
            else:
                bottom2 = None

        rows.append({
            "event_id": t05.loc[e, "event_id"],
            "season": int(t05.loc[e, "season"]) if "season" in t05.columns else None,
            "week": int(t05.loc[e, "week"]) if "week" in t05.columns else None,
            "rule_mode": str(t05.loc[e, "rule_mode"]) if "rule_mode" in t05.columns else None,
            "elim_count": k,
            "use_two_stage": int(use_two_stage[e]),
            "elim_slots": _json_dumps_intlist(elim),
            "bottomk_slots_by_Smean": _json_dumps_intlist(bottomk),
            "hard_hit_set": int(hit_set),
            "hard_hit_all_in_bottomk": int(hit_all_in_bottomk),
            "hard_hit_k1_exact": (int(hit_k1) if hit_k1 is not None else None),
            "hard_stage1_hit_bottom2": (int(stage1_hit) if stage1_hit is not None else None),
            "hard_stage2_pred_slot": stage2_pred,
            "hard_stage2_hit_pred": (int(stage2_hit) if stage2_hit is not None else None),
        })

    out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    # summary
    def _rate(col: str) -> float:
        s = out[col].dropna().astype(float)
        if len(s) == 0:
            return float("nan")
        return float(s.mean())

    summary: Dict[str, Any] = {
        "trace": str(trace_path),
        "t05": str(t05_path),
        "mask": str(mask_path),
        "n_events": int(E),
        "rates": {
            "hard_hit_set": _rate("hard_hit_set"),
            "hard_hit_all_in_bottomk": _rate("hard_hit_all_in_bottomk"),
            "hard_hit_k1_exact": _rate("hard_hit_k1_exact"),
            "hard_stage1_hit_bottom2": _rate("hard_stage1_hit_bottom2"),
            "hard_stage2_hit_pred": _rate("hard_stage2_hit_pred"),
        },
        "by_use_two_stage": {},
        "by_rule_mode": {},
    }

    for gname, gcol in [("use_two_stage", "use_two_stage"), ("rule_mode", "rule_mode")]:
        grp = out.groupby(gcol, dropna=False)
        payload = {}
        for key, df in grp:
            payload[str(key)] = {
                "n": int(len(df)),
                "hard_hit_set": float(df["hard_hit_set"].mean()),
                "hard_hit_k1_exact": float(df["hard_hit_k1_exact"].dropna().astype(float).mean()) if df["hard_hit_k1_exact"].notna().any() else float("nan"),
                "hard_stage1_hit_bottom2": float(df["hard_stage1_hit_bottom2"].dropna().astype(float).mean()) if df["hard_stage1_hit_bottom2"].notna().any() else float("nan"),
                "hard_stage2_hit_pred": float(df["hard_stage2_hit_pred"].dropna().astype(float).mean()) if df["hard_stage2_hit_pred"].notna().any() else float("nan"),
            }
        summary[f"by_{gname}"] = payload

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 90)
    print("[T10b] Saved:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print("[T10b] Rates:", summary["rates"])
    print("=" * 90)


if __name__ == "__main__":
    main()
