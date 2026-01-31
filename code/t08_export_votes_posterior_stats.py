#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T08 — Export posterior stats for fan votes (logV) per (contestant, week/event).

Key design:
- Read posterior and constant_data directly from PyMC InferenceData netcdf
  to guarantee exact consistency with the fitted model (no re-normalization drift).
- Reconstruct contestant_keys ordering exactly as in T07 by:
    contestant_keys = sorted(set(all active_ids across events))
- Validate alignment:
    act_sorted[j] == contestant_keys[active_key_idx[e,j]] for active slots

Outputs:
- data/T08_logV_posterior_stats_<tag>.csv
  Includes:
    logV_mean, logV_sd, logV_ci2_5, logV_ci97_5
    logV_centered_mean, ..., (event-wise centered)
    share_mean, share_sd, share_ci2_5, share_ci97_5 (softmax share within event)
    eliminated_flag (slot is in eliminated_indices)
- data/T08_alignment_report_<tag>.json

Usage:
  python code/t08_export_votes_posterior_stats.py

Recommended:
  Use the high-quality trace from T07b:
    --trace data/T07_trace_route2_v2_ta97.nc
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr


def _parse_list_cell(x: Any) -> list:
    """Robustly parse list-like cells: may be JSON string or Python repr."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if s == "":
        return []
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Cannot parse list cell: {x}") from e


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    logits: (S, nA)
    return: (S, nA)
    """
    m = np.max(logits, axis=1, keepdims=True)
    z = logits - m
    ez = np.exp(z)
    denom = np.sum(ez, axis=1, keepdims=True)
    return ez / denom


def q025_q975(x: np.ndarray) -> Tuple[float, float]:
    lo, hi = np.quantile(x, [0.025, 0.975])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--trace", default="data/T07_trace_route2_v2_ta97.nc")
    ap.add_argument("--t05", default="data/T05_model_ready_events_route2_v2.csv")

    ap.add_argument("--out_csv", default="data/T08_logV_posterior_stats_route2_v2_ta97.csv")
    ap.add_argument("--out_report", default="data/T08_alignment_report_route2_v2_ta97.json")

    # performance / behavior
    ap.add_argument("--float32", action="store_true", help="cast posterior arrays to float32 to reduce RAM")
    ap.add_argument("--max_events_debug", type=int, default=0, help=">0 to process only first N events (debug)")

    args = ap.parse_args()

    trace_path = Path(args.trace)
    t05_path = Path(args.t05)
    out_csv = Path(args.out_csv)
    out_report = Path(args.out_report)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    if not trace_path.exists():
        raise FileNotFoundError(f"[T08] missing trace: {trace_path}")
    if not t05_path.exists():
        raise FileNotFoundError(f"[T08] missing T05: {t05_path}")

    print(f"[T08] Loading T05: {t05_path}")
    ev = pd.read_csv(t05_path)
    req = {"season", "week", "active_ids", "eliminated_indices", "eliminated_ids", "n_active", "elim_count"}
    miss = req - set(ev.columns)
    if miss:
        raise ValueError(f"[T08] T05 missing columns: {sorted(miss)}")

    # parse list cells and build event_id (keep row order!)
    ev = ev.copy()
    ev["season"] = ev["season"].astype(int)
    ev["week"] = ev["week"].astype(int)
    ev["active_ids"] = ev["active_ids"].apply(_parse_list_cell)
    ev["eliminated_ids"] = ev["eliminated_ids"].apply(_parse_list_cell)
    ev["eliminated_indices"] = ev["eliminated_indices"].apply(_parse_list_cell)
    ev["event_id"] = ev.apply(lambda r: f"{int(r['season'])}-{int(r['week'])}", axis=1)

    # reconstruct contestant_keys ordering (must match T07 build_model_input)
    contestant_keys = sorted({k for lst in ev["active_ids"] for k in lst})
    key2idx = {k: i for i, k in enumerate(contestant_keys)}

    print(f"[T08] Loading netcdf groups: posterior + constant_data from {trace_path}")
    post = xr.open_dataset(trace_path, group="posterior")
    const = xr.open_dataset(trace_path, group="constant_data")

    # posterior arrays
    beta0 = post["beta0"].values  # (chain, draw)
    beta1 = post["beta1"].values
    u = post["u"].values          # (chain, draw, n_contestants)
    w = post["w"].values          # (chain, draw, n_events)

    # constant data arrays (exactly used in model)
    active_key_idx = const["active_key_idx"].values.astype(np.int32)  # (E, A)
    score_z = const["score_z"].values                                  # (E, A)
    active_mask = const["active_mask"].values.astype(np.int8)          # (E, A)

    E, A = active_key_idx.shape
    if len(ev) != E:
        raise ValueError(f"[T08] event count mismatch: T05 has {len(ev)} rows, trace has {E} events")

    # optional debug cut
    if args.max_events_debug and args.max_events_debug > 0:
        E_use = min(E, int(args.max_events_debug))
        ev = ev.iloc[:E_use].copy()
        active_key_idx = active_key_idx[:E_use, :]
        score_z = score_z[:E_use, :]
        active_mask = active_mask[:E_use, :]
        w = w[:, :, :E_use]
        E = E_use
        print(f"[T08] DEBUG: processing only first {E_use} events")

    # flatten chain/draw -> S
    S = beta0.shape[0] * beta0.shape[1]
    beta0_f = beta0.reshape(-1)
    beta1_f = beta1.reshape(-1)
    u_f = u.reshape(S, -1)  # (S, C)
    w_f = w.reshape(S, -1)  # (S, E)

    if args.float32:
        beta0_f = beta0_f.astype(np.float32)
        beta1_f = beta1_f.astype(np.float32)
        u_f = u_f.astype(np.float32)
        w_f = w_f.astype(np.float32)
        score_z = score_z.astype(np.float32)

    # alignment checks
    mismatch = 0
    mismatch_samples: List[Dict[str, Any]] = []

    for e in range(E):
        act = list(ev.iloc[e]["active_ids"])
        act_sorted = sorted(act)
        nA = int(active_mask[e].sum())
        if nA != len(act_sorted):
            mismatch += 1
            if len(mismatch_samples) < 10:
                mismatch_samples.append({
                    "event_id": str(ev.iloc[e]["event_id"]),
                    "reason": "n_active mismatch between trace.active_mask and T05.active_ids",
                    "nA_trace": nA,
                    "nA_t05": len(act_sorted),
                })
            continue

        # compare slot-to-key mapping
        for j in range(nA):
            idx = int(active_key_idx[e, j])
            k_from_trace = contestant_keys[idx]
            k_from_t05 = act_sorted[j]
            if k_from_trace != k_from_t05:
                mismatch += 1
                if len(mismatch_samples) < 10:
                    mismatch_samples.append({
                        "event_id": str(ev.iloc[e]["event_id"]),
                        "slot": int(j),
                        "key_from_trace": k_from_trace,
                        "key_from_t05": k_from_t05,
                    })
                break

    report = {
        "trace": str(trace_path),
        "t05": str(t05_path),
        "dims": {"S_samples": int(S), "n_events": int(E), "max_active": int(A), "n_contestants": int(len(contestant_keys))},
        "alignment": {"mismatch_count": int(mismatch), "mismatch_samples": mismatch_samples},
    }

    if mismatch > 0:
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"[T08] Alignment mismatch detected (count={mismatch}). "
            f"See report: {out_report} . Fix T05 ordering / active_ids sorting to match trace."
        )

    print("[T08] Alignment OK. Computing posterior stats per active slot...")

    rows: List[Dict[str, Any]] = []

    for e in range(E):
        season = int(ev.iloc[e]["season"])
        week = int(ev.iloc[e]["week"])
        event_id = str(ev.iloc[e]["event_id"])

        act = sorted(list(ev.iloc[e]["active_ids"]))
        nA = int(active_mask[e].sum())
        elim_inds = list(ev.iloc[e]["eliminated_indices"])
        elim_set = set(int(x) for x in elim_inds) if elim_inds else set()

        # Build logV samples for all active slots in this event: (S, nA)
        w_e = w_f[:, e]  # (S,)

        L = np.empty((S, nA), dtype=np.float32 if args.float32 else np.float64)
        for j in range(nA):
            idx = int(active_key_idx[e, j])
            sc = float(score_z[e, j])
            # logV = posterior draws of beta0 + beta1*score + u[idx] + w[e]
            L[:, j] = beta0_f + beta1_f * sc + u_f[:, idx] + w_e

        # event-wise centering (per draw): centered logits
        L_mean = L.mean(axis=1, keepdims=True)
        Lc = L - L_mean

        # within-event vote share (softmax on centered logV)
        share = stable_softmax(Lc)  # (S, nA)

        for j in range(nA):
            idx = int(active_key_idx[e, j])
            k = contestant_keys[idx]
            sc = float(score_z[e, j])

            x = L[:, j]
            xc = Lc[:, j]
            sh = share[:, j]

            lo, hi = q025_q975(x)
            clo, chi = q025_q975(xc)
            slo, shi = q025_q975(sh)

            rows.append({
                "season": season,
                "week": week,
                "event_id": event_id,
                "slot": int(j),
                "n_active": int(nA),

                "contestant_key": k,
                "contestant_idx": int(idx),
                "score_z": sc,

                "eliminated_flag": int(j in elim_set),

                # logV (raw)
                "logV_mean": float(np.mean(x)),
                "logV_sd": float(np.std(x, ddof=0)),
                "logV_ci2_5": lo,
                "logV_ci97_5": hi,

                # logV centered within event
                "logV_centered_mean": float(np.mean(xc)),
                "logV_centered_sd": float(np.std(xc, ddof=0)),
                "logV_centered_ci2_5": clo,
                "logV_centered_ci97_5": chi,

                # vote share within event
                "share_mean": float(np.mean(sh)),
                "share_sd": float(np.std(sh, ddof=0)),
                "share_ci2_5": slo,
                "share_ci97_5": shi,
            })

    df = pd.DataFrame(rows)
    df.sort_values(["season", "week", "slot"], inplace=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[T08] Saved: {out_csv}")
    print(f"[T08] Saved: {out_report}")
    print("[T08] Done.")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
