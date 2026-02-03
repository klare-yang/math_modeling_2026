#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t08b_export_pi_posterior_stats_route2_v3B.py

Task point 12:
- Input: trace nc (T07f or T07g), T05_model_ready_events_route2_v3B.csv
- Output: data/T08b_votes_posterior_stats_route2_v3B.csv + json report

Exports posterior summary statistics for vote share pi_{event,slot}:
  pi_mean, pi_sd, pi_hdi_low/high (default 94%), q025/q975, pi_hdi_width.
Also emits a per-event pi_sum_mean diagnostic (sum over active slots).

Assumptions:
- Trace and T05 share the same event ordering (as constructed in T07f).
- T05 provides active_ids as a JSON list aligned to the slot dimension.

Run:
  python code/t08b_export_pi_posterior_stats_route2_v3B.py \
    --trace data/T07f_trace_route2_v3B.nc \
    --t05   data/T05_model_ready_events_route2_v3B.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    import arviz as az
except Exception as e:
    raise RuntimeError(
        "arviz is required to read InferenceData netcdf. Install: pip install arviz"
    ) from e


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
        # tolerate single-quoted pseudo-json
        s2 = s.replace("'", '"')
        return json.loads(s2)


def _stack_samples(da):
    # collapse (chain, draw) -> sample
    if "chain" in da.dims and "draw" in da.dims:
        return da.stack(sample=("chain", "draw"))
    if "sample" in da.dims:
        return da
    # fallback: treat first dim as sample
    return da


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace",
        default="data/T07f_trace_route2_v3B.nc",
        help="InferenceData netcdf from T07f/T07g",
    )
    ap.add_argument(
        "--t05",
        default="data/T05_model_ready_events_route2_v3B.csv",
        help="Model-ready events csv (T05)",
    )
    ap.add_argument(
        "--out_csv",
        default="data/T08b_votes_posterior_stats_route2_v3B.csv",
    )
    ap.add_argument(
        "--out_json",
        default="data/T08b_votes_posterior_stats_route2_v3B.json",
    )
    ap.add_argument("--hdi_prob", type=float, default=0.94)
    ap.add_argument(
        "--max_draws",
        type=int,
        default=0,
        help="If >0, randomly subsample to this many posterior samples",
    )
    ap.add_argument("--seed", type=int, default=20260202)
    args = ap.parse_args()

    trace_path = Path(args.trace)
    t05_path = Path(args.t05)
    if not trace_path.exists():
        raise FileNotFoundError(f"[t08b] trace not found: {trace_path}")
    if not t05_path.exists():
        raise FileNotFoundError(f"[t08b] T05 not found: {t05_path}")

    t05 = pd.read_csv(t05_path)
    if "event_id" not in t05.columns:
        raise ValueError("[t08b] T05 must include 'event_id' column.")
    n_events = len(t05)

    idata = az.from_netcdf(trace_path)
    post = idata.posterior
    if "pi" not in post:
        raise ValueError(
            "[t08b] trace posterior is missing variable 'pi'. "
            "Ensure T07f stored Deterministic('pi', ...)."
        )

    pi = _stack_samples(post["pi"])
    if pi.ndim != 3:
        raise ValueError(
            f"[t08b] Unexpected pi dims={pi.dims} shape={tuple(pi.shape)}; "
            "expected 3D (sample,event,slot)."
        )

    # Identify event and slot dims by matching n_events
    dims = list(pi.dims)
    if "sample" not in dims:
        raise ValueError("[t08b] Failed to create 'sample' dim from posterior.")
    other = [d for d in dims if d != "sample"]
    if len(other) != 2:
        raise ValueError(
            f"[t08b] pi dims after stacking should have 2 non-sample dims, got {dims}"
        )

    if pi.sizes[other[0]] == n_events:
        event_dim, slot_dim = other[0], other[1]
    elif pi.sizes[other[1]] == n_events:
        event_dim, slot_dim = other[1], other[0]
    else:
        raise ValueError(
            f"[t08b] Could not match pi event dimension to T05 length={n_events}. "
            f"pi sizes={pi.sizes}"
        )

    pi = pi.transpose("sample", event_dim, slot_dim)

    # Optional posterior subsampling
    rng = np.random.default_rng(args.seed)
    n_samp = int(pi.sizes["sample"])
    if args.max_draws and args.max_draws > 0 and args.max_draws < n_samp:
        idx = rng.choice(n_samp, size=args.max_draws, replace=False)
        pi = pi.isel(sample=idx)
        n_samp = int(pi.sizes["sample"])

    # Stats
    pi_mean = pi.mean(dim="sample")
    pi_sd = pi.std(dim="sample")

    hdi = az.hdi(pi, hdi_prob=args.hdi_prob, input_core_dims=[["sample"]])

    # arviz/xarray version differences: az.hdi may return DataArray or Dataset.
    try:
        import xarray as xr  # type: ignore
    except Exception:
        xr = None  # type: ignore

    if xr is not None and isinstance(hdi, xr.Dataset):
        if "pi" in hdi.data_vars:
            hdi_da = hdi["pi"]
        else:
            hdi_da = next(iter(hdi.data_vars.values()))
    else:
        hdi_da = hdi

    if "hdi" not in hdi_da.dims:
        raise ValueError(f"[t08b] Unexpected az.hdi dims: {hdi_da.dims}")

    pi_hdi_low = hdi_da.sel(hdi="lower")
    pi_hdi_high = hdi_da.sel(hdi="higher")
    pi_hdi_width = (pi_hdi_high - pi_hdi_low)

    # convert to numpy once (avoids .values/.to_numpy differences)
    pi_hdi_low_np = np.asarray(pi_hdi_low)
    pi_hdi_high_np = np.asarray(pi_hdi_high)
    pi_hdi_width_np = np.asarray(pi_hdi_width)

    q = pi.quantile([0.025, 0.975], dim="sample")
    q025 = q.sel(quantile=0.025)
    q975 = q.sel(quantile=0.975)

    # Active ids column
    active_ids_col = None
    for c in ["active_ids", "active_ids_by_event", "active_ids_json"]:
        if c in t05.columns:
            active_ids_col = c
            break
    if active_ids_col is None:
        raise ValueError(
            "[t08b] T05 missing 'active_ids' (JSON list aligned to slots)."
        )

    active_ids = [_parse_json_list(v) for v in t05[active_ids_col].tolist()]

    max_active = int(pi_mean.shape[1])
    rows = []
    for e in range(n_events):
        aids = active_ids[e]
        # pad/truncate to max_active
        if len(aids) < max_active:
            aids = aids + [None] * (max_active - len(aids))
        else:
            aids = aids[:max_active]

        for s in range(max_active):
            cid = aids[s]
            if cid is None:
                continue
            rows.append(
                {
                    "event_id": t05.loc[e, "event_id"],
                    "season": int(t05.loc[e, "season"]) if "season" in t05.columns else None,
                    "week": int(t05.loc[e, "week"]) if "week" in t05.columns else None,
                    "slot": int(s),
                    "contestant_id": cid,
                    "pi_mean": float(pi_mean.values[e, s]),
                    "pi_sd": float(pi_sd.values[e, s]),
                    "pi_hdi_low": float(pi_hdi_low_np[e, s]),
                    "pi_hdi_high": float(pi_hdi_high_np[e, s]),
                    "pi_hdi_width": float(pi_hdi_width_np[e, s]),
                    "pi_q025": float(q025.values[e, s]),
                    "pi_q975": float(q975.values[e, s]),
                }
            )

    out_df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    # pi sum diagnostics (sum over active slots)
    sum_by_event = (
        out_df.groupby("event_id")["pi_mean"].sum().reset_index().rename(columns={"pi_mean": "pi_sum_mean"})
    )
    merged = sum_by_event.merge(t05[["event_id"]], on="event_id", how="right")
    sums = merged["pi_sum_mean"].fillna(0.0).to_numpy()
    abs_err = np.abs(sums - 1.0)

    report: Dict[str, Any] = {
        "trace": str(trace_path),
        "t05": str(t05_path),
        "n_events": int(n_events),
        "max_active": int(max_active),
        "n_posterior_samples_used": int(n_samp),
        "hdi_prob": float(args.hdi_prob),
        "pi_sum_check": {
            "mean_pi_sum_mean": float(np.mean(sums)),
            "mean_abs_err": float(np.mean(abs_err)),
            "max_abs_err": float(np.max(abs_err)),
            "p95_abs_err": float(np.quantile(abs_err, 0.95)),
        },
        "outputs": {"csv": str(out_csv)},
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 90)
    print("[t08b] Saved:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print("[t08b] pi_sum_mean (over active slots):")
    print(f"  mean={report['pi_sum_check']['mean_pi_sum_mean']:.6f}  max_abs_err={report['pi_sum_check']['max_abs_err']:.6f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
