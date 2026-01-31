#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T07b: Re-fit T07 with higher-quality sampler settings and export diagnostics.

This is a wrapper around:
  code/t07_run_model_sample_route2_fixed.py

It:
  1) re-runs T07 with improved NUTS settings (target_accept, tune, draws)
  2) loads the saved netcdf trace
  3) exports az.summary + key diagnostics (divergences, rhat, ess)

Usage:
  python code/t07b_refit_high_quality.py
  python code/t07b_refit_high_quality.py --target_accept 0.97 --tune 2000 --draws 1200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import arviz as az
import numpy as np


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t07_script", default="code/t07_run_model_sample_route2_fixed.py")

    ap.add_argument("--input_t05", default="data/T05_model_ready_events_route2_v2.csv")
    ap.add_argument("--input_panel", default="data/T02_long_format_panel.csv")

    # Quality knobs
    ap.add_argument("--draws", type=int, default=1200)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.97)
    ap.add_argument("--seed", type=int, default=20260130)

    # Output prefix for this refit
    ap.add_argument("--tag", default="route2_v2_ta97")

    args = ap.parse_args()

    t07_script = Path(args.t07_script)
    if not t07_script.exists():
        raise FileNotFoundError(f"[T07b] cannot find t07 script: {t07_script}")

    out_trace = Path(f"data/T07_trace_{args.tag}.nc")
    out_report = Path(f"data/T07_input_consistency_report_{args.tag}.json")
    out_mean_consistency = Path(f"data/T07_posterior_mean_consistency_{args.tag}.json")
    out_summary = Path(f"data/T07b_summary_{args.tag}.csv")
    out_diag = Path(f"data/T07b_diagnostics_{args.tag}.json")

    out_trace.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(t07_script),
        "--input_t05", args.input_t05,
        "--input_panel", args.input_panel,
        "--out_report", str(out_report),
        "--out_trace", str(out_trace),
        "--out_mean_consistency", str(out_mean_consistency),
        "--draws", str(args.draws),
        "--tune", str(args.tune),
        "--chains", str(args.chains),
        "--target_accept", str(args.target_accept),
        "--seed", str(args.seed),
    ]

    print("[T07b] Running T07 with high-quality sampler settings...")
    print("[T07b] CMD:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("[T07b] Loading trace:", out_trace)
    idata = az.from_netcdf(out_trace)

    # ---- Diagnostics ----
    diag: Dict[str, Any] = {}
    diag["trace_path"] = str(out_trace)
    diag["sampler_settings"] = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "target_accept": args.target_accept,
        "seed": args.seed,
    }

    # Divergences
    div = idata.sample_stats.get("diverging", None)
    if div is not None:
        div_count = int(np.asarray(div).sum())
        diag["divergences_total"] = div_count
        diag["divergences_rate"] = div_count / (args.draws * args.chains)
    else:
        diag["divergences_total"] = None
        diag["divergences_rate"] = None

    # R-hat & ESS for key params (adjust names if your model differs)
    key_vars: List[str] = []
    for v in ["beta0", "beta1", "sigma_u", "sigma_w"]:
        if v in idata.posterior.data_vars:
            key_vars.append(v)

    diag["key_vars_found"] = key_vars

    if key_vars:
        rhat = az.rhat(idata, var_names=key_vars)
        ess_bulk = az.ess(idata, var_names=key_vars, method="bulk")
        ess_tail = az.ess(idata, var_names=key_vars, method="tail")

        # Convert to plain JSONable numbers (take mean over dimensions if any)
        def _reduce(x):
            arr = np.asarray(x)
            return _to_float(np.nanmean(arr))

        diag["rhat_mean"] = {v: _reduce(rhat[v]) for v in key_vars}
        diag["ess_bulk_mean"] = {v: _reduce(ess_bulk[v]) for v in key_vars}
        diag["ess_tail_mean"] = {v: _reduce(ess_tail[v]) for v in key_vars}

    # Full summary table (csv)
    summary = az.summary(
        idata,
        var_names=key_vars if key_vars else None,
        round_to=4,
    )
    summary.to_csv(out_summary, encoding="utf-8")
    print("[T07b] Saved az.summary:", out_summary)

    # Save diagnostics json
    out_diag.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[T07b] Saved diagnostics:", out_diag)

    # Quick verdict
    print("[T07b] Quick verdict:")
    print("  divergences_total =", diag["divergences_total"])
    if "rhat_mean" in diag:
        print("  rhat_mean =", diag["rhat_mean"])
    if "ess_bulk_mean" in diag:
        print("  ess_bulk_mean =", diag["ess_bulk_mean"])


if __name__ == "__main__":
    main()
