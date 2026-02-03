#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T17b_random_effects_plots_route2_v3B.py

Task point 19: Random effects / parameter plots for the v3B model.

Because variable names can differ across PyMC builds, this script is defensive:
- Reads trace (InferenceData netcdf)
- Lists posterior variables to a JSON report
- If present, plots distributions for:
    kappa_S, kappa_J, (optionally) sigma_u, sigma_w
- If present, plots posterior means for vectors:
    u (contestant RE), w (event RE)  -- variable names matched by heuristics

Outputs:
  - data/T17b_random_effects_report_route2_v3B.json
  - data/fig/T17b_kappa_S_hist.png (if available)
  - data/fig/T17b_kappa_J_hist.png (if available)
  - data/fig/T17b_u_mean.png       (if available)
  - data/fig/T17b_w_mean.png       (if available)

Notes:
- Uses matplotlib only; does not set explicit colors (defaults apply).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import arviz as az
except Exception as e:
    raise RuntimeError("arviz is required. Install: pip install arviz xarray") from e


def _stack_samples(da):
    if "chain" in da.dims and "draw" in da.dims:
        return da.stack(sample=("chain", "draw"))
    if "sample" in da.dims:
        return da
    return da


def _find_var(names: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in names:
            return c
    return None


def _find_by_substring(names: List[str], subs: List[str]) -> Optional[str]:
    for n in names:
        for s in subs:
            if s in n:
                return n
    return None


def _hist_plot(x: np.ndarray, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=40)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _line_plot(y: np.ndarray, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("posterior mean")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="data/T07f_trace_route2_v3B.nc")
    ap.add_argument("--out_report", default="data/T17b_random_effects_report_route2_v3B.json")
    ap.add_argument("--out_dir", default="data/fig")
    args = ap.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(f"[T17b] trace not found: {trace_path}")

    idata = az.from_netcdf(trace_path)
    post = idata.posterior
    var_names = list(post.data_vars.keys())

    report: Dict[str, object] = {
        "trace": str(trace_path),
        "posterior_vars": var_names,
        "selected": {},
        "notes": "Selections use heuristics; adjust candidates in script if your trace uses different variable names.",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scalars
    kS_name = _find_var(var_names, ["kappa_S", "kappaS", "kappa_s"])
    kJ_name = _find_var(var_names, ["kappa_J", "kappaJ", "kappa_j"])
    if kS_name:
        kS = np.asarray(_stack_samples(post[kS_name]).values).reshape(-1)
        _hist_plot(kS, f"{kS_name} posterior", out_dir / "T17b_kappa_S_hist.png")
        report["selected"]["kappa_S"] = kS_name
        report["selected"]["kappa_S_mean"] = float(np.mean(kS))
    if kJ_name:
        kJ = np.asarray(_stack_samples(post[kJ_name]).values).reshape(-1)
        _hist_plot(kJ, f"{kJ_name} posterior", out_dir / "T17b_kappa_J_hist.png")
        report["selected"]["kappa_J"] = kJ_name
        report["selected"]["kappa_J_mean"] = float(np.mean(kJ))

    # Vector random effects (heuristics)
    u_name = _find_var(var_names, ["u", "u_contestant", "u_c", "u_celeb", "u_id"]) or _find_by_substring(var_names, ["u_cont", "u_celeb", "u_c"])
    w_name = _find_var(var_names, ["w", "w_event", "w_t", "w_week"]) or _find_by_substring(var_names, ["w_event", "w_t"])

    if u_name:
        u_da = _stack_samples(post[u_name])
        # ensure vector by averaging over sample dimension
        u_mean = np.asarray(u_da.mean(dim="sample").values).reshape(-1)
        _line_plot(u_mean, f"{u_name} posterior mean", out_dir / "T17b_u_mean.png")
        report["selected"]["u"] = u_name
        report["selected"]["u_len"] = int(len(u_mean))

    if w_name:
        w_da = _stack_samples(post[w_name])
        w_mean = np.asarray(w_da.mean(dim="sample").values).reshape(-1)
        _line_plot(w_mean, f"{w_name} posterior mean", out_dir / "T17b_w_mean.png")
        report["selected"]["w"] = w_name
        report["selected"]["w_len"] = int(len(w_mean))

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 90)
    print("[T17b] Saved report:", out_report)
    print("[T17b] Figures (if variables found) saved in:", out_dir)
    print("=" * 90)


if __name__ == "__main__":
    main()
