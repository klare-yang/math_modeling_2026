#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M09: Fan vote latent interval-censored Bayesian LMM (Model 3 latent variant)

Goal:
- Treat fan vote (logit share) as latent outcome constrained by [logit(ci2.5), logit(ci97.5)].
- Structural model: Normal(mu_i, sigma_eps) with fixed effects (age, week, industry) and random intercepts (pro, season, celeb).

Inputs:
- data/M01_panel_m3.csv
- data/M01_keymap_m3.json

Outputs:
- data/M09_trace_m3_fans_latent.nc
- data/M09_summary_m3_fans_latent.csv
- data/M09_diagnostics_m3_fans_latent.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az
import pytensor.tensor as pt


DEFAULT_PANEL = Path("data/M01_panel_m3.csv")
DEFAULT_MAP = Path("data/M01_keymap_m3.json")

DEFAULT_TRACE_OUT = Path("data/M09_trace_m3_fans_latent.nc")
DEFAULT_SUMMARY_OUT = Path("data/M09_summary_m3_fans_latent.csv")
DEFAULT_DIAG_OUT = Path("data/M09_diagnostics_m3_fans_latent.json")


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def clip_prob(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def main():
    parser = argparse.ArgumentParser(
        description="M09: Fan latent interval-censored Bayesian LMM (Normal(mu,sigma) with interval likelihood)."
    )
    parser.add_argument("--panel", type=str, default=str(DEFAULT_PANEL))
    parser.add_argument("--map", type=str, default=str(DEFAULT_MAP))
    parser.add_argument("--out_trace", type=str, default=str(DEFAULT_TRACE_OUT))
    parser.add_argument("--out_summary", type=str, default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--out_diag", type=str, default=str(DEFAULT_DIAG_OUT))

    parser.add_argument(
        "--include_celeb_re",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include celebrity random intercept (default: True).",
    )

    parser.add_argument("--eps_clip", type=float, default=1e-6, help="Clipping epsilon for probabilities before logit.")
    parser.add_argument("--seed", type=int, default=20260201)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--target_accept", type=float, default=0.97)

    args = parser.parse_args()

    panel_path = Path(args.panel)
    map_path = Path(args.map)
    out_trace = Path(args.out_trace)
    out_summary = Path(args.out_summary)
    out_diag = Path(args.out_diag)

    if not panel_path.exists():
        raise FileNotFoundError(f"[M09] Panel not found: {panel_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"[M09] Map not found: {map_path}")

    out_trace.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_diag.parent.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    meta = json.load(open(map_path, "r", encoding="utf-8"))

    required = [
        "share_ci2_5", "share_ci97_5",
        "age_z", "week_z",
        "industry_id", "pro_id", "season_id", "celeb_id",
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"[M09] Missing columns: {missing}")

    before = len(panel)
    panel2 = panel.dropna(subset=required).copy()
    dropped = before - len(panel2)

    for c in ["industry_id", "pro_id", "season_id", "celeb_id"]:
        panel2[c] = panel2[c].astype(int)

    ci_lo = panel2["share_ci2_5"].astype(float).to_numpy()
    ci_hi = panel2["share_ci97_5"].astype(float).to_numpy()

    # Ensure bounds valid
    ci_lo = clip_prob(ci_lo, args.eps_clip)
    ci_hi = clip_prob(ci_hi, args.eps_clip)
    # If any pathological inversions, swap
    swap = ci_lo > ci_hi
    if np.any(swap):
        tmp = ci_lo[swap].copy()
        ci_lo[swap] = ci_hi[swap]
        ci_hi[swap] = tmp

    L = logit(ci_lo)
    U = logit(ci_hi)

    age = panel2["age_z"].astype(float).to_numpy()
    week = panel2["week_z"].astype(float).to_numpy()
    ind = panel2["industry_id"].to_numpy()
    pro = panel2["pro_id"].to_numpy()
    season = panel2["season_id"].to_numpy()
    celeb = panel2["celeb_id"].to_numpy()

    pro_names = meta["id_maps"]["pro"]["names"]
    ind_names = meta["id_maps"]["industry"]["names"]
    season_vals = meta["id_maps"]["season"]["values"]
    celeb_names = meta["id_maps"]["celebrity"]["names"]

    max_ok = {
        "pro_id_max": int(pro.max()),
        "pro_id_dim": len(pro_names),
        "industry_id_max": int(ind.max()),
        "industry_dim": len(ind_names),
        "season_id_max": int(season.max()),
        "season_dim": len(season_vals),
        "celeb_id_max": int(celeb.max()),
        "celeb_dim": len(celeb_names),
    }

    # -----------------------------
    # CHECK BLOCK
    # -----------------------------
    print("\n" + "=" * 90)
    print("M09 CHECK BLOCK START")
    print("=" * 90)
    print(f"[M09] panel_path = {panel_path}")
    print(f"[M09] map_path   = {map_path}")
    print(f"[M09] include_celeb_re = {bool(args.include_celeb_re)}")
    print(f"[M09] rows_before_drop = {before}")
    print(f"[M09] rows_after_drop  = {len(panel2)}")
    print(f"[M09] dropped_missing_required = {dropped}")
    print(f"[M09] eps_clip = {args.eps_clip}")

    print("\n[M09] DIMENSIONS (from meta):")
    print(f"  n_pro      = {len(pro_names)}")
    print(f"  n_industry = {len(ind_names)}")
    print(f"  n_season   = {len(season_vals)}")
    print(f"  n_celeb    = {len(celeb_names)}")

    print("\n[M09] MAX ID CHECKS:")
    for k, v in max_ok.items():
        print(f"  {k}: {v}")
    print("  NOTE: id_max must be < corresponding dim.")

    def _dist(name, arr):
        arr = np.asarray(arr)
        q = np.quantile(arr, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
        print(f"\n[M09] DIST {name}:")
        print(f"  n={arr.size} mean={arr.mean():.6f} sd={arr.std(ddof=0):.6f}")
        print(f"  min={q[0]:.6f} p01={q[1]:.6f} p05={q[2]:.6f} p50={q[3]:.6f} p95={q[4]:.6f} p99={q[5]:.6f} max={q[6]:.6f}")

    _dist("L_logit(ci2.5)", L)
    _dist("U_logit(ci97.5)", U)
    _dist("age_z", age)
    _dist("week_z", week)

    w = U - L
    _dist("interval_width_logit", w)
    print(f"\n[M09] interval_width_logit <= 1e-8 count: {(w <= 1e-8).sum()}")

    print("\nM09 CHECK BLOCK END")
    print("=" * 90 + "\n")

    assert pro.max() < len(pro_names)
    assert ind.max() < len(ind_names)
    assert season.max() < len(season_vals)
    assert celeb.max() < len(celeb_names)

    coords = {
        "obs": np.arange(len(panel2)),
        "pro": np.arange(len(pro_names)),
        "industry": np.arange(len(ind_names)),
        "season": np.arange(len(season_vals)),
        "celeb": np.arange(len(celeb_names)),
    }

    with pm.Model(coords=coords) as model:
        # Data
        age_d = pm.Data("age_z", age, dims="obs")
        week_d = pm.Data("week_z", week, dims="obs")
        ind_d = pm.Data("industry_id", ind, dims="obs")
        pro_d = pm.Data("pro_id", pro, dims="obs")
        season_d = pm.Data("season_id", season, dims="obs")
        celeb_d = pm.Data("celeb_id", celeb, dims="obs")

        L_d = pm.Data("L", L, dims="obs")
        U_d = pm.Data("U", U, dims="obs")

        # Fixed effects
        beta0 = pm.Normal("beta0", mu=0.0, sigma=1.0)
        beta_age = pm.Normal("beta_age", mu=0.0, sigma=1.0)
        beta_week = pm.Normal("beta_week", mu=0.0, sigma=1.0)

        beta_ind_raw = pm.Normal("beta_ind_raw", mu=0.0, sigma=1.0, dims="industry")
        beta_ind = pm.Deterministic("beta_ind", beta_ind_raw - pt.mean(beta_ind_raw), dims="industry")

        # Residual SD
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=1.0)

        # Random effects (non-centered)
        sigma_pro = pm.HalfNormal("sigma_pro", sigma=1.0)
        z_pro = pm.Normal("z_pro", mu=0.0, sigma=1.0, dims="pro")
        a_pro = pm.Deterministic("a_pro", z_pro * sigma_pro, dims="pro")

        sigma_season = pm.HalfNormal("sigma_season", sigma=1.0)
        z_season = pm.Normal("z_season", mu=0.0, sigma=1.0, dims="season")
        a_season = pm.Deterministic("a_season", z_season * sigma_season, dims="season")

        if args.include_celeb_re:
            sigma_celeb = pm.HalfNormal("sigma_celeb", sigma=1.0)
            z_celeb = pm.Normal("z_celeb", mu=0.0, sigma=1.0, dims="celeb")
            a_celeb = pm.Deterministic("a_celeb", z_celeb * sigma_celeb, dims="celeb")
            sigma_celeb_fixed = None
        else:
            a_celeb = pm.Deterministic("a_celeb", pt.zeros((len(celeb_names),), dtype="float64"), dims="celeb")
            sigma_celeb_fixed = 0.0

        mu = (
            beta0
            + beta_age * age_d
            + beta_week * week_d
            + beta_ind[ind_d]
            + a_pro[pro_d]
            + a_season[season_d]
            + a_celeb[celeb_d]
        )

        # Interval-censored Normal likelihood:
        # log( Phi((U-mu)/sigma) - Phi((L-mu)/sigma) )
        base = pm.Normal.dist(mu=mu, sigma=sigma_eps)
        logcdf_U = pm.logcdf(base, U_d)
        logcdf_L = pm.logcdf(base, L_d)
        logp = pm.math.logdiffexp(logcdf_U, logcdf_L)
        pm.Potential("interval_like", pm.math.sum(logp))

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.seed,
            return_inferencedata=True,
            progressbar=True,
        )

    print(f"[M09] Saving trace to: {out_trace}")
    idata.to_netcdf(out_trace, engine="h5netcdf")

    summary_vars: List[str] = ["beta0", "beta_age", "beta_week", "sigma_eps", "sigma_pro", "sigma_season"]
    if args.include_celeb_re:
        summary_vars.append("sigma_celeb")

    summ = az.summary(idata, var_names=summary_vars, round_to=6)
    summ.to_csv(out_summary)
    print(f"[M09] Saved summary: {out_summary}")

    # Diagnostics
    div = int(idata.sample_stats["diverging"].sum().values) if "diverging" in idata.sample_stats else None
    rhat = az.rhat(idata, var_names=summary_vars)
    ess = az.ess(idata, var_names=summary_vars)

    qc = {
        "n_obs_used": int(len(panel2)),
        "divergences_total": div,
        "rhat_max": float(np.nanmax(rhat.to_array().values)),
        "ess_min": float(np.nanmin(ess.to_array().values)),
        "max_id_check": max_ok,
        "include_celeb_re": bool(args.include_celeb_re),
        "sigma_celeb_fixed_if_disabled": sigma_celeb_fixed,
        "interval_width_logit_min": float(np.min(U - L)),
        "interval_width_logit_p01": float(np.quantile(U - L, 0.01)),
        "interval_width_logit_mean": float(np.mean(U - L)),
    }

    diag: Dict[str, Any] = {
        "inputs": {"panel": str(panel_path), "map": str(map_path)},
        "outputs": {"trace": str(out_trace), "summary": str(out_summary)},
        "config": {
            "seed": int(args.seed),
            "chains": int(args.chains),
            "draws": int(args.draws),
            "tune": int(args.tune),
            "target_accept": float(args.target_accept),
            "include_celeb_re": bool(args.include_celeb_re),
            "eps_clip": float(args.eps_clip),
        },
        "qc": qc,
    }

    with open(out_diag, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f"[M09] Saved diagnostics: {out_diag}")

    print("\n" + "=" * 90)
    print("M09 POST-SAMPLING CHECK BLOCK START")
    print("=" * 90)
    print("[M09] Summary (key vars):")
    print(summ.to_string())
    print("\n[M09] Diagnostics:")
    print(json.dumps(qc, indent=2))
    print("M09 POST-SAMPLING CHECK BLOCK END")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
