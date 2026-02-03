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

DEFAULT_TRACE_OUT = Path("data/M07_trace_m3_judges_celeb.nc")
DEFAULT_SUMMARY_OUT = Path("data/M07_summary_m3_judges_celeb.csv")
DEFAULT_DIAG_OUT = Path("data/M07_diagnostics_m3_judges_celeb.json")


def main():
    parser = argparse.ArgumentParser(
        description="M07: Fit Bayesian LMM for JUDGES outcome (score_z) with pro/season/celeb random effects (celeb RE ON by default)."
    )
    parser.add_argument("--panel", type=str, default=str(DEFAULT_PANEL), help="Input panel csv from M01.")
    parser.add_argument("--map", type=str, default=str(DEFAULT_MAP), help="Input keymap json from M01.")
    parser.add_argument("--out_trace", type=str, default=str(DEFAULT_TRACE_OUT), help="Output trace netcdf.")
    parser.add_argument("--out_summary", type=str, default=str(DEFAULT_SUMMARY_OUT), help="Output ArviZ summary csv.")
    parser.add_argument("--out_diag", type=str, default=str(DEFAULT_DIAG_OUT), help="Output diagnostics json.")
    parser.add_argument(
        "--include_celeb_re",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include celebrity random intercept. (default: True)",
    )
    parser.add_argument("--seed", type=int, default=20260201, help="Random seed.")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--target_accept", type=float, default=0.95)
    args = parser.parse_args()

    panel_path = Path(args.panel)
    map_path = Path(args.map)
    out_trace = Path(args.out_trace)
    out_summary = Path(args.out_summary)
    out_diag = Path(args.out_diag)

    if not panel_path.exists():
        raise FileNotFoundError(f"[M07] Panel not found: {panel_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"[M07] Map not found: {map_path}")

    out_trace.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_diag.parent.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    meta = json.load(open(map_path, "r", encoding="utf-8"))

    required_cols = [
        "y_j_score_z",
        "age_z", "week_z",
        "industry_id", "pro_id", "season_id", "celeb_id",
    ]
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"[M07] Missing required columns in panel: {missing}")

    before = len(panel)
    panel2 = panel.dropna(subset=required_cols).copy()
    dropped = before - len(panel2)

    for c in ["industry_id", "pro_id", "season_id", "celeb_id"]:
        panel2[c] = panel2[c].astype(int)

    y = panel2["y_j_score_z"].astype(float).to_numpy()
    age_z = panel2["age_z"].astype(float).to_numpy()
    week_z = panel2["week_z"].astype(float).to_numpy()
    industry_id = panel2["industry_id"].to_numpy()
    pro_id = panel2["pro_id"].to_numpy()
    season_id = panel2["season_id"].to_numpy()
    celeb_id = panel2["celeb_id"].to_numpy()

    pro_names = meta["id_maps"]["pro"]["names"]
    ind_names = meta["id_maps"]["industry"]["names"]
    season_vals = meta["id_maps"]["season"]["values"]
    celeb_names = meta["id_maps"]["celebrity"]["names"]

    max_ok = {
        "pro_id_max": int(pro_id.max()),
        "pro_id_dim": len(pro_names),
        "industry_id_max": int(industry_id.max()),
        "industry_dim": len(ind_names),
        "season_id_max": int(season_id.max()),
        "season_dim": len(season_vals),
        "celeb_id_max": int(celeb_id.max()),
        "celeb_dim": len(celeb_names),
    }

    # -----------------------------
    # PRINT QC BLOCK
    # -----------------------------
    print("\n" + "=" * 90)
    print("M07 CHECK BLOCK START")
    print("=" * 90)
    print(f"[M07] panel_path = {panel_path}")
    print(f"[M07] map_path   = {map_path}")
    print(f"[M07] include_celeb_re = {bool(args.include_celeb_re)}")
    print(f"[M07] rows_before_drop = {before}")
    print(f"[M07] rows_after_drop  = {len(panel2)}")
    print(f"[M07] dropped_missing_required = {dropped}")

    print("\n[M07] DIMENSIONS (from meta):")
    print(f"  n_pro      = {len(pro_names)}")
    print(f"  n_industry = {len(ind_names)}")
    print(f"  n_season   = {len(season_vals)}")
    print(f"  n_celeb    = {len(celeb_names)}")

    print("\n[M07] MAX ID CHECKS:")
    for k, v in max_ok.items():
        print(f"  {k}: {v}")
    print("  NOTE: id_max must be < corresponding dim.")

    def _dist(name, arr):
        arr = np.asarray(arr)
        q = np.quantile(arr, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
        print(f"\n[M07] DIST {name}:")
        print(f"  n={arr.size} mean={arr.mean():.6f} sd={arr.std(ddof=0):.6f}")
        print(f"  min={q[0]:.6f} p01={q[1]:.6f} p05={q[2]:.6f} p50={q[3]:.6f} p95={q[4]:.6f} p99={q[5]:.6f} max={q[6]:.6f}")

    _dist("y_j_score_z", y)
    _dist("age_z", age_z)
    _dist("week_z", week_z)

    print("\n[M07] CATEGORY COUNTS (top 10):")
    print("  pro_id top10:", pd.Series(pro_id).value_counts().head(10).to_dict())
    print("  industry_id top10:", pd.Series(industry_id).value_counts().head(10).to_dict())

    print("\nM07 CHECK BLOCK END")
    print("=" * 90 + "\n")

    # Hard fail if ids out of range
    assert pro_id.max() < len(pro_names), "[M07] pro_id out of range."
    assert industry_id.max() < len(ind_names), "[M07] industry_id out of range."
    assert season_id.max() < len(season_vals), "[M07] season_id out of range."
    assert celeb_id.max() < len(celeb_names), "[M07] celeb_id out of range."

    coords = {
        "obs": np.arange(len(panel2)),
        "pro": np.arange(len(pro_names)),
        "industry": np.arange(len(ind_names)),
        "season": np.arange(len(season_vals)),
        "celeb": np.arange(len(celeb_names)),
    }

    with pm.Model(coords=coords) as model:
        y_obs = pm.Data("y_obs", y, dims="obs")
        age = pm.Data("age_z", age_z, dims="obs")
        week = pm.Data("week_z", week_z, dims="obs")

        pro_idx = pm.Data("pro_id", pro_id, dims="obs")
        ind_idx = pm.Data("industry_id", industry_id, dims="obs")
        season_idx = pm.Data("season_id", season_id, dims="obs")
        celeb_idx = pm.Data("celeb_id", celeb_id, dims="obs")

        beta0 = pm.Normal("beta0", mu=0.0, sigma=1.0)
        beta_age = pm.Normal("beta_age", mu=0.0, sigma=1.0)
        beta_week = pm.Normal("beta_week", mu=0.0, sigma=1.0)

        beta_ind_raw = pm.Normal("beta_ind_raw", mu=0.0, sigma=1.0, dims="industry")
        beta_ind = pm.Deterministic("beta_ind", beta_ind_raw - pm.math.mean(beta_ind_raw), dims="industry")

        sigma_eps = pm.HalfNormal("sigma_eps", sigma=1.0)

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
            + beta_age * age
            + beta_week * week
            + beta_ind[ind_idx]
            + a_pro[pro_idx]
            + a_season[season_idx]
            + a_celeb[celeb_idx]
        )

        pm.Normal("likelihood", mu=mu, sigma=sigma_eps, observed=y_obs, dims="obs")

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.seed,
            return_inferencedata=True,
            progressbar=True,
        )

        ppc = pm.sample_posterior_predictive(
            idata, var_names=["likelihood"], random_seed=args.seed, progressbar=True
        )
        idata.extend(ppc)

    print(f"[M07] Saving trace to: {out_trace}")
    idata.to_netcdf(out_trace, engine="h5netcdf")

    summary_vars: List[str] = ["beta0", "beta_age", "beta_week", "sigma_eps", "sigma_pro", "sigma_season"]
    if args.include_celeb_re:
        summary_vars.append("sigma_celeb")

    summ = az.summary(idata, var_names=summary_vars, round_to=6)
    summ.to_csv(out_summary)
    print(f"[M07] Saved summary: {out_summary}")

    div = int(idata.sample_stats["diverging"].sum().values) if "diverging" in idata.sample_stats else None
    rhat = az.rhat(idata, var_names=summary_vars)
    ess = az.ess(idata, var_names=summary_vars)

    rhat_max = float(np.nanmax(rhat.to_array().values))
    ess_min = float(np.nanmin(ess.to_array().values))

    qc = {
        "n_obs_used": int(len(panel2)),
        "divergences_total": div,
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "max_id_check": max_ok,
        "include_celeb_re": bool(args.include_celeb_re),
        "sigma_celeb_fixed_if_disabled": sigma_celeb_fixed,
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
        },
        "qc": qc,
    }

    with open(out_diag, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f"[M07] Saved diagnostics: {out_diag}")

    print("\n" + "=" * 90)
    print("M07 POST-SAMPLING CHECK BLOCK START")
    print("=" * 90)
    print("[M07] Summary (key vars):")
    print(summ.to_string())
    print("\n[M07] Diagnostics:")
    print(json.dumps(qc, indent=2))
    print("M07 POST-SAMPLING CHECK BLOCK END")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
