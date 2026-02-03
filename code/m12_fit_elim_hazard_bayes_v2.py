#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
M12 (v2): Bayesian hierarchical logistic hazard model for elimination probability.

Why v2:
- Removes scikit-learn dependency (common failure in contest venvs).
- Computes posterior mean probability from posterior deterministic p (no posterior_predictive p hack).
- More defensive about required columns; can (re)construct score_c / fan_c if absent.
- Adds optional switches to disable celeb RE or performance covariates for ablation / debugging.

Inputs (default):
- data/M11_elim_panel.csv
- data/M01_keymap_m3.json

Outputs (default):
- data/M12_trace_elim_hazard.nc
- data/M12_summary_elim_hazard.csv
- data/M12_diagnostics_elim_hazard.json
- data/M12_metrics_elim_hazard.json
- data/M12_predprob_obs.csv
- fig/M12_roc.png
- fig/M12_calibration.png

This script is designed to be copy-paste runnable inside the user's repo.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az
import pytensor.tensor as pt

import matplotlib.pyplot as plt


DEFAULT_ELIM = Path("data/M11_elim_panel.csv")
DEFAULT_MAP = Path("data/M01_keymap_m3.json")

OUT_TRACE = Path("data/M12_trace_elim_hazard.nc")
OUT_SUMMARY = Path("data/M12_summary_elim_hazard.csv")
OUT_DIAG = Path("data/M12_diagnostics_elim_hazard.json")
OUT_METRICS = Path("data/M12_metrics_elim_hazard.json")
OUT_PRED = Path("data/M12_predprob_obs.csv")

FIG_DIR = Path("fig")


def _dist(name: str, arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    q = np.quantile(arr, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    stats = {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=0)),
        "min": float(q[0]),
        "p01": float(q[1]),
        "p05": float(q[2]),
        "p50": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
        "max": float(q[6]),
    }
    print(f"\n[M12] DIST {name}:")
    print(f"  n={int(stats['n'])} mean={stats['mean']:.6f} sd={stats['sd']:.6f}")
    print(
        f"  min={stats['min']:.6f} p01={stats['p01']:.6f} p05={stats['p05']:.6f} "
        f"p50={stats['p50']:.6f} p95={stats['p95']:.6f} p99={stats['p99']:.6f} max={stats['max']:.6f}"
    )
    return stats


def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via rank statistic (handles ties with average rank using pandas)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.min() == y_true.max():
        return float("nan")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()  # 1..n
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_curve_simple(y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 200) -> Dict[str, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    # thresholds on quantiles for a smooth curve
    qs = np.linspace(1.0, 0.0, n_thresholds)
    thr = np.quantile(y_score, qs)
    thr = np.unique(thr)[::-1]

    tpr_list = []
    fpr_list = []
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    for t in thr:
        y_pred = (y_score >= t).astype(int)
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        tpr_list.append(TP / P if P > 0 else np.nan)
        fpr_list.append(FP / N if N > 0 else np.nan)
    return {"thresholds": thr, "tpr": np.asarray(tpr_list), "fpr": np.asarray(fpr_list)}


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    p = np.asarray(p).astype(float)
    return float(np.mean((p - y_true) ** 2))


def main():
    parser = argparse.ArgumentParser(description="M12(v2): Bayesian hierarchical logistic hazard model for elimination.")
    parser.add_argument("--elim_panel", type=str, default=str(DEFAULT_ELIM))
    parser.add_argument("--map", type=str, default=str(DEFAULT_MAP))

    parser.add_argument("--out_trace", type=str, default=str(OUT_TRACE))
    parser.add_argument("--out_summary", type=str, default=str(OUT_SUMMARY))
    parser.add_argument("--out_diag", type=str, default=str(OUT_DIAG))
    parser.add_argument("--out_metrics", type=str, default=str(OUT_METRICS))
    parser.add_argument("--out_pred", type=str, default=str(OUT_PRED))

    parser.add_argument("--include_perf_covariates", action=argparse.BooleanOptionalAction, default=True,
                        help="Include score_c and fan_c covariates (default True).")
    parser.add_argument("--include_celeb_re", action=argparse.BooleanOptionalAction, default=True,
                        help="Include celebrity random intercept (default True).")

    parser.add_argument("--fan_col", type=str, default="fan_c",
                        help="Fan column to use. Default fan_c. If missing, will try to reconstruct.")
    parser.add_argument("--score_col", type=str, default="score_c",
                        help="Score column to use. Default score_c. If missing, will try to reconstruct.")

    parser.add_argument("--seed", type=int, default=20260201)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--target_accept", type=float, default=0.95)
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: draws=800 tune=800 chains=4 (useful for debugging).")

    args = parser.parse_args()

    if args.fast:
        args.draws = 800
        args.tune = 800

    elim_path = Path(args.elim_panel)
    map_path = Path(args.map)
    for p in [elim_path, map_path]:
        if not p.exists():
            raise FileNotFoundError(f"[M12] Missing input: {p}")

    # ensure output dirs
    for p in [Path(args.out_trace), Path(args.out_summary), Path(args.out_diag), Path(args.out_metrics), Path(args.out_pred)]:
        p.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(elim_path)
    meta = json.load(open(map_path, "r", encoding="utf-8"))

    # base required columns
    required_base = ["eliminated_flag", "age_z", "week_z", "industry_id", "pro_id", "season_id", "celeb_id"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"[M12] Missing required base columns: {missing_base}")

    # attempt to reconstruct score/fan if requested but missing
    if args.include_perf_covariates:
        # score
        if args.score_col not in df.columns:
            if "y_j_score_z" in df.columns and "event_id" in df.columns:
                df[args.score_col] = df["y_j_score_z"] - df.groupby("event_id")["y_j_score_z"].transform("mean")
            else:
                raise ValueError(f"[M12] include_perf_covariates=True but {args.score_col} missing and cannot reconstruct")

        # fan
        if args.fan_col not in df.columns:
            # try common candidates
            candidates = ["fan_c", "y_latent_mean", "y_f_logit_share_mean"]
            src = None
            for cand in candidates:
                if cand in df.columns:
                    src = cand
                    break
            if src is None or "event_id" not in df.columns:
                raise ValueError(f"[M12] include_perf_covariates=True but {args.fan_col} missing and cannot reconstruct")
            df[args.fan_col] = df[src] - df.groupby("event_id")[src].transform("mean")

    # drop missing rows
    required = required_base.copy()
    if args.include_perf_covariates:
        required += [args.score_col, args.fan_col]

    before = len(df)
    df2 = df.dropna(subset=required).copy()
    dropped = before - len(df2)

    # cast ids
    for c in ["industry_id", "pro_id", "season_id", "celeb_id"]:
        df2[c] = df2[c].astype(int)

    # arrays
    y = df2["eliminated_flag"].astype(int).to_numpy()
    age = df2["age_z"].astype(float).to_numpy()
    week = df2["week_z"].astype(float).to_numpy()
    ind = df2["industry_id"].to_numpy()
    pro = df2["pro_id"].to_numpy()
    season = df2["season_id"].to_numpy()
    celeb = df2["celeb_id"].to_numpy()

    score = df2[args.score_col].astype(float).to_numpy() if args.include_perf_covariates else None
    fan = df2[args.fan_col].astype(float).to_numpy() if args.include_perf_covariates else None

    # dims
    n_pro = len(meta["id_maps"]["pro"]["names"])
    n_ind = len(meta["id_maps"]["industry"]["names"])
    n_season = len(meta["id_maps"]["season"]["values"])
    n_celeb = len(meta["id_maps"]["celebrity"]["names"])

    max_ok = {
        "pro_id_max": int(pro.max()),
        "pro_id_dim": n_pro,
        "industry_id_max": int(ind.max()),
        "industry_dim": n_ind,
        "season_id_max": int(season.max()),
        "season_dim": n_season,
        "celeb_id_max": int(celeb.max()),
        "celeb_dim": n_celeb,
    }

    # CHECK BLOCK
    print("\n" + "=" * 90)
    print("M12 CHECK BLOCK START")
    print("=" * 90)
    print(f"[M12] elim_panel = {elim_path}")
    print(f"[M12] map        = {map_path}")
    print(f"[M12] include_perf_covariates = {bool(args.include_perf_covariates)}")
    print(f"[M12] include_celeb_re        = {bool(args.include_celeb_re)}")
    if args.include_perf_covariates:
        print(f"[M12] score_col = {args.score_col}")
        print(f"[M12] fan_col   = {args.fan_col}")
    print(f"[M12] rows_before_drop = {before}")
    print(f"[M12] rows_after_drop  = {len(df2)}")
    print(f"[M12] dropped_missing_required = {dropped}")

    print("\n[M12] ID MAX CHECKS:")
    for k, v in max_ok.items():
        print(f"  {k}: {v}")
    print("  NOTE: id_max must be < corresponding dim.")

    print("\n[M12] Outcome prevalence:")
    print(f"  eliminated_rate = {y.mean():.6f} ({y.sum()}/{len(y)})")

    _dist("age_z", age)
    _dist("week_z", week)
    if args.include_perf_covariates:
        _dist(args.score_col, score)
        _dist(args.fan_col, fan)

    print("\nM12 CHECK BLOCK END")
    print("=" * 90 + "\n")

    assert pro.max() < n_pro
    assert ind.max() < n_ind
    assert season.max() < n_season
    assert celeb.max() < n_celeb

    coords = {
        "obs": np.arange(len(df2)),
        "pro": np.arange(n_pro),
        "industry": np.arange(n_ind),
        "season": np.arange(n_season),
        "celeb": np.arange(n_celeb),
    }

    with pm.Model(coords=coords) as model:
        y_obs = pm.Data("y", y, dims="obs")
        age_d = pm.Data("age_z", age, dims="obs")
        week_d = pm.Data("week_z", week, dims="obs")
        ind_d = pm.Data("industry_id", ind, dims="obs")
        pro_d = pm.Data("pro_id", pro, dims="obs")
        season_d = pm.Data("season_id", season, dims="obs")
        celeb_d = pm.Data("celeb_id", celeb, dims="obs")

        if args.include_perf_covariates:
            score_d = pm.Data(args.score_col, score, dims="obs")
            fan_d = pm.Data(args.fan_col, fan, dims="obs")

        # fixed effects
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.0)
        beta_age = pm.Normal("beta_age", mu=0.0, sigma=1.0)
        beta_week = pm.Normal("beta_week", mu=0.0, sigma=1.0)
        if args.include_perf_covariates:
            beta_score = pm.Normal("beta_score", mu=0.0, sigma=1.0)
            beta_fan = pm.Normal("beta_fan", mu=0.0, sigma=1.0)

        beta_ind_raw = pm.Normal("beta_ind_raw", mu=0.0, sigma=1.0, dims="industry")
        beta_ind = pm.Deterministic("beta_ind", beta_ind_raw - pt.mean(beta_ind_raw), dims="industry")

        # random intercepts
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
        else:
            # deterministic zeros with correct dim
            a_celeb = pm.Deterministic("a_celeb", pt.zeros((n_celeb,)), dims="celeb")
            sigma_celeb = pm.Deterministic("sigma_celeb", pt.as_tensor_variable(0.0))

        eta = (
            alpha0
            + beta_age * age_d
            + beta_week * week_d
            + beta_ind[ind_d]
            + a_pro[pro_d]
            + a_season[season_d]
            + a_celeb[celeb_d]
        )
        if args.include_perf_covariates:
            eta = eta + beta_score * score_d + beta_fan * fan_d

        # deterministic prob BEFORE sampling
        p = pm.Deterministic("p", pm.math.sigmoid(eta), dims="obs")

        pm.Bernoulli("likelihood", logit_p=eta, observed=y_obs, dims="obs")

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.seed,
            return_inferencedata=True,
            progressbar=True,
        )

    print(f"[M12] Saving trace to: {args.out_trace}")
    idata.to_netcdf(args.out_trace, engine="h5netcdf")

    # summary vars
    summary_vars: List[str] = ["alpha0", "beta_age", "beta_week", "sigma_pro", "sigma_season", "sigma_celeb"]
    if args.include_perf_covariates:
        summary_vars += ["beta_score", "beta_fan"]

    summ = az.summary(idata, var_names=summary_vars + ["beta_ind"], round_to=6)
    summ.to_csv(args.out_summary)
    print(f"[M12] Saved summary: {args.out_summary}")

    # posterior mean p per observation
    p_post = idata.posterior["p"].mean(dim=("chain", "draw")).values.reshape(-1)

    # metrics
    auc = auc_rank(y, p_post)
    brier = brier_score(y, p_post)

    # calibration
    bins = np.linspace(0, 1, 11)
    bin_id = np.digitize(p_post, bins) - 1
    cal = []
    for b in range(10):
        m = bin_id == b
        if m.sum() == 0:
            continue
        cal.append({
            "bin": int(b),
            "p_mean": float(p_post[m].mean()),
            "y_rate": float(y[m].mean()),
            "n": int(m.sum()),
        })

    metrics = {
        "n_obs": int(len(y)),
        "eliminated_rate": float(y.mean()),
        "auc": float(auc),
        "brier": float(brier),
        "calibration_bins": cal,
        "include_perf_covariates": bool(args.include_perf_covariates),
        "include_celeb_re": bool(args.include_celeb_re),
        "fan_col": args.fan_col if args.include_perf_covariates else None,
        "score_col": args.score_col if args.include_perf_covariates else None,
        "config": {
            "seed": int(args.seed),
            "chains": int(args.chains),
            "draws": int(args.draws),
            "tune": int(args.tune),
            "target_accept": float(args.target_accept),
            "fast": bool(args.fast),
        },
    }
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # diagnostics
    div = int(idata.sample_stats["diverging"].sum().values) if "diverging" in idata.sample_stats else None
    rhat = az.rhat(idata, var_names=summary_vars)
    ess = az.ess(idata, var_names=summary_vars)
    qc = {
        "n_obs_used": int(len(y)),
        "divergences_total": div,
        "rhat_max": float(np.nanmax(rhat.to_array().values)),
        "ess_min": float(np.nanmin(ess.to_array().values)),
        "max_id_check": max_ok,
    }

    diag: Dict[str, Any] = {
        "inputs": {"elim_panel": str(elim_path), "map": str(map_path)},
        "outputs": {
            "trace": str(args.out_trace),
            "summary": str(args.out_summary),
            "metrics": str(args.out_metrics),
            "pred": str(args.out_pred),
        },
        "qc": qc,
        "notes": {
            "auc_method": "rank (pandas average-rank ties)",
            "roc_method": "quantile thresholds",
            "reconstructed_cols": {
                "score_col": (args.score_col not in df.columns) if args.include_perf_covariates else None,
                "fan_col": (args.fan_col not in df.columns) if args.include_perf_covariates else None,
            },
        },
    }
    with open(args.out_diag, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    # save per-observation predicted probability with key columns if available
    pred_df = pd.DataFrame({"p_mean": p_post, "eliminated_flag": y})
    for k in ["event_id", "contestant_key", "season_id", "week", "pro_id", "celeb_id"]:
        if k in df2.columns:
            pred_df[k] = df2[k].to_numpy()
    pred_df.to_csv(args.out_pred, index=False)

    # plots
    roc = roc_curve_simple(y, p_post, n_thresholds=200)
    roc_path = FIG_DIR / "M12_roc.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(roc["fpr"], roc["tpr"])
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (posterior mean p)")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)

    cal_path = FIG_DIR / "M12_calibration.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if len(cal) > 0:
        ax.plot([c["p_mean"] for c in cal], [c["y_rate"] for c in cal], marker="o")
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("Predicted p (bin mean)")
    ax.set_ylabel("Observed rate (bin)")
    ax.set_title("Calibration")
    fig.tight_layout()
    fig.savefig(cal_path, dpi=200)
    plt.close(fig)

    # POST CHECK BLOCK
    print("\n" + "=" * 90)
    print("M12 POST-SAMPLING CHECK BLOCK START")
    print("=" * 90)
    print("[M12] Summary (key vars):")
    print(az.summary(idata, var_names=summary_vars, round_to=6).to_string())
    print("\n[M12] QC:")
    print(json.dumps(qc, indent=2))
    print("\n[M12] Metrics:")
    keep = {k: metrics[k] for k in ["n_obs", "eliminated_rate", "auc", "brier", "include_perf_covariates", "include_celeb_re"]}
    keep["fan_col"] = metrics.get("fan_col")
    keep["score_col"] = metrics.get("score_col")
    print(json.dumps(keep, indent=2))
    print(f"\n[M12] Saved pred probs: {args.out_pred}")
    print(f"[M12] Figures: {roc_path} , {cal_path}")
    print("M12 POST-SAMPLING CHECK BLOCK END")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
