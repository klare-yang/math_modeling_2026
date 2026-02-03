#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M10: Postprocess M09 fan latent interval model

Outputs:
- Fixed effects table (summary)
- Pro random effects table
- Structural variance decomposition
- Observation-level posterior means for mu and latent y (truncated normal mean), for downstream hazard model / interpretation.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import arviz as az
from scipy.stats import norm


DEFAULT_PANEL = Path("data/M01_panel_m3.csv")
DEFAULT_MAP = Path("data/M01_keymap_m3.json")
DEFAULT_TRACE = Path("data/M09_trace_m3_fans_latent.nc")

OUT_FIXED = Path("data/M10_latent_fixed_effects_fans.csv")
OUT_PRO = Path("data/M10_latent_random_effects_pro_fans.csv")
OUT_VAR = Path("data/M10_latent_variance_decomp.json")
OUT_OBS = Path("data/M10_fans_latent_obs_posterior.csv")

FIG_DIR = Path("fig")


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def clip_prob(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def hdi94(x: np.ndarray) -> Tuple[float, float]:
    h = az.hdi(np.asarray(x).reshape(-1), hdi_prob=0.94)
    return float(h[0]), float(h[1])


def summarize(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x).reshape(-1)
    lo, hi = hdi94(x)
    return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=0)), "hdi_3": lo, "hdi_97": hi}


def select_draws(n: int, max_draws: int) -> np.ndarray:
    if max_draws is None or max_draws <= 0:
        return np.arange(n, dtype=int)
    m = min(n, int(max_draws))
    if m >= n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=m, dtype=int)


def truncated_normal_mean_var(mu, sigma, L, U):
    """
    Vectorized truncated Normal moments for each element (mu,sigma,L,U) on same shape.
    """
    a = (L - mu) / sigma
    b = (U - mu) / sigma
    Phi_a = norm.cdf(a)
    Phi_b = norm.cdf(b)
    Z = np.maximum(Phi_b - Phi_a, 1e-12)

    phi_a = norm.pdf(a)
    phi_b = norm.pdf(b)

    mean = mu + sigma * (phi_a - phi_b) / Z
    var = sigma**2 * (1.0 + (a * phi_a - b * phi_b) / Z - ((phi_a - phi_b) / Z) ** 2)
    var = np.maximum(var, 1e-12)
    return mean, var


def main():
    parser = argparse.ArgumentParser(description="M10: Postprocess fan latent interval model (M09).")
    parser.add_argument("--panel", type=str, default=str(DEFAULT_PANEL))
    parser.add_argument("--map", type=str, default=str(DEFAULT_MAP))
    parser.add_argument("--trace", type=str, default=str(DEFAULT_TRACE))
    parser.add_argument("--eps_clip", type=float, default=1e-6, help="Clip for CI->logit conversion.")
    parser.add_argument("--max_draws_obs", type=int, default=400, help="Max posterior draws for obs-level posterior means.")
    args = parser.parse_args()

    panel_path = Path(args.panel)
    map_path = Path(args.map)
    trace_path = Path(args.trace)

    for p in [panel_path, map_path, trace_path]:
        if not p.exists():
            raise FileNotFoundError(f"[M10] Missing input: {p}")

    OUT_FIXED.parent.mkdir(parents=True, exist_ok=True)
    OUT_PRO.parent.mkdir(parents=True, exist_ok=True)
    OUT_VAR.parent.mkdir(parents=True, exist_ok=True)
    OUT_OBS.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    meta = json.load(open(map_path, "r", encoding="utf-8"))
    ind_names = meta["id_maps"]["industry"]["names"]
    pro_names = meta["id_maps"]["pro"]["names"]

    # interval bounds in logit space
    ci_lo = clip_prob(panel["share_ci2_5"].astype(float).to_numpy(), args.eps_clip)
    ci_hi = clip_prob(panel["share_ci97_5"].astype(float).to_numpy(), args.eps_clip)
    swap = ci_lo > ci_hi
    if np.any(swap):
        tmp = ci_lo[swap].copy()
        ci_lo[swap] = ci_hi[swap]
        ci_hi[swap] = tmp

    L = logit(ci_lo)
    U = logit(ci_hi)

    # load posterior
    idata = az.from_netcdf(trace_path)
    post = idata.posterior.stack(sample=("chain", "draw"))

    # fixed effects summary
    var_names = ["beta0", "beta_age", "beta_week", "sigma_eps", "sigma_pro", "sigma_season"]
    if "sigma_celeb" in post:
        var_names.append("sigma_celeb")

    summ = az.summary(idata, var_names=var_names + ["beta_ind"], round_to=6)
    summ.to_csv(OUT_FIXED)
    print(f"[M10] Saved fixed effects summary: {OUT_FIXED}")

    # pro random effects
    a_pro = post["a_pro"].values  # shape (n_pro, n_samples)
    # ensure orientation (samples, n_pro)
    if a_pro.shape[0] == len(pro_names):
        a_pro = a_pro.T
    elif a_pro.shape[-1] == len(pro_names):
        # already (samples, n_pro)
        pass
    else:
        raise ValueError(f"[M10] Unexpected a_pro shape: {a_pro.shape}")

    pro_rows = []
    for k, name in enumerate(pro_names):
        draws = a_pro[:, k]
        s = summarize(draws)
        pro_rows.append({"pro_id": k, "pro_name": name, **s})
    df_pro = pd.DataFrame(pro_rows).sort_values("mean", ascending=False)
    df_pro.to_csv(OUT_PRO, index=False)
    print(f"[M10] Saved pro random effects: {OUT_PRO}")

    # structural variance decomposition (posterior draws of sigmas)
    sigma_eps = post["sigma_eps"].values.reshape(-1)
    sigma_pro = post["sigma_pro"].values.reshape(-1)
    sigma_season = post["sigma_season"].values.reshape(-1)
    sigma_celeb = post["sigma_celeb"].values.reshape(-1) if "sigma_celeb" in post else np.zeros_like(sigma_eps)

    total = sigma_eps**2 + sigma_pro**2 + sigma_season**2 + sigma_celeb**2
    shares = {
        "eps": (sigma_eps**2) / total,
        "pro": (sigma_pro**2) / total,
        "season": (sigma_season**2) / total,
        "celeb": (sigma_celeb**2) / total,
    }

    var_payload: Dict[str, Any] = {
        "inputs": {"trace": str(trace_path), "panel": str(panel_path), "map": str(map_path)},
        "structural_shares": {k: summarize(v) for k, v in shares.items()},
        "sigmas": {
            "sigma_eps": summarize(sigma_eps),
            "sigma_pro": summarize(sigma_pro),
            "sigma_season": summarize(sigma_season),
            "sigma_celeb": summarize(sigma_celeb),
        },
    }
    with open(OUT_VAR, "w", encoding="utf-8") as f:
        json.dump(var_payload, f, ensure_ascii=False, indent=2)
    print(f"[M10] Saved variance decomposition: {OUT_VAR}")

    # Observation-level posterior mean for mu and truncated latent y
    # Reconstruct mu for each draw subset
    age = panel["age_z"].astype(float).to_numpy()
    week = panel["week_z"].astype(float).to_numpy()
    ind = panel["industry_id"].astype(int).to_numpy()
    pro = panel["pro_id"].astype(int).to_numpy()
    season = panel["season_id"].astype(int).to_numpy()
    celeb = panel["celeb_id"].astype(int).to_numpy()

    # posterior arrays
    beta0 = post["beta0"].values.reshape(-1)
    beta_age = post["beta_age"].values.reshape(-1)
    beta_week = post["beta_week"].values.reshape(-1)
    beta_ind = post["beta_ind"].values  # (industry, samples) or (samples, industry)
    if beta_ind.shape[0] == len(ind_names):
        beta_ind = beta_ind.T  # (samples, industry)
    elif beta_ind.shape[1] == len(ind_names):
        pass
    else:
        raise ValueError(f"[M10] Unexpected beta_ind shape: {beta_ind.shape}")

    a_season = post["a_season"].values
    if a_season.shape[0] == len(meta["id_maps"]["season"]["values"]):
        a_season = a_season.T  # (samples, season)
    a_celeb = post["a_celeb"].values
    if a_celeb.shape[0] == len(meta["id_maps"]["celebrity"]["names"]):
        a_celeb = a_celeb.T  # (samples, celeb)

    # select draws
    n_draws = beta0.shape[0]
    sel = select_draws(n_draws, args.max_draws_obs)
    S = len(sel)
    N = len(panel)

    # allocate accumulators
    mu_mean = np.zeros(N)
    mu_m2 = np.zeros(N)

    y_mean = np.zeros(N)
    y_m2 = np.zeros(N)

    # compute per selected draw
    for t, idx in enumerate(sel):
        mu = (
            beta0[idx]
            + beta_age[idx] * age
            + beta_week[idx] * week
            + beta_ind[idx, ind]
            + a_pro[idx, pro]
            + a_season[idx, season]
            + a_celeb[idx, celeb]
        )
        sig = float(sigma_eps[idx])
        y_draw_mean, y_draw_var = truncated_normal_mean_var(mu, sig, L, U)

        # online mean/var update
        # mu
        delta = mu - mu_mean
        mu_mean += delta / (t + 1)
        mu_m2 += delta * (mu - mu_mean)

        # y
        delta2 = y_draw_mean - y_mean
        y_mean += delta2 / (t + 1)
        y_m2 += delta2 * (y_draw_mean - y_mean)

    mu_sd = np.sqrt(np.maximum(mu_m2 / max(S - 1, 1), 0.0))
    y_sd = np.sqrt(np.maximum(y_m2 / max(S - 1, 1), 0.0))

    df_obs = pd.DataFrame({
        "event_id": panel["event_id"],
        "season": panel["season"],
        "season_id": panel["season_id"],
        "week": panel["week"],
        "contestant_key": panel["contestant_key"],
        "celeb_id": panel["celeb_id"],
        "pro_id": panel["pro_id"],
        "industry_id": panel["industry_id"],
        "age_z": panel["age_z"],
        "week_z": panel["week_z"],
        "eliminated_flag": panel["eliminated_flag"],
        "L_logit": L,
        "U_logit": U,
        "mu_mean": mu_mean,
        "mu_sd": mu_sd,
        "y_latent_mean": y_mean,
        "y_latent_sd": y_sd,
        "share_latent_mean": 1.0 / (1.0 + np.exp(-y_mean)),
    })
    df_obs.to_csv(OUT_OBS, index=False)
    print(f"[M10] Saved obs-level posterior means: {OUT_OBS}")

    # CHECK BLOCK
    print("\n" + "=" * 90)
    print("M10 CHECK BLOCK START")
    print("=" * 90)
    print("[M10] Inputs:")
    print(f"  trace = {trace_path}")
    print(f"  panel = {panel_path}")
    print(f"  map   = {map_path}")
    print("\n[M10] Outputs:")
    print(f"  fixed_summary = {OUT_FIXED}")
    print(f"  pro_effects   = {OUT_PRO}")
    print(f"  var_decomp    = {OUT_VAR}")
    print(f"  obs_posterior = {OUT_OBS}")
    print("\n[M10] Structural shares (mean + 94% HDI):")
    for k in ["pro", "season", "celeb", "eps"]:
        v = var_payload["structural_shares"][k]
        print(f"  {k:>6s} mean={v['mean']:.4f} hdi=[{v['hdi_3']:.4f},{v['hdi_97']:.4f}]")
    print("\n[M10] Obs posterior quick stats:")
    print(f"  y_latent_mean: mean={df_obs['y_latent_mean'].mean():.6f} sd={df_obs['y_latent_mean'].std(ddof=0):.6f}")
    print(f"  share_latent_mean: mean={df_obs['share_latent_mean'].mean():.6f} sd={df_obs['share_latent_mean'].std(ddof=0):.6f}")
    print("\nM10 CHECK BLOCK END")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
