import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt


DEFAULT_PANEL = Path("data/M01_panel_m3.csv")
DEFAULT_MAP = Path("data/M01_keymap_m3.json")

DEFAULT_TRACE_F = Path("data/M06_trace_m3_fans_celeb.nc")
DEFAULT_TRACE_J = Path("data/M07_trace_m3_judges_celeb.nc")

OUT_EFF_F = Path("data/M08_effectsizes_fans.csv")
OUT_EFF_J = Path("data/M08_effectsizes_judges.csv")
OUT_VAR_JSON = Path("data/M08_varshare_fixed_random.json")

FIG_DIR = Path("fig")


def ensure_exists(*paths: Path):
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"[M08] Missing input: {p}")


def hdi94(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return float("nan"), float("nan")
    h = az.hdi(x, hdi_prob=0.94)
    return float(h[0]), float(h[1])


def summarize_1d(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x).reshape(-1)
    lo, hi = hdi94(x)
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=0)),
        "hdi_3": lo,
        "hdi_97": hi,
    }


def stack_samples(idata) -> Any:
    post = idata.posterior
    if "chain" in post.dims and "draw" in post.dims:
        return post.stack(sample=("chain", "draw"))
    return post


def get_1d(post_s, name: str) -> np.ndarray:
    if name not in post_s:
        raise KeyError(f"[M08] posterior var not found: {name}")
    arr = np.asarray(post_s[name].values)
    return arr.reshape(-1)


def get_2d(post_s, name: str, dim0: str) -> np.ndarray:
    if name not in post_s:
        raise KeyError(f"[M08] posterior var not found: {name}")
    da = post_s[name]
    if dim0 in da.dims and "sample" in da.dims:
        return np.asarray(da.transpose("sample", dim0).values)
    arr = np.asarray(da.values)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"[M08] Unexpected shape for {name}: {arr.shape}")


def select_samples(n: int, max_draws: int) -> np.ndarray:
    m = min(int(max_draws), int(n))
    if m <= 0:
        m = n
    if m >= n:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, num=m, dtype=int)
    return idx


def build_effect_table(post_s, sigma_eps: np.ndarray, ind_names: List[str], prefix: str) -> pd.DataFrame:
    rows = []

    def add(term: str, draws: np.ndarray, std_draws: np.ndarray = None):
        s = summarize_1d(draws)
        row = {
            "term": term,
            "mean": s["mean"],
            "sd": s["sd"],
            "hdi_3": s["hdi_3"],
            "hdi_97": s["hdi_97"],
        }
        if std_draws is not None:
            ss = summarize_1d(std_draws)
            row.update({
                "std_mean": ss["mean"],
                "std_sd": ss["sd"],
                "std_hdi_3": ss["hdi_3"],
                "std_hdi_97": ss["hdi_97"],
            })
        rows.append(row)

    beta0 = get_1d(post_s, "beta0")
    beta_age = get_1d(post_s, "beta_age")
    beta_week = get_1d(post_s, "beta_week")

    add(f"{prefix}:beta0", beta0, None)
    add(f"{prefix}:beta_age", beta_age, beta_age / sigma_eps)
    add(f"{prefix}:beta_week", beta_week, beta_week / sigma_eps)

    if "beta_ind" in post_s:
        beta_ind = get_2d(post_s, "beta_ind", "industry")  # (sample, n_ind)
        for k, nm in enumerate(ind_names):
            draws = beta_ind[:, k]
            add(f"{prefix}:beta_ind[{nm}]", draws, draws / sigma_eps)

        ind_sd = np.std(beta_ind, axis=1, ddof=0)
        add(f"{prefix}:industry_effect_sd", ind_sd, ind_sd / sigma_eps)

        ind_range = (np.max(beta_ind, axis=1) - np.min(beta_ind, axis=1))
        add(f"{prefix}:industry_effect_range", ind_range, ind_range / sigma_eps)

    add(f"{prefix}:sigma_eps", sigma_eps, None)
    if "sigma_pro" in post_s:
        add(f"{prefix}:sigma_pro", get_1d(post_s, "sigma_pro"), None)
    if "sigma_season" in post_s:
        add(f"{prefix}:sigma_season", get_1d(post_s, "sigma_season"), None)
    if "sigma_celeb" in post_s:
        add(f"{prefix}:sigma_celeb", get_1d(post_s, "sigma_celeb"), None)

    return pd.DataFrame(rows)


def compute_fixed_varshare(
    post_s,
    panel: pd.DataFrame,
    ind_names: List[str],
    max_draws: int,
    prefix: str,
) -> Dict[str, Any]:
    """
    ANOVA-style variance share:
      var_fixed = Var_over_obs( beta0 + beta_age*age + beta_week*week + beta_ind[ind] )
      total = var_fixed + sigma_eps^2 + sigma_pro^2 + sigma_season^2 (+ sigma_celeb^2)
      share_component = component / total
    """
    age = panel["age_z"].astype(float).to_numpy()
    week = panel["week_z"].astype(float).to_numpy()
    ind_idx = panel["industry_id"].astype(int).to_numpy()

    beta0 = get_1d(post_s, "beta0")
    beta_age = get_1d(post_s, "beta_age")
    beta_week = get_1d(post_s, "beta_week")
    sigma_eps = get_1d(post_s, "sigma_eps")
    sigma_pro = get_1d(post_s, "sigma_pro")
    sigma_season = get_1d(post_s, "sigma_season")
    sigma_celeb = get_1d(post_s, "sigma_celeb") if "sigma_celeb" in post_s else None

    beta_ind = get_2d(post_s, "beta_ind", "industry") if "beta_ind" in post_s else None
    if beta_ind is None:
        raise ValueError("[M08] beta_ind missing; expected deterministic 'beta_ind' in posterior.")

    n = beta0.shape[0]
    sel = select_samples(n, max_draws)

    b0 = beta0[sel]
    ba = beta_age[sel]
    bw = beta_week[sel]
    seps = sigma_eps[sel]
    spro = sigma_pro[sel]
    ssn = sigma_season[sel]
    scb = sigma_celeb[sel] if sigma_celeb is not None else None

    bind = beta_ind[sel, :]  # (S, K)

    mu_fixed = (
        b0[:, None]
        + ba[:, None] * age[None, :]
        + bw[:, None] * week[None, :]
        + bind[:, ind_idx]
    )
    var_fixed = np.var(mu_fixed, axis=1, ddof=0)

    total = var_fixed + seps**2 + spro**2 + ssn**2
    if scb is not None:
        total = total + scb**2

    shares = {
        "fixed": var_fixed / total,
        "eps": (seps**2) / total,
        "pro": (spro**2) / total,
        "season": (ssn**2) / total,
        "celeb": (scb**2) / total if scb is not None else np.zeros_like(var_fixed),
    }

    out = {
        "prefix": prefix,
        "n_draws_used": int(len(sel)),
        "var_fixed": summarize_1d(var_fixed),
        "shares": {k: summarize_1d(v) for k, v in shares.items()},
    }
    return out


def plot_varshare_bar(varshare_payload: Dict[str, Any], out_path: Path):
    """
    Accept either:
      (A) {"fans": {"shares": {...}}, "judges": {"shares": {...}}}
      (B) {"fans": {...shares...}, "judges": {...shares...}}
    """
    comps = ["fixed", "pro", "season", "celeb", "eps"]
    gnames = ["fans", "judges"]

    means = []
    for g in gnames:
        gobj = varshare_payload[g]
        shares = gobj["shares"] if isinstance(gobj, dict) and "shares" in gobj else gobj
        means.append([shares[c]["mean"] for c in comps])

    x = np.arange(len(comps))
    width = 0.35
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x - width/2, means[0], width, label="fans")
    ax.bar(x + width/2, means[1], width, label="judges")
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylim(0, 1)
    ax.set_title("Variance shares (ANOVA-style): fixed vs random vs residual")
    ax.set_ylabel("share")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="M08: Compute standardized effects (beta/sigma_eps) and ANOVA-style variance shares (fixed + random + residual)."
    )
    parser.add_argument("--panel", type=str, default=str(DEFAULT_PANEL))
    parser.add_argument("--map", type=str, default=str(DEFAULT_MAP))
    parser.add_argument("--trace_fans", type=str, default=str(DEFAULT_TRACE_F))
    parser.add_argument("--trace_judges", type=str, default=str(DEFAULT_TRACE_J))
    parser.add_argument("--out_eff_fans", type=str, default=str(OUT_EFF_F))
    parser.add_argument("--out_eff_judges", type=str, default=str(OUT_EFF_J))
    parser.add_argument("--out_var_json", type=str, default=str(OUT_VAR_JSON))
    parser.add_argument("--max_draws", type=int, default=1000, help="Max posterior draws used for variance-share computation.")
    args = parser.parse_args()

    panel_path = Path(args.panel)
    map_path = Path(args.map)
    trace_f = Path(args.trace_fans)
    trace_j = Path(args.trace_judges)

    out_eff_f = Path(args.out_eff_fans)
    out_eff_j = Path(args.out_eff_judges)
    out_var = Path(args.out_var_json)

    ensure_exists(panel_path, map_path, trace_f, trace_j)
    out_eff_f.parent.mkdir(parents=True, exist_ok=True)
    out_eff_j.parent.mkdir(parents=True, exist_ok=True)
    out_var.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    meta = json.load(open(map_path, "r", encoding="utf-8"))
    ind_names = meta["id_maps"]["industry"]["names"]

    mean_se2_fans = float(np.mean(panel["y_f_logit_share_se"].astype(float).to_numpy() ** 2)) if "y_f_logit_share_se" in panel.columns else float("nan")

    idata_f = az.from_netcdf(trace_f)
    idata_j = az.from_netcdf(trace_j)

    post_f = stack_samples(idata_f)
    post_j = stack_samples(idata_j)

    sigma_eps_f = get_1d(post_f, "sigma_eps")
    sigma_eps_j = get_1d(post_j, "sigma_eps")

    df_eff_f = build_effect_table(post_f, sigma_eps_f, ind_names, prefix="fans")
    df_eff_j = build_effect_table(post_j, sigma_eps_j, ind_names, prefix="judges")

    df_eff_f.to_csv(out_eff_f, index=False)
    df_eff_j.to_csv(out_eff_j, index=False)

    required_design = ["age_z", "week_z", "industry_id"]
    miss = [c for c in required_design if c not in panel.columns]
    if miss:
        raise ValueError(f"[M08] Missing design columns in panel: {miss}")

    var_f = compute_fixed_varshare(post_f, panel, ind_names, max_draws=args.max_draws, prefix="fans")
    var_j = compute_fixed_varshare(post_j, panel, ind_names, max_draws=args.max_draws, prefix="judges")

    payload: Dict[str, Any] = {
        "inputs": {
            "panel": str(panel_path),
            "map": str(map_path),
            "trace_fans": str(trace_f),
            "trace_judges": str(trace_j),
        },
        "outputs": {
            "effectsizes_fans": str(out_eff_f),
            "effectsizes_judges": str(out_eff_j),
            "varshare_json": str(out_var),
            "fig_varshare_bar": str(FIG_DIR / "M08_varshare_bar.png"),
        },
        "notes": {
            "standardized_effect": "beta / sigma_eps (signal-to-noise on outcome scale).",
            "variance_share": "ANOVA-style: Var(Xbeta) + sigma^2 components; fixed share depends on covariate distribution.",
            "fans_measurement_error_mean_se2": mean_se2_fans,
        },
        "fans": var_f,
        "judges": var_j,
    }

    with open(out_var, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # fixed: pass var_f/var_j (each has 'shares')
    plot_varshare_bar({"fans": var_f, "judges": var_j}, FIG_DIR / "M08_varshare_bar.png")

    print("\n" + "=" * 90)
    print("M08 CHECK BLOCK START")
    print("=" * 90)
    print("[M08] Inputs:")
    print(f"  panel        = {panel_path}")
    print(f"  map          = {map_path}")
    print(f"  trace_fans   = {trace_f}")
    print(f"  trace_judges = {trace_j}")
    print("\n[M08] Outputs:")
    print(f"  effects_fans   = {out_eff_f}")
    print(f"  effects_judges = {out_eff_j}")
    print(f"  varshare_json  = {out_var}")
    print(f"  fig_varshare   = {FIG_DIR / 'M08_varshare_bar.png'}")
    print("\n[M08] Fans measurement error (mean se^2):")
    print(f"  mean_se2_fans = {mean_se2_fans:.6f}")

    def _pick(df: pd.DataFrame, term: str) -> Dict[str, Any]:
        r = df[df["term"] == term]
        if r.empty:
            return {"term": term, "found": False}
        d = r.iloc[0].to_dict()
        d["found"] = True
        return d

    print("\n[M08] Standardized effects (beta/sigma_eps) — selected terms:")
    print("  fans   beta_age :", json.dumps(_pick(df_eff_f, "fans:beta_age"), ensure_ascii=False))
    print("  fans   beta_week:", json.dumps(_pick(df_eff_f, "fans:beta_week"), ensure_ascii=False))
    print("  judges beta_age :", json.dumps(_pick(df_eff_j, "judges:beta_age"), ensure_ascii=False))
    print("  judges beta_week:", json.dumps(_pick(df_eff_j, "judges:beta_week"), ensure_ascii=False))

    print("\n[M08] Variance shares (mean + 94% HDI):")
    for g in ["fans", "judges"]:
        print(f"  {g}:")
        for k, v in payload[g]["shares"].items():
            print(f"    {k:>6s} mean={v['mean']:.4f} hdi=[{v['hdi_3']:.4f},{v['hdi_97']:.4f}]")

    print("\nM08 CHECK BLOCK END")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
