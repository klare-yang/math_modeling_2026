# code/m04_m3_postprocess.py
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import arviz as az
import xarray as xr


DEFAULT_TRACE_FANS = Path("data/M02_trace_m3_fans.nc")
DEFAULT_TRACE_JUDGES = Path("data/M03_trace_m3_judges.nc")
DEFAULT_MAP = Path("data/M01_keymap_m3.json")
DEFAULT_PANEL = Path("data/M01_panel_m3.csv")

OUT_FIXED_FANS = Path("data/M04_m3_fixed_effects_fans.csv")
OUT_FIXED_JUDGES = Path("data/M04_m3_fixed_effects_judges.csv")
OUT_PRO_FANS = Path("data/M04_m3_random_effects_pro_fans.csv")
OUT_PRO_JUDGES = Path("data/M04_m3_random_effects_pro_judges.csv")
OUT_VAR = Path("data/M04_m3_variance_decomp.json")
OUT_PPC = Path("data/M04_m3_ppc_metrics.json")


def _extract_indexed_param(param: str) -> Tuple[str, Optional[int]]:
    m = re.match(r"^([^\[]+)\[(\d+)\]$", param)
    if not m:
        return param, None
    return m.group(1), int(m.group(2))


def _flatten_samples(x: xr.DataArray) -> np.ndarray:
    return x.stack(sample=("chain", "draw")).values


def variance_decomp_structural(idata: az.InferenceData) -> Dict[str, Any]:
    sig_eps = _flatten_samples(idata.posterior["sigma_eps"])
    sig_pro = _flatten_samples(idata.posterior["sigma_pro"])
    sig_sea = _flatten_samples(idata.posterior["sigma_season"])

    # sigma_celeb optional
    if "sigma_celeb" in idata.posterior:
        sc = idata.posterior["sigma_celeb"]
        if set(sc.dims) >= {"chain", "draw"}:
            sig_ce = _flatten_samples(sc)
        else:
            sig_ce = np.zeros_like(sig_eps)
    else:
        sig_ce = np.zeros_like(sig_eps)

    v_eps = sig_eps ** 2
    v_pro = sig_pro ** 2
    v_sea = sig_sea ** 2
    v_ce = sig_ce ** 2

    v_tot = v_eps + v_pro + v_sea + v_ce
    v_tot = np.where(v_tot <= 0, np.nan, v_tot)

    shares = {
        "pro": v_pro / v_tot,
        "season": v_sea / v_tot,
        "celeb": v_ce / v_tot,
        "eps": v_eps / v_tot,
    }

    def summ(arr: np.ndarray) -> Dict[str, float]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "hdi_3": float("nan"), "hdi_97": float("nan")}
        hdi = az.hdi(arr, hdi_prob=0.94)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "hdi_3": float(hdi[0]),
            "hdi_97": float(hdi[1]),
        }

    out = {k: summ(v) for k, v in shares.items()}
    out["n_samples"] = int(np.isfinite(list(shares.values())[0]).sum())
    return out


def variance_decomp_observed_noise_adjusted(idata: az.InferenceData, mean_se2: float) -> Dict[str, Any]:
    sig_eps = _flatten_samples(idata.posterior["sigma_eps"])
    sig_pro = _flatten_samples(idata.posterior["sigma_pro"])
    sig_sea = _flatten_samples(idata.posterior["sigma_season"])

    if "sigma_celeb" in idata.posterior:
        sc = idata.posterior["sigma_celeb"]
        if set(sc.dims) >= {"chain", "draw"}:
            sig_ce = _flatten_samples(sc)
        else:
            sig_ce = np.zeros_like(sig_eps)
    else:
        sig_ce = np.zeros_like(sig_eps)

    v_eps = sig_eps ** 2
    v_pro = sig_pro ** 2
    v_sea = sig_sea ** 2
    v_ce = sig_ce ** 2
    v_me = float(mean_se2)

    v_tot = v_eps + v_me + v_pro + v_sea + v_ce
    v_tot = np.where(v_tot <= 0, np.nan, v_tot)

    shares = {
        "pro": v_pro / v_tot,
        "season": v_sea / v_tot,
        "celeb": v_ce / v_tot,
        "eps": v_eps / v_tot,
        "measurement": np.full_like(v_eps, v_me) / v_tot,
    }

    def summ(arr: np.ndarray) -> Dict[str, float]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "hdi_3": float("nan"), "hdi_97": float("nan")}
        hdi = az.hdi(arr, hdi_prob=0.94)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "hdi_3": float(hdi[0]),
            "hdi_97": float(hdi[1]),
        }

    out = {k: summ(v) for k, v in shares.items()}
    out["mean_se2_used"] = float(mean_se2)
    out["n_samples"] = int(np.isfinite(list(shares.values())[0]).sum())
    return out


def ppc_metrics(idata: az.InferenceData, y_obs: np.ndarray, group_name: str) -> Dict[str, Any]:
    if not hasattr(idata, "posterior_predictive") or "likelihood" not in idata.posterior_predictive:
        return {"group": group_name, "ppc_available": False}

    yrep = idata.posterior_predictive["likelihood"]  # chain, draw, obs
    yrep_stack = yrep.stack(sample=("chain", "draw")).transpose("sample", "obs").values
    pred_mean = np.mean(yrep_stack, axis=0)

    resid = pred_mean - y_obs
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))

    if np.std(pred_mean) > 1e-12 and np.std(y_obs) > 1e-12:
        corr = float(np.corrcoef(pred_mean, y_obs)[0, 1])
    else:
        corr = float("nan")

    return {
        "group": group_name,
        "ppc_available": True,
        "rmse_predmean": rmse,
        "mae_predmean": mae,
        "corr_predmean_obs": corr,
        "n_obs": int(len(y_obs)),
    }


def build_pro_table(idata: az.InferenceData, pro_names: list, out_path: Path, group_label: str):
    s = az.summary(idata, var_names=["a_pro"], hdi_prob=0.94, round_to=None).reset_index().rename(columns={"index": "param"})
    s = s[s["param"].str.startswith("a_pro[")].copy()

    rows = []
    for _, r in s.iterrows():
        base, idx = _extract_indexed_param(r["param"])
        if idx is None or idx < 0 or idx >= len(pro_names):
            continue
        rows.append({
            "group": group_label,
            "pro_id": int(idx),
            "pro_name": pro_names[idx],
            "mean": float(r["mean"]),
            "sd": float(r["sd"]),
            "hdi_3": float(r["hdi_3%"]),
            "hdi_97": float(r["hdi_97%"]),
            "ess_bulk": float(r.get("ess_bulk", np.nan)),
            "r_hat": float(r.get("r_hat", np.nan)),
        })

    df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def build_fixed_table(idata: az.InferenceData, ind_names: list, out_path: Path, group_label: str):
    # ✅ FIX: sigma_celeb may be absent in trace; only request existing vars
    base_vars = ["beta0", "beta_age", "beta_week", "beta_ind", "sigma_eps", "sigma_pro", "sigma_season"]
    if "sigma_celeb" in idata.posterior.data_vars:
        base_vars.append("sigma_celeb")

    s = az.summary(
        idata,
        var_names=base_vars,
        hdi_prob=0.94,
        round_to=None
    ).reset_index().rename(columns={"index": "param"})

    rows = []
    for _, r in s.iterrows():
        p = r["param"]
        base, idx = _extract_indexed_param(p)
        rec = {
            "group": group_label,
            "param": p,
            "mean": float(r["mean"]),
            "sd": float(r["sd"]),
            "hdi_3": float(r["hdi_3%"]) if "hdi_3%" in r else float("nan"),
            "hdi_97": float(r["hdi_97%"]) if "hdi_97%" in r else float("nan"),
            "ess_bulk": float(r.get("ess_bulk", np.nan)),
            "r_hat": float(r.get("r_hat", np.nan)),
            "label": "",
        }
        if base == "beta_ind" and idx is not None and 0 <= idx < len(ind_names):
            rec["label"] = f"industry={ind_names[idx]}"
        rows.append(rec)

    # ✅ FIX: if sigma_celeb missing, append a deterministic zero row for downstream consistency
    if "sigma_celeb" not in idata.posterior.data_vars:
        rows.append({
            "group": group_label,
            "param": "sigma_celeb",
            "mean": 0.0,
            "sd": 0.0,
            "hdi_3": 0.0,
            "hdi_97": 0.0,
            "ess_bulk": float("nan"),
            "r_hat": float("nan"),
            "label": "absent_in_trace->assumed_0",
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="M04: Postprocess M02/M03 traces. Export tables + variance decomposition + PPC metrics.")
    parser.add_argument("--trace_fans", type=str, default=str(DEFAULT_TRACE_FANS))
    parser.add_argument("--trace_judges", type=str, default=str(DEFAULT_TRACE_JUDGES))
    parser.add_argument("--map", type=str, default=str(DEFAULT_MAP))
    parser.add_argument("--panel", type=str, default=str(DEFAULT_PANEL))
    args = parser.parse_args()

    trace_fans = Path(args.trace_fans)
    trace_judges = Path(args.trace_judges)
    map_path = Path(args.map)
    panel_path = Path(args.panel)

    for p in [trace_fans, trace_judges, map_path, panel_path]:
        if not p.exists():
            raise FileNotFoundError(f"[M04] Missing required input: {p}")

    meta = json.load(open(map_path, "r", encoding="utf-8"))
    pro_names = meta["id_maps"]["pro"]["names"]
    ind_names = meta["id_maps"]["industry"]["names"]

    panel = pd.read_csv(panel_path)
    mean_se2_fans = float(np.mean(np.square(panel["y_f_logit_share_se"].astype(float).to_numpy()))) if "y_f_logit_share_se" in panel.columns else 0.0

    idata_f = az.from_netcdf(trace_fans)
    idata_j = az.from_netcdf(trace_judges)

    fixed_f = build_fixed_table(idata_f, ind_names, OUT_FIXED_FANS, "fans")
    fixed_j = build_fixed_table(idata_j, ind_names, OUT_FIXED_JUDGES, "judges")

    pro_f = build_pro_table(idata_f, pro_names, OUT_PRO_FANS, "fans")
    pro_j = build_pro_table(idata_j, pro_names, OUT_PRO_JUDGES, "judges")

    var_struct_f = variance_decomp_structural(idata_f)
    var_struct_j = variance_decomp_structural(idata_j)

    var_obs_f = variance_decomp_observed_noise_adjusted(idata_f, mean_se2=mean_se2_fans)
    var_obs_j = variance_decomp_observed_noise_adjusted(idata_j, mean_se2=0.0)

    delta_struct = {
        "pro": var_struct_f["pro"]["mean"] - var_struct_j["pro"]["mean"],
        "season": var_struct_f["season"]["mean"] - var_struct_j["season"]["mean"],
        "celeb": var_struct_f["celeb"]["mean"] - var_struct_j["celeb"]["mean"],
        "eps": var_struct_f["eps"]["mean"] - var_struct_j["eps"]["mean"],
    }

    var_payload = {
        "structural": {
            "fans": var_struct_f,
            "judges": var_struct_j,
            "delta_mean_fans_minus_judges": delta_struct,
            "definition": "shares computed from sigma^2 components excluding measurement error",
        },
        "observed_noise_adjusted": {
            "fans": var_obs_f,
            "judges": var_obs_j,
            "definition": "shares computed from sigma^2 components including mean(se_obs^2) as measurement component (fans only)",
        },
    }
    OUT_VAR.parent.mkdir(parents=True, exist_ok=True)
    json.dump(var_payload, open(OUT_VAR, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    y_f_obs = panel["y_f_logit_share_mean"].astype(float).to_numpy() if "y_f_logit_share_mean" in panel.columns else None
    y_j_obs = panel["y_j_score_z"].astype(float).to_numpy() if "y_j_score_z" in panel.columns else None

    ppc_f = ppc_metrics(idata_f, y_f_obs, "fans") if y_f_obs is not None else {"group": "fans", "ppc_available": False}
    ppc_j = ppc_metrics(idata_j, y_j_obs, "judges") if y_j_obs is not None else {"group": "judges", "ppc_available": False}

    ppc_payload = {
        "fans": ppc_f,
        "judges": ppc_j,
        "note": "PPC uses posterior predictive of likelihood; metrics based on posterior predictive mean per observation.",
    }
    OUT_PPC.parent.mkdir(parents=True, exist_ok=True)
    json.dump(ppc_payload, open(OUT_PPC, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("M04 CHECK BLOCK START")
    print("=" * 90)
    print("[M04] Inputs:")
    print(f"  trace_fans   = {trace_fans}")
    print(f"  trace_judges = {trace_judges}")
    print(f"  map          = {map_path}")
    print(f"  panel        = {panel_path}")

    print("\n[M04] Posterior vars present:")
    print("  fans  :", sorted(list(idata_f.posterior.data_vars)))
    print("  judges:", sorted(list(idata_j.posterior.data_vars)))

    print("\n[M04] Outputs:")
    print(f"  fixed_fans   = {OUT_FIXED_FANS}")
    print(f"  fixed_judges = {OUT_FIXED_JUDGES}")
    print(f"  pro_fans     = {OUT_PRO_FANS}")
    print(f"  pro_judges   = {OUT_PRO_JUDGES}")
    print(f"  var_decomp   = {OUT_VAR}")
    print(f"  ppc_metrics  = {OUT_PPC}")

    print("\n[M04] Panel-derived measurement error (fans):")
    print(f"  mean_se2_fans = {mean_se2_fans:.6f}")

    print("\n[M04] Variance decomposition (STRUCTURAL shares, posterior mean + 94% HDI):")
    for k in ["pro", "season", "celeb", "eps"]:
        f = var_struct_f[k]
        j = var_struct_j[k]
        print(f"  {k:>7s} | fans mean={f['mean']:.4f} hdi=[{f['hdi_3']:.4f},{f['hdi_97']:.4f}]"
              f"  || judges mean={j['mean']:.4f} hdi=[{j['hdi_3']:.4f},{j['hdi_97']:.4f}]"
              f"  || delta_mean={delta_struct[k]:+.4f}")

    print("\n[M04] Random effect pro table sizes:")
    print(f"  pro_fans_rows   = {len(pro_f)}")
    print(f"  pro_judges_rows = {len(pro_j)}")

    print("\n[M04] Fixed effects table sizes:")
    print(f"  fixed_fans_rows   = {len(fixed_f)}")
    print(f"  fixed_judges_rows = {len(fixed_j)}")

    print("\nM04 CHECK BLOCK END")
    print("=" * 90 + "\n")

    print("\n" + "=" * 90)
    print("M04 PPC CHECK BLOCK START")
    print("=" * 90)
    print("[M04] PPC metrics:")
    print(json.dumps(ppc_payload, indent=2, ensure_ascii=False))
    print("M04 PPC CHECK BLOCK END")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
