#!/usr/bin/env python3
# T11_soft_consistency_route2_v2_ta97.py
# Soft consistency (probabilistic calibration):
# For each event t, compute posterior E[p_elim(t)] and 95% HDI, where
#   p_elim(t) = sum_{i in eliminated_t} softmax(-logV_{i,t}) over active set A_t.
#
# Outputs (keep your naming convention):
#   data/T11_soft_consistency_by_event_route2_v2_ta97.csv
#   data/T11_soft_consistency_summary_route2_v2_ta97.json
#
# Preferred backend: arviz.from_netcdf (no netCDF4 needed).
# If arviz import fails in your env, use xarray+h5netcdf fallback.

import json
import numpy as np
import pandas as pd

TAG = "route2_v2_ta97"
TRACE_NC = "data/T07_trace_route2_v2_ta97.nc"
T08_CSV  = "data/T08_logV_posterior_stats_route2_v2_ta97.csv"

def hdi_1d(x, prob=0.95):
    x = np.sort(np.asarray(x))
    S = x.size
    m = int(np.floor(prob * S))
    if m < 1:
        return (np.nan, np.nan)
    widths = x[m:] - x[:S-m]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i+m])

def load_from_arviz():
    import arviz as az
    idata = az.from_netcdf(TRACE_NC)
    beta0 = idata.posterior["beta0"].values.reshape(-1)
    beta1 = idata.posterior["beta1"].values.reshape(-1)
    u = idata.posterior["u"].values
    w = idata.posterior["w"].values
    u = u.reshape(-1, u.shape[-1])
    w = w.reshape(-1, w.shape[-1])

    cd = idata.constant_data
    active_key_idx = cd["active_key_idx"].values.astype(int)
    score_z = cd["score_z"].values.astype(float)
    active_mask = cd["active_mask"].values.astype(bool)
    elim_idx = cd["elim_idx"].values.astype(int)
    elim_mask = cd["elim_mask"].values.astype(bool)
    return beta0, beta1, u, w, active_key_idx, score_z, active_mask, elim_idx, elim_mask

def load_from_xarray():
    # Fallback: xarray + h5netcdf (pip install h5netcdf)
    import xarray as xr
    ds_post = xr.open_dataset(TRACE_NC, group="posterior", engine="h5netcdf")
    ds_const = xr.open_dataset(TRACE_NC, group="constant_data", engine="h5netcdf")

    beta0 = ds_post["beta0"].values.reshape(-1)
    beta1 = ds_post["beta1"].values.reshape(-1)
    u = ds_post["u"].values.reshape(beta0.size, -1)
    w = ds_post["w"].values.reshape(beta0.size, -1)

    active_key_idx = ds_const["active_key_idx"].values.astype(int)
    score_z = ds_const["score_z"].values.astype(float)
    active_mask = ds_const["active_mask"].values.astype(bool)
    elim_idx = ds_const["elim_idx"].values.astype(int)
    elim_mask = ds_const["elim_mask"].values.astype(bool)
    return beta0, beta1, u, w, active_key_idx, score_z, active_mask, elim_idx, elim_mask

def main():
    try:
        beta0, beta1, u, w, active_key_idx, score_z, active_mask, elim_idx, elim_mask = load_from_arviz()
        backend = "arviz"
    except Exception:
        beta0, beta1, u, w, active_key_idx, score_z, active_mask, elim_idx, elim_mask = load_from_xarray()
        backend = "xarray+h5netcdf"

    t08 = pd.read_csv(T08_CSV)
    events = (
        t08[["season","week","event_id"]]
        .drop_duplicates()
        .sort_values(["season","week"])
        .reset_index(drop=True)
    )

    n_events = len(events)
    assert active_key_idx.shape[0] == n_events == w.shape[1], "event dimension mismatch"

    n_active = active_mask.sum(axis=1).astype(int)
    n_elim = elim_mask.sum(axis=1).astype(int)

    p_mean = np.full(n_events, np.nan)
    p_lo = np.full(n_events, np.nan)
    p_hi = np.full(n_events, np.nan)

    for e in range(n_events):
        k = int(n_elim[e])
        n = int(n_active[e])
        if k == 0:
            continue

        cidx = active_key_idx[e, :n]
        sc = score_z[e, :n]

        logV = (beta0[:, None] + w[:, e][:, None] + beta1[:, None] * sc[None, :] + u[:, cidx])

        # stable softmax of (-logV)
        x = -logV
        x = x - x.max(axis=1, keepdims=True)
        p = np.exp(x)
        p = p / p.sum(axis=1, keepdims=True)

        elim_slots = elim_idx[e, :k]
        p_elim = p[:, elim_slots].sum(axis=1)

        p_mean[e] = p_elim.mean()
        p_lo[e], p_hi[e] = hdi_1d(p_elim, 0.95)

    out = events.copy()
    out["n_active"] = n_active
    out["elim_count"] = n_elim
    out["p_elim_mean"] = p_mean
    out["p_elim_hdi_low"] = p_lo
    out["p_elim_hdi_high"] = p_hi

    out_csv = f"data/T11_soft_consistency_by_event_route2_v2_ta97.csv"
    out.to_csv(out_csv, index=False)

    mask = out["elim_count"] > 0
    summary = {
        "tag": TAG,
        "backend_used": backend,
        "definition": {
            "single_elim": "E[p_elim | data], p_elim = softmax(-logV_elim) within active set",
            "multi_elim": "E[p_elim_set | data], p_elim_set = sum_{i in eliminated} softmax(-logV_i)"
        },
        "n_events_total": int(n_events),
        "n_events_with_elim": int(mask.sum()),
        "p_elim_mean_over_events": float(out.loc[mask, "p_elim_mean"].mean()),
        "p_elim_p25": float(out.loc[mask, "p_elim_mean"].quantile(0.25)),
        "p_elim_p50": float(out.loc[mask, "p_elim_mean"].quantile(0.50)),
        "p_elim_p75": float(out.loc[mask, "p_elim_mean"].quantile(0.75)),
        "by_season_mean": out.loc[mask].groupby("season")["p_elim_mean"].mean().to_dict()
    }

    out_json = f"data/T11_soft_consistency_summary_route2_v2_ta97.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_csv, out_json)

if __name__ == "__main__":
    main()
