"""
T10 — Hard posterior consistency (posterior-level)

Definition (single elimination):
  Pr( eliminated contestant is argmin vote among active | data )

Generalization (multiple elimination in same event):
  Pr( all eliminated contestants are among bottom-k by logV within active set | data )
  where k = elim_count.

Inputs:
  - data/T07_trace_route2_v2_ta97.nc
  - data/T08_logV_posterior_stats_route2_v2_ta97.csv  (used only for event ordering + meta)

Outputs:
  - data/T10_hard_consistency_by_event_route2_v2_ta97.csv
  - data/T10_hard_consistency_summary_route2_v2_ta97.json
"""

import json
import numpy as np
import pandas as pd
import xarray as xr

# -----------------------------
# Configure (edit if needed)
# -----------------------------
TAG = "route2_v2_ta97"
TRACE_NC = f"data/T07_trace_{TAG}.nc"
T08_CSV = f"data/T08_logV_posterior_stats_{TAG}.csv"

OUT_CSV = f"data/T10_hard_consistency_by_event_{TAG}.csv"
OUT_JSON = f"data/T10_hard_consistency_summary_{TAG}.json"


def main():
    # ---- load T08 for event order/meta ----
    t08 = pd.read_csv(T08_CSV)
    events = (
        t08[["season", "week", "event_id", "n_active"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )
    n_events = len(events)

    # ---- read posterior + constant_data from netcdf (no netCDF4 needed) ----
    post = xr.open_dataset(TRACE_NC, engine="h5netcdf", group="posterior")
    const = xr.open_dataset(TRACE_NC, engine="h5netcdf", group="constant_data")

    # posterior draws (S = chain*draw)
    beta0 = post["beta0"].values.reshape(-1)  # (S,)
    beta1 = post["beta1"].values.reshape(-1)  # (S,)
    S = beta0.size

    # u: (S, n_contestants), w: (S, n_events)
    u = post["u"].values.reshape(post.sizes["chain"] * post.sizes["draw"], -1)
    w = post["w"].values.reshape(post.sizes["chain"] * post.sizes["draw"], -1)

    # constant_data arrays
    active_key_idx = const["active_key_idx"].values.astype(int)  # (n_events, max_active)
    active_mask = const["active_mask"].values.astype(bool)       # (n_events, max_active)
    score_z = const["score_z"].values.astype(float)              # (n_events, max_active)
    elim_idx = const["elim_idx"].values.astype(int)              # (n_events, max_elim), slot indices
    elim_mask = const["elim_mask"].values.astype(bool)           # (n_events, max_elim)

    # ---- alignment checks ----
    assert w.shape[1] == n_events == active_key_idx.shape[0], "event dimension mismatch"
    n_active_trace = active_mask.sum(axis=1).astype(int)
    mismatch = (events["n_active"].values != n_active_trace).sum()
    if mismatch != 0:
        raise ValueError(f"n_active mismatch between T08 and trace: {mismatch} events")

    # ---- compute hard posterior consistency per event ----
    n_elim = elim_mask.sum(axis=1).astype(int)
    hard_prob = np.full(n_events, np.nan, dtype=float)

    for e in range(n_events):
        k = int(n_elim[e])
        if k == 0:
            continue
        n = int(n_active_trace[e])

        # indices for active contestants for event e
        cidx = active_key_idx[e, :n]
        sc = score_z[e, :n]

        # logV samples shape (S, n)
        logV = (
            beta0[:, None]
            + w[:, e][:, None]
            + beta1[:, None] * sc[None, :]
            + u[:, cidx]
        )

        if k == 1:
            elim_slot = int(elim_idx[e, np.where(elim_mask[e])[0][0]])
            argmin_slot = np.argmin(logV, axis=1)
            hard_prob[e] = (argmin_slot == elim_slot).mean()
        else:
            elim_slots = elim_idx[e, np.where(elim_mask[e])[0][:k]]
            bottomk = np.argpartition(logV, kth=k - 1, axis=1)[:, :k]  # (S, k)

            ok = np.ones(S, dtype=bool)
            for s in elim_slots:
                ok &= (bottomk == s).any(axis=1)
            hard_prob[e] = ok.mean()

    t10 = events[["event_id", "season", "week"]].copy()
    t10["elim_count"] = n_elim
    t10["n_active"] = n_active_trace
    t10["hard_consistency_prob"] = hard_prob

    t10.to_csv(OUT_CSV, index=False)

    # ---- summary json ----
    mask = t10["elim_count"] > 0
    summary = {
        "tag": TAG,
        "definition": {
            "single_elim": "Pr(elim is argmin logV among active | data)",
            "multi_elim": "Pr(all eliminated are in bottom-k logV among active | data), k=elim_count",
        },
        "n_events_total": int(n_events),
        "n_events_with_elim": int(mask.sum()),
        "hard_prob_mean": float(np.nanmean(t10.loc[mask, "hard_consistency_prob"])),
        "hard_prob_p25": float(np.nanquantile(t10.loc[mask, "hard_consistency_prob"], 0.25)),
        "hard_prob_p50": float(np.nanquantile(t10.loc[mask, "hard_consistency_prob"], 0.50)),
        "hard_prob_p75": float(np.nanquantile(t10.loc[mask, "hard_consistency_prob"], 0.75)),
        "by_season_mean": {
            str(k): float(v)
            for k, v in t10.loc[mask].groupby("season")["hard_consistency_prob"].mean().to_dict().items()
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", OUT_CSV, OUT_JSON)
    print("hard_prob_mean:", summary["hard_prob_mean"])


if __name__ == "__main__":
    main()
