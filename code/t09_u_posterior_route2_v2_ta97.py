import numpy as np
import pandas as pd
import netCDF4

TRACE_NC = "data/T07_trace_route2_v2_ta97.nc"
T08_CSV  = "data/T08_logV_posterior_stats_route2_v2_ta97.csv"

def hdi_vec(samples_2d, hdi_prob=0.95):
    """
    Vectorized HDI for samples array shape (S, D).
    Returns low, high arrays shape (D,)
    """
    S, D = samples_2d.shape
    m = int(np.floor(hdi_prob * S))
    xs = np.sort(samples_2d, axis=0)
    widths = xs[m:, :] - xs[:S-m, :]
    idx = np.argmin(widths, axis=0)
    low = xs[idx, np.arange(D)]
    high = xs[idx + m, np.arange(D)]
    return low, high

# ---- mapping from T08 (contestant_idx <-> contestant_key, event order) ----
t08 = pd.read_csv(T08_CSV)
cont_map = (
    t08[["contestant_idx","contestant_key"]]
    .drop_duplicates()
    .sort_values("contestant_idx")
    .reset_index(drop=True)
)
events = (
    t08[["season","week","event_id"]]
    .drop_duplicates()
    .sort_values(["season","week"])
    .reset_index(drop=True)
)

# ---- load posterior u,w from netcdf ----
nc = netCDF4.Dataset(TRACE_NC, "r")
post = nc.groups["posterior"]
u = np.array(post.variables["u"][:])  # (chain, draw, n_contestants)
w = np.array(post.variables["w"][:])  # (chain, draw, n_events)

u_s = u.reshape(-1, u.shape[-1])  # (S, n_contestants)
w_s = w.reshape(-1, w.shape[-1])  # (S, n_events)

# ---- posterior stats ----
u_mean = u_s.mean(axis=0)
u_sd   = u_s.std(axis=0, ddof=1)
u_lo, u_hi = hdi_vec(u_s, 0.95)

w_mean = w_s.mean(axis=0)
w_sd   = w_s.std(axis=0, ddof=1)
w_lo, w_hi = hdi_vec(w_s, 0.95)

# ---- build outputs ----
t09_u = cont_map.copy()
t09_u["u_mean"] = u_mean
t09_u["u_sd"] = u_sd
t09_u["u_hdi_low"] = u_lo
t09_u["u_hdi_high"] = u_hi

t09_w = events.copy()
t09_w["w_mean"] = w_mean
t09_w["w_sd"] = w_sd
t09_w["w_hdi_low"] = w_lo
t09_w["w_hdi_high"] = w_hi

t09_u.to_csv("data/T09_u_posterior.csv", index=False)
t09_w.to_csv("data/T09_w_posterior.csv", index=False)

print("Saved:", "data/T09_u_posterior.csv", "data/T09_w_posterior.csv")
print("Sanity:", "mean(u_mean)=", t09_u["u_mean"].mean(), "mean(w_mean)=", t09_w["w_mean"].mean())
