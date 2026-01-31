import numpy as np
import pandas as pd
import arviz as az

# -----------------------------
# paths（按你的实际路径改）
# -----------------------------
TRACE_NC = "data/T07_trace_route2_v2_ta97.nc"
T08_CSV  = "data/T08_logV_posterior_stats_route2_v2_ta97.csv"

# -----------------------------
# 1. 读取 InferenceData
# -----------------------------
idata = az.from_netcdf(TRACE_NC)

# posterior 中的随机效应
# u: (chain, draw, contestant)
# w: (chain, draw, event)
u = idata.posterior["u"].values
w = idata.posterior["w"].values

# 合并 chain + draw → (S, dim)
u_s = u.reshape(-1, u.shape[-1])
w_s = w.reshape(-1, w.shape[-1])

# -----------------------------
# 2. 95% HDI 计算函数（不依赖 arviz.summary）
# -----------------------------
def hdi(samples_2d, prob=0.95):
    """
    samples_2d: shape (S, D)
    return: low, high (D,)
    """
    S, D = samples_2d.shape
    m = int(np.floor(prob * S))
    xs = np.sort(samples_2d, axis=0)
    widths = xs[m:] - xs[:S-m]
    idx = np.argmin(widths, axis=0)
    lo = xs[idx, np.arange(D)]
    hi = xs[idx + m, np.arange(D)]
    return lo, hi

# -----------------------------
# 3. 后验统计量
# -----------------------------
u_mean = u_s.mean(axis=0)
u_sd   = u_s.std(axis=0, ddof=1)
u_lo, u_hi = hdi(u_s)

w_mean = w_s.mean(axis=0)
w_sd   = w_s.std(axis=0, ddof=1)
w_lo, w_hi = hdi(w_s)

# -----------------------------
# 4. 映射 contestant / event 信息（复用 T08）
# -----------------------------
t08 = pd.read_csv(T08_CSV)

contestants = (
    t08[["contestant_idx", "contestant_key"]]
    .drop_duplicates()
    .sort_values("contestant_idx")
    .reset_index(drop=True)
)

events = (
    t08[["season", "week", "event_id"]]
    .drop_duplicates()
    .sort_values(["season", "week"])
    .reset_index(drop=True)
)

# -----------------------------
# 5. 输出结果
# -----------------------------
t09_u = contestants.copy()
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

# -----------------------------
# 6. 快速 sanity print（可写进论文）
# -----------------------------
print("mean(u_mean) =", t09_u["u_mean"].mean())
print("mean(w_mean) =", t09_w["w_mean"].mean())
