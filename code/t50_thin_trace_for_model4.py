# t50_thin_trace_for_model4.py
# Proper thinning for ArviZ InferenceData (.nc)

import numpy as np
import arviz as az

INPUT_TRACE = "./data/T07f_trace_route2_v3B.nc"
OUTPUT_TRACE = "./data/T07f_trace_route2_v3B_thin200.nc"
N_DRAWS_THIN = 200
RANDOM_SEED = 202602

print(f"Loading InferenceData from: {INPUT_TRACE}")
idata = az.from_netcdf(INPUT_TRACE)

# -----------------------------
# Check posterior group
# -----------------------------
if not hasattr(idata, "posterior"):
    raise ValueError("No 'posterior' group found in InferenceData.")

posterior = idata.posterior
print("\nPosterior dims:")
print(posterior.dims)

if "draw" not in posterior.dims:
    raise ValueError(
        f"No 'draw' dimension in posterior. Found dims: {posterior.dims}"
    )

n_total = posterior.dims["draw"]
print(f"Total posterior draws: {n_total}")

if N_DRAWS_THIN >= n_total:
    raise ValueError(
        f"N_DRAWS_THIN ({N_DRAWS_THIN}) >= total draws ({n_total})"
    )

# -----------------------------
# Thin draws
# -----------------------------
rng = np.random.default_rng(RANDOM_SEED)
thin_idx = np.sort(
    rng.choice(n_total, size=N_DRAWS_THIN, replace=False)
)

idata_thin = idata.sel(draw=thin_idx)

# -----------------------------
# Save thin InferenceData
# -----------------------------
print(f"\nSaving thin InferenceData to: {OUTPUT_TRACE}")
az.to_netcdf(idata_thin, OUTPUT_TRACE)

print("\nThin trace saved successfully.")
print("New posterior dims:")
print(idata_thin.posterior.dims)
