"""T17 — random effects plots (u_i and w_t)

Inputs (preferred):
  - data/T09_u_posterior_route2_v2_ta97.csv
  - data/T09_w_posterior_route2_v2_ta97.csv

Fallback (if you kept untagged names):
  - data/T09_u_posterior.csv
  - data/T09_w_posterior.csv

Outputs:
  - fig/T17_u_forest_topbottom_route2_v2_ta97.png/.pdf
  - fig/T17_w_line_hdi_route2_v2_ta97.png/.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TAG = "route2_v2_ta97"

ROOT = Path(__file__).resolve().parents[1]  # project root if code/ is one level under
DATA = ROOT / "data"
FIG  = ROOT / "fig"
FIG.mkdir(parents=True, exist_ok=True)

def load_csv(preferred: Path, fallback: Path) -> pd.DataFrame:
    if preferred.exists():
        return pd.read_csv(preferred)
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"Missing both: {preferred} and {fallback}")

u_path = DATA / f"T09_u_posterior_{TAG}.csv"
w_path = DATA / f"T09_w_posterior_{TAG}.csv"
u_fb   = DATA / "T09_u_posterior.csv"
w_fb   = DATA / "T09_w_posterior.csv"

u_df = load_csv(u_path, u_fb)
w_df = load_csv(w_path, w_fb)

# -----------------------------
# Figure 1: u_i forest (top/bottom 20 by posterior mean)
# -----------------------------
u_sorted = u_df.sort_values("u_mean")
bottom = u_sorted.head(20).copy()
top = u_sorted.tail(20).copy()
sel = pd.concat([bottom, top], axis=0).sort_values("u_mean").reset_index(drop=True)

y = np.arange(len(sel))
x = sel["u_mean"].to_numpy()
xerr = np.vstack([x - sel["u_hdi_low"].to_numpy(), sel["u_hdi_high"].to_numpy() - x])

plt.figure(figsize=(10, 12))
plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=2)
plt.yticks(y, sel["contestant_key"].astype(str).to_list(), fontsize=8)
plt.axvline(0, linewidth=1)
plt.xlabel("u_mean (posterior mean)")
plt.title("T17: Contestant random effects u_i (bottom 20 & top 20) with 95% HDI")
plt.tight_layout()

out1_png = FIG / f"T17_u_forest_topbottom_{TAG}.png"
out1_pdf = FIG / f"T17_u_forest_topbottom_{TAG}.pdf"
plt.savefig(out1_png, dpi=200)
plt.savefig(out1_pdf)
plt.close()

# -----------------------------
# Figure 2: w_t line with HDI band (sorted by season/week)
# -----------------------------
w_sorted = w_df.sort_values(["season", "week"]).reset_index(drop=True)
xx = np.arange(len(w_sorted))
wm = w_sorted["w_mean"].to_numpy()
lo = w_sorted["w_hdi_low"].to_numpy()
hi = w_sorted["w_hdi_high"].to_numpy()

plt.figure(figsize=(12, 4))
plt.plot(xx, wm, linewidth=1)
plt.fill_between(xx, lo, hi, alpha=0.2)
plt.axhline(0, linewidth=1)
plt.xlabel("event index (sorted by season, week)")
plt.ylabel("w (posterior)")
plt.title("T17: Week/event random effects w_t with 95% HDI band")

# season tick marks (avoid clutter)
season = w_sorted["season"].to_numpy()
changes = np.where(np.diff(season) != 0)[0] + 1
tick_pos = [0] + changes.tolist() + [len(w_sorted) - 1]
tick_lab = [f"S{w_sorted['season'].iloc[p]}" for p in tick_pos]

if len(tick_pos) > 15:
    starts = [0] + changes.tolist()
    idx = np.linspace(0, len(starts) - 1, num=min(12, len(starts)), dtype=int).tolist()
    tick_pos = [starts[i] for i in idx] + [len(w_sorted) - 1]
    tick_lab = [f"S{w_sorted['season'].iloc[p]}" for p in tick_pos[:-1]] + ["end"]

plt.xticks(tick_pos, tick_lab, fontsize=8)
plt.tight_layout()

out2_png = FIG / f"T17_w_line_hdi_{TAG}.png"
out2_pdf = FIG / f"T17_w_line_hdi_{TAG}.pdf"
plt.savefig(out2_png, dpi=200)
plt.savefig(out2_pdf)
plt.close()

print("Saved:")
print(" -", out1_png)
print(" -", out1_pdf)
print(" -", out2_png)
print(" -", out2_pdf)
