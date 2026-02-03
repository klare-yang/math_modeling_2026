# code/m05_m3_figures_and_consistency.py
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PANEL = Path("data/M01_panel_m3.csv")
KEYMAP = Path("data/M01_keymap_m3.json")

VAR_JSON = Path("data/M04_m3_variance_decomp.json")
PRO_FANS = Path("data/M04_m3_random_effects_pro_fans.csv")
PRO_JUDGES = Path("data/M04_m3_random_effects_pro_judges.csv")

OUT_SEASON = Path("data/M05_consistency_by_season.csv")
OUT_OVERALL = Path("data/M05_consistency_overall.json")

FIG_DIR = Path("fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def ensure_exists(*paths):
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"[M05] Missing input: {p}")


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman = Pearson corr of ranks
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    sx = rx.std(ddof=0)
    sy = ry.std(ddof=0)
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def upset_rate_event(df_event: pd.DataFrame, score_col: str, elim_col: str) -> Tuple[int, int]:
    """
    Upset definition:
      take the contestant with minimum score in the event.
      if that contestant is NOT eliminated => upset = 1
    Returns (upset, eligible_event)
    """
    if df_event.empty:
        return 0, 0
    if elim_col not in df_event.columns:
        return 0, 0
    # if no one eliminated in this event, skip
    if df_event[elim_col].sum() <= 0:
        return 0, 0
    idx_min = df_event[score_col].astype(float).idxmin()
    survived = (df_event.loc[idx_min, elim_col] == 0)
    return (1 if survived else 0), 1


def bottomk_consistency_event(df_event: pd.DataFrame, score_col: str, elim_col: str) -> Tuple[float, int]:
    """
    Elimination-consistency:
      let k = number of eliminated in the event (>=1)
      compute whether all eliminated contestants are within bottom-k by score
    Returns (consistency_flag, eligible_event)
    """
    if df_event.empty:
        return np.nan, 0
    e = int(df_event[elim_col].sum())
    if e <= 0:
        return np.nan, 0
    # bottom-k set by score
    df_sorted = df_event.sort_values(score_col, ascending=True)
    bottomk_ids = set(df_sorted.head(e).index.tolist())
    elim_ids = set(df_event.index[df_event[elim_col] == 1].tolist())
    ok = 1.0 if elim_ids.issubset(bottomk_ids) else 0.0
    return ok, 1


def plot_variance_shares(var_payload: Dict[str, Any], out_path: Path):
    # structural shares with HDI
    keys = ["pro", "season", "eps"]  # celeb is 0 in your runs; omit to reduce clutter
    fans = var_payload["structural"]["fans"]
    judges = var_payload["structural"]["judges"]

    def extract(group, k):
        return group[k]["mean"], group[k]["hdi_3"], group[k]["hdi_97"]

    x = np.arange(len(keys))
    width = 0.35

    fans_mean = []
    fans_err_lo = []
    fans_err_hi = []
    judges_mean = []
    judges_err_lo = []
    judges_err_hi = []

    for k in keys:
        m, lo, hi = extract(fans, k)
        fans_mean.append(m)
        fans_err_lo.append(m - lo)
        fans_err_hi.append(hi - m)

        m2, lo2, hi2 = extract(judges, k)
        judges_mean.append(m2)
        judges_err_lo.append(m2 - lo2)
        judges_err_hi.append(hi2 - m2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x - width/2, fans_mean, width, yerr=[fans_err_lo, fans_err_hi], capsize=4, label="fans")
    ax.bar(x + width/2, judges_mean, width, yerr=[judges_err_lo, judges_err_hi], capsize=4, label="judges")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 1)
    ax.set_title("Model 3 variance share (structural) with 94% HDI")
    ax.set_ylabel("share")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pro_top_bottom(df: pd.DataFrame, out_path: Path, title: str, topn: int = 10):
    df2 = df.copy()
    df2 = df2.sort_values("mean", ascending=False)
    top = df2.head(topn)
    bottom = df2.tail(topn).sort_values("mean", ascending=True)

    plot_df = pd.concat([bottom, top], axis=0)
    labels = plot_df["pro_name"].tolist()
    means = plot_df["mean"].to_numpy()
    lo = plot_df["hdi_3"].to_numpy()
    hi = plot_df["hdi_97"].to_numpy()
    err_lo = means - lo
    err_hi = hi - means

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    y = np.arange(len(labels))
    ax.errorbar(means, y, xerr=[err_lo, err_hi], fmt="o")
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("pro random intercept (a_pro)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pro_scatter(df_f: pd.DataFrame, df_j: pd.DataFrame, out_path: Path):
    m = df_f.merge(df_j, on=["pro_id", "pro_name"], suffixes=("_fans", "_judges"))
    x = m["mean_fans"].to_numpy()
    y = m["mean_judges"].to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    ax.axvline(0.0, linewidth=1)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("a_pro mean (fans)")
    ax.set_ylabel("a_pro mean (judges)")
    ax.set_title("Pro effects: fans vs judges")
    # correlation
    if np.std(x) > 1e-12 and np.std(y) > 1e-12:
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = float("nan")
    ax.text(0.02, 0.98, f"corr={corr:.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return corr


def plot_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str):
    v = values[np.isfinite(values)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v, bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_upset_by_season(df_season: pd.DataFrame, out_path: Path, title: str):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(df_season["season"].to_numpy(), df_season["upset_rate_judges"].to_numpy(), marker="o")
    ax.set_title(title)
    ax.set_xlabel("season")
    ax.set_ylabel("upset rate (judges lowest survives)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ensure_exists(PANEL, KEYMAP, VAR_JSON, PRO_FANS, PRO_JUDGES)

    panel = pd.read_csv(PANEL)
    meta = json.load(open(KEYMAP, "r", encoding="utf-8"))
    var_payload = json.load(open(VAR_JSON, "r", encoding="utf-8"))

    df_pf = pd.read_csv(PRO_FANS)
    df_pj = pd.read_csv(PRO_JUDGES)

    # -----------------------------
    # Consistency metrics per event
    # -----------------------------
    required = ["season", "week", "event_id", "score_z", "share_mean", "eliminated_flag"]
    miss = [c for c in required if c not in panel.columns]
    if miss:
        raise ValueError(f"[M05] Missing required cols in panel: {miss}")

    # group by event_id (should uniquely identify within season)
    corrs = []
    upset_j = []
    bottomk_j = []
    by_season = []

    for (season, event_id), g in panel.groupby(["season", "event_id"], sort=True):
        # rank consistency: score vs share (within event)
        corr = spearman_rank_corr(g["score_z"].astype(float).to_numpy(), g["share_mean"].astype(float).to_numpy())
        if np.isfinite(corr):
            corrs.append((season, event_id, corr))

        u, ok = upset_rate_event(g, "score_z", "eliminated_flag")
        if ok:
            upset_j.append((season, event_id, u))

        bk, ok2 = bottomk_consistency_event(g, "score_z", "eliminated_flag")
        if ok2:
            bottomk_j.append((season, event_id, bk))

    df_corr = pd.DataFrame(corrs, columns=["season", "event_id", "spearman_score_vs_share"])
    df_up = pd.DataFrame(upset_j, columns=["season", "event_id", "upset_judges_lowest_survives"])
    df_bk = pd.DataFrame(bottomk_j, columns=["season", "event_id", "elim_in_bottomk_by_judges"])

    df_ev = df_corr.merge(df_up, on=["season", "event_id"], how="outer").merge(df_bk, on=["season", "event_id"], how="outer")

    # season aggregation
    rows = []
    for season, g in df_ev.groupby("season", sort=True):
        sp = g["spearman_score_vs_share"].dropna()
        up = g["upset_judges_lowest_survives"].dropna()
        bk = g["elim_in_bottomk_by_judges"].dropna()

        rows.append({
            "season": int(season),
            "n_events": int(g["event_id"].nunique()),
            "spearman_mean": float(sp.mean()) if len(sp) else np.nan,
            "spearman_median": float(sp.median()) if len(sp) else np.nan,
            "upset_rate_judges": float(up.mean()) if len(up) else np.nan,
            "elim_bottomk_rate_judges": float(bk.mean()) if len(bk) else np.nan,
        })

    df_season = pd.DataFrame(rows).sort_values("season")
    OUT_SEASON.parent.mkdir(parents=True, exist_ok=True)
    df_season.to_csv(OUT_SEASON, index=False)

    # overall summaries
    overall = {
        "spearman_score_vs_share": {
            "mean": float(df_corr["spearman_score_vs_share"].mean()) if len(df_corr) else float("nan"),
            "median": float(df_corr["spearman_score_vs_share"].median()) if len(df_corr) else float("nan"),
            "p05": float(df_corr["spearman_score_vs_share"].quantile(0.05)) if len(df_corr) else float("nan"),
            "p95": float(df_corr["spearman_score_vs_share"].quantile(0.95)) if len(df_corr) else float("nan"),
            "n_events_used": int(df_corr.shape[0]),
        },
        "upset_rate_judges_lowest_survives": {
            "mean": float(df_up["upset_judges_lowest_survives"].mean()) if len(df_up) else float("nan"),
            "n_events_used": int(df_up.shape[0]),
        },
        "elim_in_bottomk_by_judges": {
            "mean": float(df_bk["elim_in_bottomk_by_judges"].mean()) if len(df_bk) else float("nan"),
            "n_events_used": int(df_bk.shape[0]),
        },
    }

    # -----------------------------
    # Figures
    # -----------------------------
    plot_variance_shares(var_payload, FIG_DIR / "M05_variance_shares.png")

    plot_pro_top_bottom(df_pf, FIG_DIR / "M05_pro_effects_top_bottom_fans.png",
                        "Top/Bottom pro effects (fans) with 94% HDI", topn=10)
    plot_pro_top_bottom(df_pj, FIG_DIR / "M05_pro_effects_top_bottom_judges.png",
                        "Top/Bottom pro effects (judges) with 94% HDI", topn=10)

    corr_pro = plot_pro_scatter(df_pf, df_pj, FIG_DIR / "M05_pro_effects_scatter.png")

    plot_hist(df_corr["spearman_score_vs_share"].to_numpy(),
              FIG_DIR / "M05_spearman_by_event_hist.png",
              "Event-level Spearman(rank) between judges score and fan share",
              "Spearman(score_z, share_mean)")

    plot_upset_by_season(df_season, FIG_DIR / "M05_upset_rate_by_season.png",
                         "Upset rate by season (judges lowest survives)")

    overall["pro_effect_corr_fans_vs_judges"] = float(corr_pro)

    OUT_OVERALL.parent.mkdir(parents=True, exist_ok=True)
    json.dump(overall, open(OUT_OVERALL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # -----------------------------
    # PRINT CHECK BLOCKS
    # -----------------------------
    print("\n" + "=" * 90)
    print("M05 CHECK BLOCK START")
    print("=" * 90)
    print("[M05] Inputs:")
    print(f"  panel   = {PANEL}")
    print(f"  keymap  = {KEYMAP}")
    print(f"  varjson = {VAR_JSON}")
    print(f"  pro_f   = {PRO_FANS}")
    print(f"  pro_j   = {PRO_JUDGES}")

    print("\n[M05] Outputs:")
    print(f"  season_metrics = {OUT_SEASON}")
    print(f"  overall_json   = {OUT_OVERALL}")
    print(f"  figures_dir    = {FIG_DIR.resolve()}")

    print("\n[M05] Structural ProShare (from M04 json):")
    f_pro = var_payload["structural"]["fans"]["pro"]
    j_pro = var_payload["structural"]["judges"]["pro"]
    print(f"  fans  proshare mean={f_pro['mean']:.4f} hdi=[{f_pro['hdi_3']:.4f},{f_pro['hdi_97']:.4f}]")
    print(f"  judges proshare mean={j_pro['mean']:.4f} hdi=[{j_pro['hdi_3']:.4f},{j_pro['hdi_97']:.4f}]")
    print(f"  delta mean = {var_payload['structural']['delta_mean_fans_minus_judges']['pro']:+.4f}")

    print("\n[M05] Consistency metrics (overall):")
    print(json.dumps(overall, indent=2, ensure_ascii=False))

    print("\n[M05] Top-5 pro (fans) by mean a_pro:")
    print(df_pf.sort_values("mean", ascending=False)[["pro_name", "mean", "hdi_3", "hdi_97"]].head(5).to_string(index=False))

    print("\n[M05] Bottom-5 pro (fans) by mean a_pro:")
    print(df_pf.sort_values("mean", ascending=True)[["pro_name", "mean", "hdi_3", "hdi_97"]].head(5).to_string(index=False))

    print("\nM05 CHECK BLOCK END")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
