# code/t07_run_model_sample_route2.py
# -*- coding: utf-8 -*-

"""
T07 (Route2, Normalized) — Build tensors, run strict consistency checks, then sample PyMC model.

Outputs:
- data/T07_input_consistency_report.json
- data/T07_trace.nc

Hard guarantees going forward:
- Only read canonical T03 payload (schema_version == route2_norm_v1).
- contestant_key canonicalization is identical to T03.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pymc as pm

from t06_topk_likelihood_pymc import ModelData, build_model


# =========================
# CONFIG
# =========================
DATA_DIR = Path("data")

INPUT_PANEL = DATA_DIR / "T02_long_format_panel.csv"
INPUT_T03   = DATA_DIR / "T03_active_elimination_structure_route2.json"

TRACE_OUT   = DATA_DIR / "T07_trace.nc"
REPORT_OUT  = DATA_DIR / "T07_input_consistency_report.json"

FEATURE_COLS = ["judge_score_total"]

W_INDEX_MODE = "event"  # "event" or "week_number"
LIKELIHOOD   = "plackett_luce_set"  # or "softmax_set_approx"

DRAWS = 1000
TUNE = 1000
CHAINS = 2
TARGET_ACCEPT = 0.9
RANDOM_SEED = 2026

DROP_ZERO_ELIM_EVENTS = False
MISSING_SCORE_POLICY = "raise"  # "raise" or "zero"

SCHEMA_VERSION_REQUIRED = "route2_norm_v1"


def normalize_name(x: str) -> str:
    return " ".join(str(x).strip().split())


def make_contestant_key(season: int, celebrity_name: str) -> str:
    return f"S{int(season)}:{normalize_name(celebrity_name)}"


def load_t03_payload(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "events" in obj and "schema_version" in obj:
        meta = {k: v for k, v in obj.items() if k != "events"}
        return meta, obj["events"]
    raise ValueError("T03 is not normalized payload. Re-run T03.")


def pad_and_mask(list_of_lists: List[List[int]], maxlen: int, pad_val: int = -1):
    arr = np.full((len(list_of_lists), maxlen), pad_val, dtype=np.int32)
    mask = np.zeros((len(list_of_lists), maxlen), dtype=bool)
    for i, l in enumerate(list_of_lists):
        arr[i, :len(l)] = np.asarray(l, dtype=np.int32)
        mask[i, :len(l)] = True
    return arr, mask


def main():
    panel = pd.read_csv(INPUT_PANEL)
    panel["season"] = panel["season"].astype(int)
    panel["week"] = panel["week"].astype(int)
    panel["celebrity_name"] = panel["celebrity_name"].astype(str).map(normalize_name)
    panel["contestant_key"] = panel.apply(lambda r: make_contestant_key(r["season"], r["celebrity_name"]), axis=1)
    panel["event_key"] = panel.apply(lambda r: f"{r['season']}-{r['week']}", axis=1)

    missing_cols = [c for c in FEATURE_COLS if c not in panel.columns]
    if missing_cols:
        raise KeyError(f"Missing FEATURE_COLS: {missing_cols}")

    dup = panel.duplicated(subset=["event_key", "contestant_key"], keep=False)
    dup_rows = int(dup.sum())
    if dup_rows > 0:
        panel = (
            panel.groupby(["event_key", "season", "week", "contestant_key"], as_index=False)[FEATURE_COLS]
            .mean()
        )

    score_lookup = {}
    for _, r in panel.iterrows():
        score_lookup[(r["event_key"], r["contestant_key"])] = np.asarray(
            [float(r[c]) for c in FEATURE_COLS], dtype=np.float64
        )

    meta, events = load_t03_payload(INPUT_T03)
    if meta.get("schema_version") != SCHEMA_VERSION_REQUIRED:
        raise ValueError(f"T03 schema_version mismatch: {meta.get('schema_version')}")

    def ek_sort_key(ek: str):
        s, w = ek.split("-")
        return (int(s), int(w))

    event_keys = sorted(events.keys(), key=ek_sort_key)

    anomalies = {
        "dup_panel_event_contestant_rows": dup_rows,
        "event_missing_fields": 0,
        "active_ids_not_sorted": 0,
        "elim_not_subset_active": 0,
        "elim_indices_mismatch": 0,
        "elim_count_mismatch": 0,
        "missing_scores": 0,
        "missing_scores_samples": [],
        "max_elim_count": 0,
    }

    active_ids_by_event: List[List[str]] = []
    eliminated_pos_by_event: List[List[int]] = []
    event_week_labels = []

    for ek in event_keys:
        e = events[ek]
        required = ["season", "week", "active_ids", "eliminated_ids", "eliminated_indices", "n_active", "elim_count", "terminal"]
        if any(k not in e for k in required):
            anomalies["event_missing_fields"] += 1
            continue

        week = int(e["week"])
        A = list(e["active_ids"])
        E_ids = list(e["eliminated_ids"])
        E_idx = list(e["eliminated_indices"])

        if A != sorted(A):
            anomalies["active_ids_not_sorted"] += 1
            A = sorted(A)

        if not set(E_ids).issubset(set(A)):
            anomalies["elim_not_subset_active"] += 1

        if int(e["elim_count"]) != len(E_ids) or int(e["elim_count"]) != len(E_idx):
            anomalies["elim_count_mismatch"] += 1

        anomalies["max_elim_count"] = max(anomalies["max_elim_count"], len(E_ids))

        idx_map = {cid: j for j, cid in enumerate(A)}
        E_idx_re = [idx_map[cid] for cid in E_ids]
        if E_idx_re != E_idx:
            anomalies["elim_indices_mismatch"] += 1
            E_idx = E_idx_re  # force canonical

        active_ids_by_event.append(A)
        eliminated_pos_by_event.append(E_idx)

        if W_INDEX_MODE == "event":
            event_week_labels.append(ek)
        elif W_INDEX_MODE == "week_number":
            event_week_labels.append(week)
        else:
            raise ValueError(f"Unknown W_INDEX_MODE={W_INDEX_MODE}")

    n_events = len(active_ids_by_event)
    if n_events == 0:
        raise RuntimeError("No valid events loaded from T03.")

    all_contestants = sorted({cid for A in active_ids_by_event for cid in A})
    key2id = {k: i for i, k in enumerate(all_contestants)}
    n_contestants = len(all_contestants)

    if W_INDEX_MODE == "event":
        week2id = {label: i for i, label in enumerate(event_week_labels)}
        event_week_idx = np.asarray([week2id[label] for label in event_week_labels], dtype=np.int32)
        n_weeks = len(week2id)
    else:
        uniq_week = sorted(set(event_week_labels))
        week2id = {w: i for i, w in enumerate(uniq_week)}
        event_week_idx = np.asarray([week2id[w] for w in event_week_labels], dtype=np.int32)
        n_weeks = len(week2id)

    max_active = max(len(a) for a in active_ids_by_event)
    max_elim = max(len(e) for e in eliminated_pos_by_event)

    active_idx_list: List[List[int]] = []
    X_active_list: List[np.ndarray] = []

    for i in range(n_events):
        ek = event_keys[i]
        A = active_ids_by_event[i]

        active_idx_list.append([key2id[cid] for cid in A])

        X_rows = []
        for cid in A:
            vec = score_lookup.get((ek, cid))
            if vec is None:
                anomalies["missing_scores"] += 1
                if len(anomalies["missing_scores_samples"]) < 10:
                    anomalies["missing_scores_samples"].append({"event_key": ek, "contestant_key": cid})
                if MISSING_SCORE_POLICY == "raise":
                    raise KeyError(f"Missing score for (event_key, contestant_key)=({ek},{cid})")
                vec = np.zeros((len(FEATURE_COLS),), dtype=np.float64)
            X_rows.append(vec)
        X_active_list.append(np.vstack(X_rows).astype(np.float64))

    active_idx_pad, active_mask = pad_and_mask(active_idx_list, max_active)

    p = len(FEATURE_COLS)
    X_active_pad = np.zeros((n_events, max_active, p), dtype=np.float64)
    for i, Xmat in enumerate(X_active_list):
        X_active_pad[i, :Xmat.shape[0], :] = Xmat

    elim_pos_pad, elim_mask = pad_and_mask(eliminated_pos_by_event, max_elim)

    if DROP_ZERO_ELIM_EVENTS:
        k = elim_mask.sum(axis=1)
        zero = (k == 0)
        elim_mask[zero, :] = False
        elim_pos_pad[zero, :] = -1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"INPUT_PANEL": str(INPUT_PANEL), "INPUT_T03": str(INPUT_T03)},
        "config": {
            "FEATURE_COLS": FEATURE_COLS,
            "W_INDEX_MODE": W_INDEX_MODE,
            "LIKELIHOOD": LIKELIHOOD,
            "DROP_ZERO_ELIM_EVENTS": DROP_ZERO_ELIM_EVENTS,
            "MISSING_SCORE_POLICY": MISSING_SCORE_POLICY,
        },
        "dimensions": {
            "n_events": n_events,
            "n_contestants": n_contestants,
            "n_weeks": int(n_weeks),
            "max_active": int(max_active),
            "max_elim": int(max_elim),
        },
        "anomalies": anomalies,
    }
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[T07] Consistency report saved: {REPORT_OUT}")
    print(json.dumps(report["dimensions"], indent=2))
    print("anomalies:", json.dumps(anomalies, indent=2))

    data = ModelData(
        n_contestants=n_contestants,
        n_events=n_events,
        n_weeks=int(n_weeks),
        p=p,
        event_week_idx=event_week_idx,
        max_active=max_active,
        active_idx_pad=active_idx_pad,
        active_mask=active_mask,
        X_active_pad=X_active_pad,
        max_elim=max_elim,
        elim_pos_pad=elim_pos_pad,
        elim_mask=elim_mask,
    )

    print("[T07] Building model...")
    model = build_model(data, likelihood=LIKELIHOOD)

    print("[T07] Sampling...")
    with model:
        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
        )

    # avoid hard dependency on arviz
    try:
        import arviz as az
        az.to_netcdf(trace, TRACE_OUT)
    except ImportError:
        trace.to_netcdf(TRACE_OUT)

    print(f"[T07] Trace saved: {TRACE_OUT}")


if __name__ == "__main__":
    main()
