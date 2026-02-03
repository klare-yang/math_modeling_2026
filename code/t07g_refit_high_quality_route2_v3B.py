#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T07g (route2_v3B): High-quality refit of the T07f model.

Scope
-----
- Same model specification as T07f (imported) to guarantee identical likelihood.
- Higher default sampling budget and stricter target_accept.

Outputs
-------
- data/T07g_trace_route2_v3B_highq.nc
- data/T07g_model_metadata_route2_v3B_highq.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from t07f_run_model_S_setlik_judgessave_2stage_route2_v3B import build_model_input, build_and_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t05_path", default="data/T05_model_ready_events_route2_v3B.csv")
    ap.add_argument("--mask_path", default="data/T04_judges_save_event_mask_route2_v3B.csv")
    ap.add_argument("--out_trace", default="data/T07g_trace_route2_v3B_highq.nc")
    ap.add_argument("--out_meta", default="data/T07g_model_metadata_route2_v3B_highq.json")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=20260202)
    ap.add_argument("--tau_rank", type=float, default=0.15)
    args = ap.parse_args()

    t05_path = Path(args.t05_path)
    mask_path = Path(args.mask_path)
    if not t05_path.exists():
        raise FileNotFoundError(f"[T07g] T05 not found: {t05_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"[T07g] mask not found: {mask_path}")

    t05 = pd.read_csv(t05_path)
    mask_df = pd.read_csv(mask_path)
    mi = build_model_input(t05, mask_df)

    meta = {
        "t05_path": str(t05_path),
        "mask_path": str(mask_path),
        "n_events": int(mi.n_events),
        "use_two_stage": int(mi.use_two_stage.sum()),
        "tau_rank": float(args.tau_rank),
        "sampling": {
            "draws": int(args.draws),
            "tune": int(args.tune),
            "chains": int(args.chains),
            "cores": int(args.cores),
            "target_accept": float(args.target_accept),
            "seed": int(args.seed),
        },
        "note": "Model spec imported from T07f for exact consistency.",
    }

    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 90)
    print("[T07g] High-quality refit starting...")
    print("=" * 90)
    print(f"[T07g] n_events={mi.n_events}, use_two_stage={int(mi.use_two_stage.sum())}")
    print("[T07g] Sampling config:")
    print(f"  draws={args.draws}, tune={args.tune}, chains={args.chains}, cores={args.cores}, target_accept={args.target_accept}, seed={args.seed}")

    idata = build_and_sample(
        mi,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        seed=args.seed,
        tau_rank=float(args.tau_rank),
    )

    out_trace = Path(args.out_trace)
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(out_trace)
    print(f"[T07g] Saved trace: {out_trace}")
    print(f"[T07g] Saved meta: {out_meta}")

    print("=" * 90)
    print("[T07g] Done.")
    print("=" * 90)


if __name__ == "__main__":
    main()
