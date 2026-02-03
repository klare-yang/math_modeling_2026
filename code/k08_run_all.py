#!/usr/bin/env python3
"""k08_run_all.py
Model IV: run full pipeline with a runlog.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SEED = 20260202
DEFAULT_DRAWS = 200


def guess_base_dir(cli_base: str | None) -> Path:
    if cli_base:
        return Path(cli_base).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists() or (cwd / "model4_results").exists():
        return cwd
    here = Path(__file__).resolve()
    if here.parent.name == "code":
        return here.parent.parent
    return here.parent


def find_out_dir(base_dir: Path) -> Path:
    candidates = [
        base_dir / "data" / "model4_results" / "out",
        base_dir / "model4_results" / "out",
        base_dir / "out",
        base_dir / "data" / "out",
    ]
    out = next((p for p in candidates if p.exists()), candidates[0])
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_script(script_path: Path, extra_args: list[str]) -> tuple[bool, str, float]:
    start_time = time.time()
    result = subprocess.run([sys.executable, str(script_path)] + extra_args, capture_output=True, text=True)
    duration = time.time() - start_time
    if result.returncode != 0:
        output = (result.stdout or "") + (result.stderr or "")
        return False, output, duration
    return True, result.stdout, duration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=None, help="Project base directory (auto-detected if omitted).")
    ap.add_argument("--n_draws", type=int, default=DEFAULT_DRAWS, help="Monte Carlo draws for k02 (min 100).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for k02 draws.")
    args = ap.parse_args()

    base_dir = guess_base_dir(args.base_dir)
    code_dir = base_dir / "code"
    out_dir = find_out_dir(base_dir)

    scripts = [
        code_dir / "k01_validate_and_profile_inputs.py",
        code_dir / "k02_load_posterior_samples.py",
        code_dir / "k03_define_policy_grid.py",
        code_dir / "k04_replay_season_under_policy.py",
        code_dir / "k05_compute_objectives.py",
        code_dir / "k06_pareto_and_selection.py",
        code_dir / "k07_make_figures.py",
    ]

    log_content = []
    log_content.append(f"Run Date: {time.ctime()}")
    log_content.append(f"Python Version: {sys.version}")
    log_content.append(f"Global Seed: {int(args.seed)}")
    log_content.append(f"Draws: {int(args.n_draws)} (min 100 enforced in k02)")
    log_content.append("Draw construction: Dirichlet moment-match from T08b pi_mean/pi_sd (k02)")
    log_content.append("f2 definition: mean bottom-half (by judges) survival fraction per elimination week (k05)")
    log_content.append("Pareto selection: maximize f1,f2 minimize f3; select min utopia distance on Pareto set (k06)")
    log_content.append("-" * 20)

    all_success = True
    for script in scripts:
        extra_args = ["--base_dir", str(base_dir)]
        if script.name == "k02_load_posterior_samples.py":
            extra_args += ["--n_draws", str(int(args.n_draws)), "--seed", str(int(args.seed))]
        success, output, duration = run_script(script, extra_args)
        log_content.append(f"Script: {script.name}")
        log_content.append(f"Success: {success}")
        log_content.append(f"Duration: {duration:.2f}s")
        log_content.append("Output:")
        log_content.append(output)
        log_content.append("-" * 20)
        if not success:
            all_success = False
            break

    if all_success:
        log_content.append("All scripts completed successfully.")
    else:
        log_content.append("Pipeline failed.")

    (out_dir / "k08_runlog.txt").write_text("\n".join(log_content))

    if all_success:
        print("Full pipeline executed successfully. Check out/ and fig/ for results.")
        return 0
    print("Pipeline failed. Check k08_runlog.txt for details.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
