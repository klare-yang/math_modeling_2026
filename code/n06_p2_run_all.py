import subprocess
import os
import sys
import pandas as pd

def run_step(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running step: {result.stderr}")
        sys.exit(1)
    print(result.stdout)

def main():
    base_dir = os.path.dirname(__file__)
    data_csv = os.path.abspath(os.path.join(base_dir, "../MANUS_P2_handoff_package_v2/data/p2_interface_week_panel.csv"))
    out_dir = os.path.abspath(os.path.join(base_dir, "../out"))
    fig_dir = os.path.abspath(os.path.join(base_dir, "../fig"))
    
    cf_csv = os.path.join(out_dir, "p2_week_counterfactual.csv")
    season_csv = os.path.join(out_dir, "p2_season_metrics.csv")
    overall_json = os.path.join(out_dir, "p2_overall_metrics.json")
    cases_csv = os.path.join(out_dir, "p2_controversy_cases.csv")
    
    # Step 2
    run_step([sys.executable, os.path.join(base_dir, "n02_p2_week_counterfactual.py"), 
              "--in_csv", data_csv, "--out_csv", cf_csv])
    
    # Step 3
    run_step([sys.executable, os.path.join(base_dir, "n03_p2_metrics.py"), 
              "--panel_csv", data_csv, "--cf_csv", cf_csv, 
              "--out_season", season_csv, "--out_overall", overall_json])
    
    # Step 4
    run_step([sys.executable, os.path.join(base_dir, "n04_p2_controversy_cases.py"), 
              "--panel_csv", data_csv, "--cf_csv", cf_csv, "--out_csv", cases_csv])
    
    # Step 5
    run_step([sys.executable, os.path.join(base_dir, "n05_p2_figures.py"), 
              "--season_csv", season_csv, "--overall_json", overall_json, "--fig_dir", fig_dir])
    
    # Sanity Checks
    print("\n--- Sanity Checks ---")
    df_panel = pd.read_csv(data_csv)
    print(f"Input rows: {len(df_panel)} (Expected: 2777)")
    print(f"Input events: {df_panel['event_id'].nunique()} (Expected: 335)")
    
    df_cf = pd.read_csv(cf_csv)
    print(f"CF output rows: {len(df_cf)} (Expected: 335 * 4 = 1340)")
    
    df_cases = pd.read_csv(cases_csv)
    target_contestants = ["S2:Jerry Rice", "S4:Billy Ray Cyrus", "S11:Bristol Palin", "S27:Bobby Bones"]
    found_cases = df_cases['contestant_id'].unique()
    print(f"Controversy cases found: {len(found_cases)} / 4")
    for c in target_contestants:
        if c in found_cases:
            n_weeks = len(df_cases[df_cases['contestant_id'] == c])
            print(f"  - {c}: {n_weeks} weeks found")
        else:
            print(f"  - {c}: NOT FOUND")

if __name__ == "__main__":
    main()
