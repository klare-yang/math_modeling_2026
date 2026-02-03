import pandas as pd
import os

def extract_cases(panel_csv, cf_csv, out_csv):
    df_panel = pd.read_csv(panel_csv)
    df_cf = pd.read_csv(cf_csv)
    
    target_contestants = [
        "S2:Jerry Rice",
        "S4:Billy Ray Cyrus",
        "S11:Bristol Palin",
        "S27:Bobby Bones"
    ]
    
    # 1. Calculate is_judge_worst per event
    df_panel['judge_rank_in_week'] = df_panel.groupby('event_id')['judge_total_score'].rank(ascending=True, method='min')
    # judge_rank_in_week == 1 means lowest score in that week
    df_panel['is_judge_worst'] = (df_panel['judge_rank_in_week'] == 1).astype(int)
    
    # 2. Pivot CF data to get pred_elim for each variant
    df_cf_pivot = df_cf.pivot(index='event_id', columns='rule_variant', values='pred_eliminated_ids')
    
    # 3. Filter panel to target contestants
    df_cases = df_panel[df_panel['contestant_id'].isin(target_contestants)].copy()
    
    # 4. Merge with CF predictions
    df_cases = df_cases.merge(df_cf_pivot, on='event_id', how='left')
    
    # 5. Check if the contestant was predicted to be eliminated
    def was_elim(row, variant):
        pred_str = str(row[variant])
        if pred_str == 'nan': return 0
        return int(row['contestant_id'] in pred_str.split(';'))
    
    for v in ['rank', 'percent', 'rank_js', 'percent_js']:
        df_cases[f'pred_elim_{v}'] = df_cases.apply(lambda r: was_elim(r, v), axis=1)
        
    # Keep required columns
    cols = [
        'season', 'week', 'contestant_id', 'judge_total_score', 'fan_vote_estimate',
        'is_judge_worst', 'pred_elim_rank', 'pred_elim_percent', 
        'pred_elim_rank_js', 'pred_elim_percent_js', 'eliminated_real'
    ]
    
    df_out = df_cases[cols]
    df_out.to_csv(out_csv, index=False)
    print(f"Saved controversy cases to {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel_csv", default="../data/p2_interface_week_panel.csv")
    parser.add_argument("--cf_csv", default="../out/p2_week_counterfactual.csv")
    parser.add_argument("--out_csv", default="../out/p2_controversy_cases.csv")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(__file__)
    extract_cases(
        os.path.join(base_dir, args.panel_csv),
        os.path.join(base_dir, args.cf_csv),
        os.path.join(base_dir, args.out_csv)
    )
