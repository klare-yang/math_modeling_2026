import pandas as pd
import os
import sys

# Add current directory to path to import n01
sys.path.append(os.path.dirname(__file__))
from n01_p2_rules import calculate_rank_method, calculate_percent_method, apply_judges_save

def run_counterfactual(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    events = df['event_id'].unique()
    
    results = []
    
    for event_id in events:
        df_event = df[df['event_id'] == event_id]
        season = df_event.iloc[0]['season']
        week = df_event.iloc[0]['week']
        elim_count = df_event.iloc[0]['elim_count']
        real_eliminated = df_event[df_event['eliminated_real'] == 1]['contestant_id'].tolist()
        
        # Rule variants
        variants = ['rank', 'percent', 'rank_js', 'percent_js']
        
        for variant in variants:
            bottom_2_ids = []
            if variant == 'rank':
                pred_elim, _ = calculate_rank_method(df_event, elim_count)
            elif variant == 'percent':
                pred_elim, _ = calculate_percent_method(df_event, elim_count)
            elif variant == 'rank_js':
                pred_elim, bottom_2_ids = apply_judges_save(df_event, elim_count, 'rank')
            elif variant == 'percent_js':
                pred_elim, bottom_2_ids = apply_judges_save(df_event, elim_count, 'percent')
            
            # Match metrics
            pred_set = set(pred_elim)
            real_set = set(real_eliminated)
            
            match_real_any = len(pred_set.intersection(real_set)) > 0 if elim_count > 0 else True
            match_real_all = pred_set == real_set
            
            results.append({
                'season': season,
                'week': week,
                'event_id': event_id,
                'rule_variant': variant,
                'elim_count': elim_count,
                'pred_eliminated_ids': ';'.join(pred_elim),
                'pred_bottom2_ids': ';'.join(bottom_2_ids),
                'real_eliminated_ids': ';'.join(real_eliminated),
                'match_real_any': int(match_real_any),
                'match_real_all': int(match_real_all)
            })
            
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved counterfactual results to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="../data/p2_interface_week_panel.csv")
    parser.add_argument("--out_csv", default="../out/p2_week_counterfactual.csv")
    args = parser.parse_args()
    
    # Resolve paths relative to script location if they are default
    base_dir = os.path.dirname(__file__)
    in_path = os.path.join(base_dir, args.in_csv)
    out_path = os.path.join(base_dir, args.out_csv)
    
    run_counterfactual(in_path, out_path)
