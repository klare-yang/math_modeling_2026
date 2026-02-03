import pandas as pd
import numpy as np
import os
import json
# scipy not available, using pandas corr

def calculate_metrics(panel_csv, cf_csv, out_season_csv, out_overall_json):
    df_panel = pd.read_csv(panel_csv)
    df_cf = pd.read_csv(cf_csv)
    
    # 1. Consistency Metrics
    # Filter to events where elim_count >= 1
    df_cf_elim = df_cf[df_cf['elim_count'] >= 1]
    
    season_metrics = df_cf_elim.groupby(['season', 'rule_variant']).agg({
        'match_real_all': 'mean',
        'match_real_any': 'mean'
    }).reset_index()
    season_metrics.rename(columns={'match_real_all': 'exact_match_rate', 'match_real_any': 'hit_rate_any'}, inplace=True)
    
    # 2. Bias Metrics (Overturn Rate)
    # We need to identify judge-worst and fan-worst for each event
    bias_data = []
    events = df_panel['event_id'].unique()
    for eid in events:
        sub = df_panel[df_panel['event_id'] == eid]
        # Judge worst (lowest score)
        # Tie handling: use contestant_id to be stable
        j_worst_id = sub.sort_values(by=['judge_total_score', 'contestant_id']).iloc[0]['contestant_id']
        v_worst_id = sub.sort_values(by=['fan_vote_estimate', 'contestant_id']).iloc[0]['contestant_id']
        
        is_disagreement = (j_worst_id != v_worst_id)
        
        bias_data.append({
            'event_id': eid,
            'j_worst_id': j_worst_id,
            'v_worst_id': v_worst_id,
            'is_disagreement': is_disagreement
        })
    df_bias_ref = pd.DataFrame(bias_data)
    
    # Merge with CF data
    df_cf_bias = df_cf.merge(df_bias_ref, on='event_id')
    
    # Only look at elim_count == 1 for simple alignment check
    df_cf_bias_1 = df_cf_bias[df_cf_bias['elim_count'] == 1].copy()
    
    def check_align(row, target_col):
        pred_ids = row['pred_eliminated_ids'].split(';')
        return int(row[target_col] in pred_ids)

    df_cf_bias_1['align_fan'] = df_cf_bias_1.apply(lambda r: check_align(r, 'v_worst_id'), axis=1)
    df_cf_bias_1['align_judge'] = df_cf_bias_1.apply(lambda r: check_align(r, 'j_worst_id'), axis=1)
    
    # Bias summary by season and variant
    bias_metrics = df_cf_bias_1.groupby(['season', 'rule_variant']).agg({
        'align_fan': 'mean',
        'align_judge': 'mean'
    }).reset_index()
    
    # 3. Correlation-based sensitivity
    # This requires merging the original scores back to CF
    # For simplicity, we'll calculate it per event
    corr_data = []
    for eid in events:
        sub_panel = df_panel[df_panel['event_id'] == eid]
        
        # Calculate R_i and P_i for this event
        # Rank
        r_j = sub_panel['judge_total_score'].rank(ascending=False, method='min')
        r_v = sub_panel['fan_vote_estimate'].rank(ascending=False, method='min')
        R_i = r_j + r_v
        
        # Percent
        sum_j = sub_panel['judge_total_score'].sum()
        sum_v = sub_panel['fan_vote_estimate'].sum()
        P_i = (sub_panel['judge_total_score'] / sum_j) + (sub_panel['fan_vote_estimate'] / sum_v)
        
        # Risk: higher R_i is more risk, lower P_i is more risk
        # Spearman correlation with fan_vote_estimate and judge_total_score
        # Note: fan_vote_estimate (higher is better), judge_total_score (higher is better)
        # So we expect negative correlation with R_i and positive correlation with P_i
        # To make it "sensitivity", we look at the absolute correlation or flip signs
        
        # Sensitivity to fan: corr(risk, fan_estimate)
        # For Rank: risk = R_i. Higher R_i should mean lower fan_estimate.
        # For Percent: risk = -P_i. Higher -P_i (lower P_i) should mean lower fan_estimate.
        
        # Using Pearson as fallback since Spearman in pandas requires scipy
        corr_fan_rank = R_i.corr(sub_panel['fan_vote_estimate'], method='pearson')
        corr_judge_rank = R_i.corr(sub_panel['judge_total_score'], method='pearson')
        
        risk_pct = -P_i
        corr_fan_pct = risk_pct.corr(sub_panel['fan_vote_estimate'], method='pearson')
        corr_judge_pct = risk_pct.corr(sub_panel['judge_total_score'], method='pearson')
        
        corr_data.append({'event_id': eid, 'rule_variant': 'rank', 'sens_fan': -corr_fan_rank, 'sens_judge': -corr_judge_rank})
        corr_data.append({'event_id': eid, 'rule_variant': 'rank_js', 'sens_fan': -corr_fan_rank, 'sens_judge': -corr_judge_rank}) # approx
        corr_data.append({'event_id': eid, 'rule_variant': 'percent', 'sens_fan': -corr_fan_pct, 'sens_judge': -corr_judge_pct})
        corr_data.append({'event_id': eid, 'rule_variant': 'percent_js', 'sens_fan': -corr_fan_pct, 'sens_judge': -corr_judge_pct})

    df_corr = pd.DataFrame(corr_data)
    df_corr = df_corr.merge(df_cf[['event_id', 'season']].drop_duplicates(), on='event_id')
    corr_metrics = df_corr.groupby(['season', 'rule_variant']).agg({
        'sens_fan': 'mean',
        'sens_judge': 'mean'
    }).reset_index()
    
    # Merge all season metrics
    final_season_metrics = season_metrics.merge(bias_metrics, on=['season', 'rule_variant'], how='left')
    final_season_metrics = final_season_metrics.merge(corr_metrics, on=['season', 'rule_variant'], how='left')
    
    final_season_metrics.to_csv(out_season_csv, index=False)
    
    # Overall Metrics
    overall = final_season_metrics.groupby('rule_variant').agg({
        'exact_match_rate': 'mean',
        'align_fan': 'mean',
        'align_judge': 'mean',
        'sens_fan': 'mean',
        'sens_judge': 'mean'
    }).to_dict(orient='index')
    
    # Add Judges-Save impact
    # changed_elim_rate: pred_eliminated_ids differs between base and JS
    df_pivot = df_cf[df_cf['elim_count'] == 1].pivot(index='event_id', columns='rule_variant', values='pred_eliminated_ids')
    changed_rank = (df_pivot['rank'] != df_pivot['rank_js']).mean()
    changed_pct = (df_pivot['percent'] != df_pivot['percent_js']).mean()
    
    overall['js_impact'] = {
        'changed_elim_rate_rank': changed_rank,
        'changed_elim_rate_percent': changed_pct
    }
    
    with open(out_overall_json, 'w') as f:
        json.dump(overall, f, indent=4)
        
    print(f"Saved season metrics to {out_season_csv}")
    print(f"Saved overall metrics to {out_overall_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel_csv", default="../data/p2_interface_week_panel.csv")
    parser.add_argument("--cf_csv", default="../out/p2_week_counterfactual.csv")
    parser.add_argument("--out_season", default="../out/p2_season_metrics.csv")
    parser.add_argument("--out_overall", default="../out/p2_overall_metrics.json")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(__file__)
    calculate_metrics(
        os.path.join(base_dir, args.panel_csv),
        os.path.join(base_dir, args.cf_csv),
        os.path.join(base_dir, args.out_season),
        os.path.join(base_dir, args.out_overall)
    )
