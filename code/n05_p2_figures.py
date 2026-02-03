import pandas as pd
import matplotlib.pyplot as plt
import os
import json

def generate_figures(season_csv, overall_json, fig_dir):
    df_s = pd.read_csv(season_csv)
    with open(overall_json, 'r') as f:
        overall = json.load(f)
    
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Exact Match Rate by Season (Rank vs Percent)
    plt.figure(figsize=(12, 6))
    for var in ['rank', 'percent']:
        sub = df_s[df_s['rule_variant'] == var]
        plt.plot(sub['season'], sub['exact_match_rate'], marker='o', label=var.capitalize())
    plt.title('Exact Match Rate by Season')
    plt.xlabel('Season')
    plt.ylabel('Match Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, 'p2_match_rate_season.png'))
    plt.close()
    
    # 2. Overturn Rate / Alignment (Overall)
    variants = ['rank', 'percent', 'rank_js', 'percent_js']
    align_fan = [overall[v]['align_fan'] for v in variants]
    align_judge = [overall[v]['align_judge'] for v in variants]
    
    x = range(len(variants))
    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], align_fan, width=0.4, label='Align with Fan-Worst', color='skyblue')
    plt.bar([i + 0.2 for i in x], align_judge, width=0.4, label='Align with Judge-Worst', color='salmon')
    plt.xticks(x, [v.upper() for v in variants])
    plt.ylabel('Probability of Elimination')
    plt.title('Bias Comparison: Alignment with Fan vs Judge Worst')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, 'p2_bias_comparison.png'))
    plt.close()
    
    # 3. Judges-Save Impact
    impact = overall['js_impact']
    labels = ['Rank to Rank-JS', 'Percent to Percent-JS']
    values = [impact['changed_elim_rate_rank'], impact['changed_elim_rate_percent']]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='lightgreen')
    plt.ylabel('Rate of Changed Elimination')
    plt.title('Impact of Judges-Save Overlay')
    plt.ylim(0, max(values) * 1.2 if values else 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.005, f'{v:.1%}', ha='center')
    plt.savefig(os.path.join(fig_dir, 'p2_js_impact.png'))
    plt.close()
    
    print(f"Generated figures in {fig_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season_csv", default="../out/p2_season_metrics.csv")
    parser.add_argument("--overall_json", default="../out/p2_overall_metrics.json")
    parser.add_argument("--fig_dir", default="../fig")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(__file__)
    generate_figures(
        os.path.join(base_dir, args.season_csv),
        os.path.join(base_dir, args.overall_json),
        os.path.join(base_dir, args.fig_dir)
    )
