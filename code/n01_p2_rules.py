import pandas as pd
import numpy as np

def calculate_rank_method(df_event, elim_count):
    """
    df_event: DataFrame for a single event (one week)
    elim_count: number of contestants to eliminate
    """
    # 1. Judge rank (descending score, 1 is best)
    # Using 'min' method for ranks to handle ties (same score gets same rank)
    # Then we need to handle final elimination ties as per spec
    df = df_event.copy()
    df['r_j'] = df['judge_total_score'].rank(ascending=False, method='min')
    df['r_v'] = df['fan_vote_estimate'].rank(ascending=False, method='min')
    df['R_i'] = df['r_j'] + df['r_v']
    
    # 4. Elimination: k largest R_i
    # To handle ties stably: sort by R_i (desc), then judge_total_score (asc), then contestant_id (asc)
    df_sorted = df.sort_values(by=['R_i', 'judge_total_score', 'contestant_id'], 
                               ascending=[False, True, True])
    pred_eliminated = df_sorted.head(elim_count)['contestant_id'].tolist()
    return pred_eliminated, df_sorted

def calculate_percent_method(df_event, elim_count):
    """
    df_event: DataFrame for a single event (one week)
    elim_count: number of contestants to eliminate
    """
    df = df_event.copy()
    sum_j = df['judge_total_score'].sum()
    sum_v = df['fan_vote_estimate'].sum()
    
    if sum_j == 0 or sum_v == 0:
        # Per spec: throw error and log if sum is 0
        raise ValueError(f"Sum of scores or votes is zero in event {df.iloc[0]['event_id']}")
        
    df['p_j'] = df['judge_total_score'] / sum_j
    df['p_v'] = df['fan_vote_estimate'] / sum_v
    df['P_i'] = df['p_j'] + df['p_v']
    
    # 3. Elimination: k smallest P_i
    # To handle ties stably: sort by P_i (asc), then judge_total_score (asc), then contestant_id (asc)
    df_sorted = df.sort_values(by=['P_i', 'judge_total_score', 'contestant_id'], 
                               ascending=[True, True, True])
    pred_eliminated = df_sorted.head(elim_count)['contestant_id'].tolist()
    return pred_eliminated, df_sorted

def apply_judges_save(df_event, elim_count, base_method='rank'):
    """
    Applies the Judges-Save overlay if elim_count == 1
    """
    if elim_count != 1:
        if base_method == 'rank':
            elim, _ = calculate_rank_method(df_event, elim_count)
        else:
            elim, _ = calculate_percent_method(df_event, elim_count)
        return elim, []

    # Step 1: bottom-2
    if base_method == 'rank':
        _, df_sorted = calculate_rank_method(df_event, 2)
        # In rank method, df_sorted is sorted by R_i desc (worst first)
        bottom_2 = df_sorted.head(2)
    else:
        _, df_sorted = calculate_percent_method(df_event, 2)
        # In percent method, df_sorted is sorted by P_i asc (worst first)
        bottom_2 = df_sorted.head(2)
    
    bottom_2_ids = bottom_2['contestant_id'].tolist()
    
    # Step 2: judges choose whom to eliminate
    # Judges eliminate the one with LOWER judge_total_score
    # Tie-breaking: worse composite score, then contestant_id
    if base_method == 'rank':
        # bottom_2 is already sorted by R_i desc (worst first)
        # We want to eliminate the one with lower judge score. 
        # If judge scores same, eliminate the one with higher R_i (which is the first one in bottom_2)
        final_elim_df = bottom_2.sort_values(by=['judge_total_score', 'R_i', 'contestant_id'],
                                             ascending=[True, False, True])
    else:
        # bottom_2 is already sorted by P_i asc (worst first)
        # If judge scores same, eliminate the one with lower P_i (which is the first one in bottom_2)
        final_elim_df = bottom_2.sort_values(by=['judge_total_score', 'P_i', 'contestant_id'],
                                             ascending=[True, True, True])
        
    pred_eliminated = [final_elim_df.iloc[0]['contestant_id']]
    return pred_eliminated, bottom_2_ids
