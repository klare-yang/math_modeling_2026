import pandas as pd
import re
from pathlib import Path

# ====================================
# CONFIG
# ====================================
RAW_PATH = Path("data/2026_MCM_Problem_C_Data.csv")
OUTPUT_PATH = Path("data/T02_long_format_panel.csv")

# ====================================
# 1. 读取原始数据
# ====================================
df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape}")

# 获取非评分列（基本信息）
meta_cols = [col for col in df.columns if not re.match(r'week\d+_judge\d+', col)]

# 提取评分列（结构类似 week1_judge1, week2_judge3 ...）
score_cols = [col for col in df.columns if re.match(r'week\d+_judge\d+', col)]

# ====================================
# 2. 拆解为长表（melt）
# ====================================
df_long = df[meta_cols + score_cols].melt(
    id_vars=meta_cols,
    value_vars=score_cols,
    var_name='week_judge',
    value_name='score'
)

# Drop 全缺失的评分
df_long = df_long.dropna(subset=['score'])

# ====================================
# 3. 解析列名（week, judge）
# ====================================
def parse_week_judge(s):
    m = re.match(r'week(\d+)_judge(\d+)', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

df_long[['week', 'judge_id']] = df_long['week_judge'].apply(
    lambda s: pd.Series(parse_week_judge(s))
)

# ====================================
# 4. 聚合到每人每周（总分、有效评委数、0分个数）
# ====================================
agg_df = df_long.groupby(
    ['celebrity_name', 'season', 'week'],
    as_index=False
).agg(
    judge_score_total=('score', 'sum'),
    judge_score_count=('score', 'count'),
    raw_zero_count=('score', lambda x: (x == 0).sum())
)

# ====================================
# 5. 保存输出
# ====================================
agg_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved long-format panel data to: {OUTPUT_PATH}")
