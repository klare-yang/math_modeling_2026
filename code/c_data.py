# ============================================================
# Step 0: 导入库 & 读取原始数据
# 目的：加载数据，确认基本结构
# ============================================================

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

df_raw = pd.read_csv("./data/2026_MCM_Problem_C_Data.csv")

print("Raw data shape:", df_raw.shape)
print(df_raw.head())


# ============================================================
# Step 1: 明确 ID 列 与 评分列
# 目的：
# - ID 列：不会随 week / judge 变化
# - 评分列：week{n}_judge{m}_score
# ============================================================

id_vars = [
    "celebrity_name",
    "ballroom_partner",
    "celebrity_industry",
    "celebrity_homestate",
    "celebrity_homecountry/region",
    "celebrity_age_during_season",
    "season",
    "results",
    "placement"
]

score_cols = [
    c for c in df_raw.columns
    if c.startswith("week") and c.endswith("_score")
]

print("Number of score columns:", len(score_cols))


# ============================================================
# Step 2: 宽表 → 长表（melt）
# 目的：
# - 将每个选手的多周多评委分数，转换为
#   season–week–contestant–judge–score 的标准结构
# ============================================================

df_long = df_raw.melt(
    id_vars=id_vars,
    value_vars=score_cols,
    var_name="week_judge",
    value_name="score"
)


# ============================================================
# Step 3: 从列名中解析 week 与 judge
# 列名格式：week{数字}_judge{数字}_score
# ============================================================

df_long["week"] = (
    df_long["week_judge"]
    .str.extract(r"week(\d+)_judge\d+_score")
    .astype(int)
)

df_long["judge"] = (
    df_long["week_judge"]
    .str.extract(r"week\d+_judge(\d+)_score")
    .astype(int)
)

df_long = df_long.drop(columns="week_judge")

df_long = df_long.sort_values(
    ["season", "week", "celebrity_name", "judge"]
)

print("Long-format data preview:")
print(df_long.head())

# ============================================================
# Step 4: 缺失值处理（修正版，避免 groupby.apply 的 index 问题）
# ============================================================

# Step 4.1: 先按 season / contestant / judge 排序
df_long = df_long.sort_values(
    ["season", "celebrity_name", "judge", "week"]
)

# Step 4.2: 使用 groupby + transform 做线性插值
df_long["score"] = (
    df_long
    .groupby(["season", "celebrity_name", "judge"])["score"]
    .transform(lambda x: x.interpolate(method="linear"))
)

# Step 4.3: 仍有缺失 → 同赛季-同周-同评委中位数兜底
df_long["score"] = (
    df_long
    .groupby(["season", "week", "judge"])["score"]
    .transform(lambda x: x.fillna(x.median()))
)

print("Remaining missing scores:", df_long["score"].isna().sum())


# ============================================================
# Step 5: 计算每位选手每周的评委总分 J_{i,t}
# 目的：作为 Percent-based 计分制度的核心输入
# ============================================================

judge_sum = (
    df_long
    .groupby(["season", "week", "celebrity_name"])["score"]
    .sum()
    .reset_index(name="judge_total")
)

print(judge_sum.head())


# ============================================================
# Step 6: 解析淘汰周（results 字段）
# 目的：
# - 从文本中提取 “Eliminated Week k”
# - 得到每位选手的淘汰周 elim_week
# ============================================================

elim_info = (
    df_raw[["season", "celebrity_name", "results"]]
    .drop_duplicates()
    .copy()
)

elim_info["elim_week"] = (
    elim_info["results"]
    .str.extract(r"Week (\d+)")
    .astype(float)
)

print(elim_info.head())


# ============================================================
# Step 7: 合并评委总分 + 淘汰信息
# ============================================================

df_model = judge_sum.merge(
    elim_info[["season", "celebrity_name", "elim_week"]],
    on=["season", "celebrity_name"],
    how="left"
)

# ============================================================
# Step 8 (FIXED): 构造“是否仍在场（alive）”与“淘汰周标签”
# 关键修正：
# - 被淘汰于 Week k 的选手
#   ❌ 不应出现在 Week k 的 alive 集合中
#   ✅ 只在 week < elim_week 时 alive
# ============================================================

df_model["alive"] = (
    df_model["elim_week"].isna() |
    (df_model["week"] < df_model["elim_week"])
)

# 只保留真正参与当周比赛的选手
df_model = df_model[df_model["alive"]].copy()

# 标记“该选手是在上一周结束后被淘汰的”
# （用于分析，不用于集合）
df_model["eliminated_next"] = (
    df_model["week"] == df_model["elim_week"] - 1
)


# ============================================================
# Step 9: 计算评委分数占比（Percent-based 必需）
# 公式：
#   judge_percent_{i,t} = J_{i,t} / sum_j J_{j,t}
# ============================================================

df_model["judge_percent"] = (
    df_model["judge_total"] /
    df_model.groupby(["season", "week"])["judge_total"].transform("sum")
)

# ============================================================
# Step 11: 计算评委排名（Judge-based Rankings）
# ============================================================

# 1. 按评委总分的排名（1 = 最好）
df_model["judge_rank"] = (
    df_model
    .groupby(["season", "week"])["judge_total"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

# 2. 按评委占比的排名（理论上与 judge_rank 相同，但保留以便论文说明）
df_model["judge_percent_rank"] = (
    df_model
    .groupby(["season", "week"])["judge_percent"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

# ============================================================
# Step 10: 最终检查 & 导出
# 目的：确认数据已完全可用于建模
# ============================================================

print("Final modeling data preview:")
print(df_model.head())

print("Columns:", df_model.columns)
print("Any missing values:")
print(df_model.isna().sum())

# 可选：保存为建模输入文件
df_model.to_csv("./data/dwts_model_ready.csv", index=False)
