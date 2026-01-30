import pandas as pd
from pathlib import Path
import json
from collections import defaultdict

# ====================================
# CONFIG
# ====================================
INPUT_PATH = Path("data/T02_long_format_panel.csv")
OUTPUT_PATH_JSON = Path("data/T03_active_elimination_structure.json")

# ====================================
# 1. 读取数据
# ====================================
df = pd.read_csv(INPUT_PATH)
df = df.sort_values(by=["season", "week", "celebrity_name"])

# ====================================
# 2. 构建每周 active 映射
# ====================================
active_map = {
    (season, week): sorted(set(group["celebrity_name"]))
    for (season, week), group in df.groupby(["season", "week"])
}

# ====================================
# 3. 差分法生成淘汰结构
# ====================================
structure = dict()
all_keys = sorted(active_map.keys())

for (season, week) in all_keys:
    active_now = set(active_map[(season, week)])
    next_week = (season, week + 1)
    active_next = set(active_map.get(next_week, []))

    eliminated = sorted(list(active_now - active_next))

    if len(eliminated) == 0:
        elim_value = None
    elif len(eliminated) == 1:
        elim_value = eliminated[0]
    else:
        elim_value = eliminated  # 多淘汰者作为列表保留

    structure[f"{season}-{week}"] = {
        "season": int(season),
        "week": int(week),
        "active": sorted(list(active_now)),
        "eliminated": elim_value
    }

# ====================================
# 4. 输出结构检查摘要
# ====================================
weeks_with_elim = sum(1 for v in structure.values() if v["eliminated"] is not None)
multi_elim_weeks = sum(1 for v in structure.values() if isinstance(v["eliminated"], list))

print("\n=== STRUCTURE SUMMARY ===")
print(f"Total weeks: {len(structure)}")
print(f"Weeks with elimination: {weeks_with_elim}")
print(f"Weeks with multiple elimination: {multi_elim_weeks}")

print("\nSample (first 5 eliminated):")
for k in sorted(structure.keys())[:5]:
    elim = structure[k]["eliminated"]
    print(f"  Week {k}: eliminated = {elim}")

# ====================================
# 5. 保存为 JSON
# ====================================
with open(OUTPUT_PATH_JSON, "w", encoding="utf-8") as f:
    json.dump(structure, f, ensure_ascii=False, indent=2)

print(f"\nSaved elimination structure to: {OUTPUT_PATH_JSON}")
