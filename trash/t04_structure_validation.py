import pickle
from pathlib import Path
from collections import defaultdict

# ====================================
# CONFIG
# ====================================
STRUCT_PATH = Path("data/T03_active_elimination_structure.pkl")

# ====================================
# 1. 读取结构
# ====================================
with open(STRUCT_PATH, "rb") as f:
    structure = pickle.load(f)

# 按时间排序
time_keys = sorted(structure.keys())

# ====================================
# 2. 构建参赛记录表
# ====================================
# 每个选手在哪些 week 参赛
active_weeks = defaultdict(list)
# 每个选手在哪些 week 被淘汰
elim_weeks = defaultdict(list)

for (season, week), info in structure.items():
    active = info["active"]
    eliminated = info["eliminated"]

    for name in active:
        active_weeks[name].append((season, week))

    if eliminated is not None:
        if isinstance(eliminated, list):
            for name in eliminated:
                elim_weeks[name].append((season, week))
        else:
            elim_weeks[eliminated].append((season, week))

# ====================================
# 3. 检查规则
# ====================================
error_1 = []  # 淘汰后仍参赛
error_2 = []  # 多次淘汰
error_3 = []  # active 为空
error_4 = []  # 淘汰者不在 active 中
error_5 = []  # 淘汰周非最后参赛
warning_6 = []  # 多淘汰者（可接受）

for name in elim_weeks:
    elim_times = sorted(elim_weeks[name])
    if len(elim_times) > 1:
        error_2.append((name, elim_times))

    last_elim = elim_times[-1]
    for (s, w) in active_weeks[name]:
        if (s, w) > last_elim:
            error_1.append((name, last_elim, (s, w)))
            break

for (season, week), info in structure.items():
    if len(info["active"]) == 0:
        error_3.append((season, week))
    
    eliminated = info["eliminated"]
    if eliminated is not None:
        if isinstance(eliminated, list):
            warning_6.append((season, week, eliminated))
            for e in eliminated:
                if e not in info["active"]:
                    error_4.append((season, week, e))
        else:
            if eliminated not in info["active"]:
                error_4.append((season, week, eliminated))
            
            # 淘汰者若后续还有参赛记录也算错误（但已在 error_1 中体现）
            if eliminated in active_weeks:
                last_seen = max(active_weeks[eliminated])
                if last_seen > (season, week):
                    error_5.append((eliminated, (season, week), last_seen))

# ====================================
# 4. 输出诊断报告
# ====================================
print("\n=== STRUCTURE VALIDATION REPORT ===\n")
print(f"Total weeks: {len(structure)}")
print(f"Players with multiple eliminations: {len(error_2)}")
print(f"Players active after elimination: {len(error_1)}")
print(f"Weeks with no active players: {len(error_3)}")
print(f"Eliminated not in active set: {len(error_4)}")
print(f"Eliminated but later reappeared: {len(error_5)}")
print(f"Weeks with multi-elimination (warn only): {len(warning_6)}")

if warning_6:
    print("\n⚠️ Multi-elimination Weeks:")
    for season, week, eliminated in warning_6:
        print(f"  S{season} W{week}: {eliminated}")
