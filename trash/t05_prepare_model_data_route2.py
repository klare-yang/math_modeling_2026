import json
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_JSON = Path("data/T03_active_elimination_structure_route2.json")
OUTPUT_CSV = Path("data/T05_model_ready_events_route2.csv")

# 可选：过滤掉 elim_count=0 的事件
DROP_ZERO_ELIM = False  # True/False

# =========================
# 1) Load
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    structure = json.load(f)

records = []

for _, entry in structure.items():
    if entry["terminal"]:
        continue

    active_ids = entry["active_ids"]
    eliminated_ids = entry["eliminated_ids"]
    elim_count = int(entry["elim_count"])

    if DROP_ZERO_ELIM and elim_count == 0:
        continue

    # compute eliminated indices in active list (list)
    eliminated_indices = []
    for e in eliminated_ids:
        try:
            eliminated_indices.append(int(active_ids.index(e)))
        except ValueError:
            # theoretically should not happen; keep traceability
            eliminated_indices.append(None)

    records.append({
        "season": int(entry["season"]),
        "week": int(entry["week"]),
        "next_week": int(entry["next_week"]),
        "delta_week": int(entry["delta_week"]),
        "n_active": len(active_ids),
        "elim_count": elim_count,
        "active_ids": active_ids,                       # list column
        "eliminated_ids": eliminated_ids,               # list column
        "eliminated_indices": eliminated_indices        # list column
    })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print("\n=== T05 (Route2) SUMMARY ===")
print(f"Output: {OUTPUT_CSV}")
print(f"Rows: {len(df)}")
print("elim_count distribution:")
print(df["elim_count"].value_counts().sort_index().to_string())

print("\nSample rows:")
print(df.head(3).to_string(index=False))
