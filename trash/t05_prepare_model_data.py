# T05: 读取 JSON 替代 pkl
import json
import pandas as pd
from pathlib import Path

INPUT_JSON = Path("data/T03_active_elimination_structure.json")
OUTPUT_CSV = Path("data/T05_model_ready_observations.csv")

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    structure = json.load(f)

records = []

for key, entry in structure.items():
    active = entry["active"]
    eliminated = entry["eliminated"]
    season = entry["season"]
    week = entry["week"]

    if eliminated is None or isinstance(eliminated, list):
        continue

    try:
        eliminated_index = active.index(eliminated)
    except ValueError:
        continue  # skip if index lookup fails

    records.append({
        "season": season,
        "week": week,
        "active_ids": active,
        "eliminated_id": eliminated,
        "eliminated_index": eliminated_index
    })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved model-ready observation data to: {OUTPUT_CSV}")
