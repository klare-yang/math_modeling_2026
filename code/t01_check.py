import pandas as pd

# ===============================
# T01: Load raw data
# ===============================
DATA_PATH = "./data/2026_MCM_Problem_C_Data.csv"  # 确保文件在当前工作目录
df = pd.read_csv(DATA_PATH)

# ===============================
# Basic info
# ===============================
n_rows = len(df)

schema_records = []

for col in df.columns:
    col_dtype = df[col].dtype
    missing_ratio = df[col].isna().mean()  # 缺失比例
    unique_count = df[col].nunique(dropna=True)

    schema_records.append({
        "column_name": col,
        "dtype": str(col_dtype),
        "missing_ratio": missing_ratio,
        "unique_non_null_values": unique_count
    })

schema_df = pd.DataFrame(schema_records)

# ===============================
# Output
# ===============================
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)

print("=== Dataset Shape ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

print("=== Column Schema Summary ===")
print(schema_df)

# ===============================
# (Optional) Save schema summary
# ===============================
schema_df.to_csv("T01_schema_summary.csv", index=False)
print("\nSchema summary saved to T01_schema_summary.csv")
