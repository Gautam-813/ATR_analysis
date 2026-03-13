
import pandas as pd

# Load the parquet file
file_path = r"d:\date-wise\03-03-2026\atr_analysis_data\XAUUSD_M1_Data.parquet"
df = pd.read_parquet(file_path)

# Display basic info
print("Data Overview:")
print(f"Shape: {df.shape}")
print("\nColumns and Data Types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

# Check time range if 'time' or 'date' column exists
time_col = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
if time_col:
    print(f"\nTime Range: {df[time_col[0]].min()} to {df[time_col[0]].max()}")
