import pandas as pd

df = pd.read_csv("XAUUSD_M1_Data_merged.csv")
df.to_parquet("XAUUSD_M1_Data.parquet")