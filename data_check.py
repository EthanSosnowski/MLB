import pandas as pd
import numpy as np

# Load a sample year (e.g., 2024)
df = pd.read_parquet("statcast_final/final_2024.parquet")

print("--- DATA SANITY REPORT ---")
print(f"Total Rows: {len(df)}")

# 1. Check for 'Empty' Features
# If these percentages are high (e.g., > 10%), your rolling stats failed.
for col in ['batter_rolling_whiff_rate', 'pitcher_ff_usage']:
    missing = df[col].isna().sum()
    zeros = (df[col] == 0).sum()
    print(f"{col}: {missing} NaNs, {zeros} Zeros ({zeros/len(df):.1%} of data)")

# 2. Check Target Distribution
# If one zone is 90% of the data, the model will just guess that zone.
print("\nZone Distribution (Top 5):")
print(df['zone'].value_counts(normalize=True).head(5))

# 3. Correlation Check
# Does 'strikes' actually relate to the 'zone'? 
# In 2-strike counts, pitches should be in zones 11-14 more often.
strike_zone_rate = df[df['strikes'] == 2]['zone'].isin([11,12,13,14]).mean()
print(f"\nRate of 'Chase Zone' pitches with 2 strikes: {strike_zone_rate:.1%}")
