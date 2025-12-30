import pandas as pd
import glob
import os

# Configuration
INPUT_DIR = "statcast_yearly"
OUTPUT_DIR = "statcast_cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_year_data(df):
    # 1. Filter out non-pitch events and missing targets
    # We can't predict a pitch if we don't know the type or the zone
    df = df.dropna(subset=['pitch_type', 'zone'])
    
    # 2. Remove non-competitive pitches
    # These distort "logic" because they aren't part of the pitcher/batter battle
    non_competitive = ['intent_ball', 'pitchout', 'automatic_ball', 'automatic_strike']
    df = df[~df['description'].isin(non_competitive)]
    
    # 3. Standardize Runner Columns
    # Statcast uses Player IDs. We want 1 (runner present) or 0 (base empty).
    runner_cols = ['on_1b', 'on_2b', 'on_3b']
    for col in runner_cols:
        df[col] = df[col].notna().astype(int)
        
    # 4. Standardize Handedness and Inning
    # Models prefer numbers (0/1) over strings ('R'/'L')
    mapping = {'R': 0, 'L': 1, 'top': 0, 'bot': 1}
    df['stand'] = df['stand'].map(mapping)
    df['p_throws'] = df['p_throws'].map(mapping)
    df['inning_topbot'] = df['inning_topbot'].map(mapping)
    
    # 5. Deduplication
    # Ensures every pitch in your 6.5M rows is unique
    df = df.drop_duplicates(subset=['game_pk', 'at_bat_number', 'pitch_number'])
    
    # 6. Date Conversion
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # 7. Create 'Count' strings (Useful for rolling stats later)
    # This turns '1 ball, 2 strikes' into '1-2'
    df['count'] = df['balls'].astype(int).astype(str) + "-" + df['strikes'].astype(int).astype(str)
    
    return df

def run_cleaning_pipeline():
    all_files = sorted(glob.glob(f"{INPUT_DIR}/statcast_*.csv"))
    
    for file_path in all_files:
        year = os.path.basename(file_path).split('_')[1].split('.')[0]
        print(f"Cleaning data for {year}...")
        
        # Read data
        df = pd.read_csv(file_path, low_memory=False)
        
        # Clean data
        df_clean = clean_year_data(df)
        
        # Save as Parquet for the next step (Feature Engineering)
        # Parquet is 10x faster to load and much smaller than CSV
        output_path = f"{OUTPUT_DIR}/clean_{year}.parquet"
        df_clean.to_parquet(output_path, index=False)
        print(f"Success: {year} saved with {len(df_clean)} rows.")

if __name__ == "__main__":
    run_cleaning_pipeline()