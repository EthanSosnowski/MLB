import os
import time
import pandas as pd
import datetime
import glob
from pybaseball import statcast, cache

# Enable local caching to speed up retries
cache.enable()

# Configuration
RAW_CHUNKS_DIR = "statcast_chunks"
YEARLY_DIR = "statcast_yearly"
os.makedirs(RAW_CHUNKS_DIR, exist_ok=True)
os.makedirs(YEARLY_DIR, exist_ok=True)

USE_COLS = [
    "pitch_type", "pitch_name", "batter", "pitcher", "stand", "p_throws",
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "game_pk", "game_date", "at_bat_number", "pitch_number",
    "zone", "home_score", "away_score", "on_1b", "on_2b", "on_3b", 
    "events", "description","release_speed","release_spin_rate",""
]

def safe_statcast(start_dt, end_dt, retries=3):
    """Downloads data with a retry mechanism."""
    for attempt in range(1, retries + 1):
        try:
            df = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            time.sleep(5)
    return None

def download_all_data(start_year=2015, end_year=2024):
    for year in range(start_year, end_year + 1):
        print(f"\n>>> STARTING DOWNLOAD FOR {year}")
        current_date = datetime.date(year, 3, 25)
        season_end = datetime.date(year, 11, 5)

        while current_date <= season_end:
            window_end = min(current_date + datetime.timedelta(days=5), season_end)
            
            s_str = current_date.strftime('%Y-%m-%d')
            e_str = window_end.strftime('%Y-%m-%d')
            filename = f"{RAW_CHUNKS_DIR}/sc_{s_str}_{e_str}.csv"

            if os.path.exists(filename):
                current_date = window_end + datetime.timedelta(days=1)
                continue

            print(f"Fetching {s_str} to {e_str}...")
            df_chunk = safe_statcast(s_str, e_str)

            if df_chunk is not None and not df_chunk.empty:
                cols_to_save = [c for c in USE_COLS if c in df_chunk.columns]
                df_chunk[cols_to_save].to_csv(filename, index=False)
            else:
                print(f"  Warning: No data for {s_str}")

            current_date = window_end + datetime.timedelta(days=1)
            time.sleep(1) 

def combine_into_years(start_year=2015, end_year=2024):
    print("\n>>> COMBINING CHUNKS INTO YEARLY FILES")
    for year in range(start_year, end_year + 1):
        year_files = glob.glob(f"{RAW_CHUNKS_DIR}/sc_{year}-*.csv")
        
        if not year_files:
            print(f"No files found for {year}")
            continue
            
        print(f"Combining {year} ({len(year_files)} chunks)...")
        
        yearly_df = pd.concat((pd.read_csv(f) for f in year_files), ignore_index=True)
        
        yearly_df = yearly_df.drop_duplicates(subset=['game_pk', 'at_bat_number', 'pitch_number'])
        
        yearly_df = yearly_df.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
        
        output_path = f"{YEARLY_DIR}/statcast_{year}.csv"
        yearly_df.to_csv(output_path, index=False)
        print(f"Saved {output_path} | Rows: {len(yearly_df)}")

if __name__ == "__main__":
    download_all_data()
    combine_into_years()