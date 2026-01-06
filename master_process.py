import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURATION ---
DATA_FOLDER = "statcast_yearly"
OUTPUT_FILE = "final_data.parquet"
WINDOW_SIZE = 100  # Number of pitches for batter rolling metrics

def process_master_data():
    file_pattern = os.path.join(DATA_FOLDER, "*.csv")
    files = [f for f in glob.glob(file_pattern) if "deep_brain" not in f]
    
    if not files:
        print(f" No data files found in {DATA_FOLDER}")
        return

    print(f" Found {len(files)} files. Merging into master dataset...")
    
    # Load and combine all files
    df_list = []
    for f in files:
        print(f"Reading: {os.path.basename(f)}")
        df_list.append(pd.read_csv(f))
    
    df = pd.concat(df_list, ignore_index=True)

    # 1. CHRONOLOGICAL SORTING
    # Necessary for rolling statistics to be accurate
    print(" Sorting data by time...")
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['game_date', 'at_bat_number', 'pitch_number'])

    # 2. BASIC CLEANING & NULL HANDLING
    print(" Cleaning missing values...")
    df = df.dropna(subset=['pitch_type', 'zone'])
    df = df.drop_duplicates(subset=['game_pk', 'at_bat_number', 'pitch_number'])
    df['game_date'] = pd.to_datetime(df['game_date'])

    df['prev_pitch_type'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(1)
    df['prev_zone'] = df.groupby(['game_pk', 'at_bat_number'])['zone'].shift(1)

    # Fill the first pitch of every at-bat with 'START' and 0
    df['prev_pitch_type'] = df['prev_pitch_type'].fillna('START')
    df['prev_zone'] = df['prev_zone'].fillna(0)
    
    # Convert runners to binary flags (0 or 1)
    for col in ['on_1b', 'on_2b', 'on_3b']:
        df[col] = df[col].notnull().astype(int)
        
    mapping = {'R': 0, 'L': 1, 'top': 0, 'bot': 1}
    df['stand'] = df['stand'].map(mapping)
    df['p_throws'] = df['p_throws'].map(mapping)
    df['inning_topbot'] = df['inning_topbot'].map(mapping)

    # 3. GLOBAL PITCHER DNA (Average Velo and Spin)
    print(" Calculating Season-Long Pitcher 'Stuff' DNA...")
    # We calculate these globally so every pitch by a player knows their "baseline"
    pitcher_stats = df.groupby('pitcher').agg({
        'release_speed': 'mean',
        'release_spin_rate': 'mean'
    }).rename(columns={
        'release_speed': 'pitcher_avg_velo', 
        'release_spin_rate': 'pitcher_avg_spin'
    })
    
    # Merge averages back to the main dataframe
    df = df.merge(pitcher_stats, on='pitcher', how='left')
    
    # Fill missing averages with league medians so the model doesn't crash
    df['pitcher_avg_velo'] = df['pitcher_avg_velo'].fillna(df['pitcher_avg_velo'].median())
    df['pitcher_avg_spin'] = df['pitcher_avg_spin'].fillna(df['pitcher_avg_spin'].median())

    # 4. PITCHER ARCHETYPES (Power vs. Crafty)
    print(" Classifying Pitcher Archetypes...")
    v_med = df['pitcher_avg_velo'].median()
    s_med = df['pitcher_avg_spin'].median()

    def get_style(row):
        if row['pitcher_avg_velo'] > v_med and row['pitcher_avg_spin'] > s_med: 
            return 'Power_HighSpin'
        if row['pitcher_avg_velo'] > v_med: 
            return 'Power_Sink'
        if row['pitcher_avg_spin'] > s_med: 
            return 'Crafty_Spin'
        return 'Crafty_Finesse'

    df['pitcher_style'] = df.apply(get_style, axis=1)

    # 5. BATTER STATS: ROLLING WHIFF RATE
    print(" Calculating Batter Rolling Whiff Rates...")
    swings = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip']
    whiffs = ['swinging_strike', 'swinging_strike_blocked']
    
    df['is_whiff'] = df['description'].isin(whiffs).astype(int)

    # Rolling average of the last 100 pitches seen by that batter
    df['batter_rolling_whiff_rate'] = df.groupby('batter')['is_whiff'].transform(
        lambda x: x.rolling(window=WINDOW_SIZE, min_periods=10).mean()
    ).fillna(0.25)

    # 6. BATTER WEAK ZONES (The "Hunting" Signal)
    print(" Mapping Batter Vulnerability Zones...")
    whiff_only = df[df['is_whiff'] == 1]
    # Find the zone where the batter has the most swinging strikes
    batter_weaknesses = whiff_only.groupby('batter')['zone'].agg(
        lambda x: x.value_counts().index[0] if not x.empty else 14
    ).reset_index().rename(columns={'zone': 'batter_weak_zone'})
    
    df = df.merge(batter_weaknesses, on='batter', how='left')
    df['batter_weak_zone'] = df['batter_weak_zone'].fillna(14)

    # 7. GAME CONTEXT (Pressure Logic)
    print(" Adding Game Context & Leverage...")
    df['score_diff'] = df['home_score'] - df['away_score']
    df['is_late_inning'] = (df['inning'] >= 7).astype(int)
    
    # Fastball Usage - Identifying if they are a "one-trick" pitcher
    df['is_ff'] = (df['pitch_type'] == 'FF').astype(int)
    df['pitcher_ff_usage'] = df.groupby('pitcher')['is_ff'].transform(
        lambda x: x.expanding(min_periods=20).mean()
    ).fillna(0.35)

    # 8. FINAL CLEANUP & SAVE
    print(f" Saving Enhanced Dataset to {OUTPUT_FILE}...")
    # Drop temporary helper columns
    df.drop(columns=['is_whiff', 'is_ff'], inplace=True)
    print(df.head(10))
    # Save as parquet for speed
    df.to_parquet(OUTPUT_FILE, index=False)
    print(" SUCCESS: Data processing complete.")

if __name__ == "__main__":
    process_master_data()