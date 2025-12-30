import pandas as pd
import numpy as np
import glob
import os

# Configuration
INPUT_DIR = "statcast_cleaned"
OUTPUT_DIR = "statcast_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_at_bat_sequence_features(df):
    """
    Creates features based on what happened earlier in the SAME at-bat.
    """
    # Sort to ensure the sequence is correct: Game -> At-Bat -> Pitch Number
    df = df.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
    
    # Group by the unique at-bat identifier
    group = df.groupby(['game_pk', 'at_bat_number'])
    
    # 1. Previous Pitch Type and Zone
    # shift(1) looks at the row directly above within the same at-bat
    df['prev_pitch_type'] = group['pitch_type'].shift(1)
    df['prev_zone'] = group['zone'].shift(1)
    
    # 2. Pitch Count in At-Bat
    # (Already handled by pitch_number, but useful to keep)
    
    # 3. Fill 'First Pitch' logic
    # The first pitch of an at-bat has no 'previous'. We label it 'START'
    df['prev_pitch_type'] = df['prev_pitch_type'].fillna('START')
    df['prev_zone'] = df['prev_zone'].fillna(0) # Zone 0 = No previous zone
    
    return df

def add_rolling_stats(df):
    """
    Calculates how the batter and pitcher have been performing recently.
    """
    # Create binary outcome flags
    whiff_strings = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt']
    df['is_whiff'] = df['description'].isin(whiff_strings).astype(int)
    
    # Sort by batter and date for rolling calculations
    df = df.sort_values(['batter', 'game_date'])
    
    # Batter Rolling Whiff Rate (Last 100 pitches seen)
    # closed='left' is CRITICAL: it prevents the model from seeing the current pitch outcome
    df['batter_rolling_whiff_rate'] = df.groupby('batter')['is_whiff'].transform(
        lambda x: x.rolling(window=100, min_periods=10, closed='left').mean()
    )
    
    # Pitcher Usage (Last 50 pitches thrown)
    # We want to know if the pitcher is currently relying on their Fastball (FF)
    df['is_fastball'] = (df['pitch_type'] == 'FF').astype(int)
    df['pitcher_ff_usage'] = df.groupby('pitcher')['is_fastball'].transform(
        lambda x: x.rolling(window=50, min_periods=5, closed='left').mean()
    )
    
    return df

def run_feature_pipeline():
    all_files = sorted(glob.glob(f"{INPUT_DIR}/clean_*.parquet"))
    
    for file_path in all_files:
        year = os.path.basename(file_path).split('_')[1].split('.')[0]
        print(f"Engineering features for {year}...")
        
        df = pd.read_parquet(file_path)
        
        # Apply At-Bat Sequences
        df = add_at_bat_sequence_features(df)
        
        # Apply Rolling Performance
        df = add_rolling_stats(df)
        
        # Fill any remaining NaNs with global means
        df['batter_rolling_whiff_rate'] = df['batter_rolling_whiff_rate'].fillna(df['is_whiff'].mean())
        df['pitcher_ff_usage'] = df['pitcher_ff_usage'].fillna(0.5) # Assume 50% if unknown
        
        # Save final version
        output_path = f"{OUTPUT_DIR}/final_{year}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Final data saved: {year}")

if __name__ == "__main__":
    run_feature_pipeline()