import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

# 1. Configuration
INPUT_FILE = "statcast_final/final_2024.parquet"
FEATURES = [
    'pitcher', 'batter', 'balls', 'strikes', 'outs_when_up', 
    'stand', 'p_throws', 'on_1b', 'on_2b', 'on_3b',
    'batter_rolling_whiff_rate', 'pitcher_ff_usage',
    'prev_pitch_type', 'prev_zone'
]
CAT_FEATURES = ['pitcher', 'batter', 'prev_pitch_type', 'prev_zone', 'stand', 'p_throws']

def train_dual_optimized():
    print("Loading 2024 data...")
    df = pd.read_parquet(INPUT_FILE)
    
    # Critical: Convert to category types
    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')

    X = df[FEATURES]
    
    # --- MODEL 1: ARSENAL (Pitch Type) ---
    print("\n--- Training Model 1: Pitch Type ---")
    y_type = df['pitch_type'].astype('category')
    X_train, X_test, y_train, y_test = train_test_split(X, y_type, test_size=0.2, random_state=42)
    
    type_model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03, # Slightly faster for pitch type as it's a clearer signal
        num_leaves=128,
        class_weight='balanced',
        objective='multiclass',
        n_jobs=-1
    )
    type_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                   callbacks=[lgb.early_stopping(stopping_rounds=50)])
    type_model.booster_.save_model("model_type_optimized.txt")

    # --- MODEL 2: LOCATION (Zone) ---
    print("\n--- Training Model 2: Zone ---")
    y_zone = df['zone']
    X_train, X_test, y_train, y_test = train_test_split(X, y_zone, test_size=0.2, random_state=42)
    
    zone_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01, # Slower for Zone because it's "noisier"
        num_leaves=255,
        class_weight='balanced',
        objective='multiclass',
        n_jobs=-1
    )
    zone_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                   callbacks=[lgb.early_stopping(stopping_rounds=50)])
    zone_model.booster_.save_model("model_zone_optimized.txt")

    print("\nSUCCESS: Both Optimized Models Saved!")

if __name__ == "__main__":
    train_dual_optimized()