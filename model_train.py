import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
INPUT_FILE = "final_data.parquet"

# Updated to include "Archetype" and "Context" signals
FEATURES = [
    'pitcher_style', 'batter_weak_zone',  # Archetypes
    'pitcher', 'batter',                  # IDs
    'balls', 'strikes', 'outs_when_up',   # Count/Situation
    'stand', 'p_throws', 'on_1b', 'on_2b', 'on_3b',
    'score_diff', 'is_late_inning',       # Game Context
    'batter_rolling_whiff_rate', 'pitcher_ff_usage',
    'prev_pitch_type', 'prev_zone'
]

# Tell LightGBM which features are labels/categories
CAT_FEATURES = [
    'pitcher_style', 'batter_weak_zone', 'pitcher', 'batter', 
    'prev_pitch_type', 'prev_zone', 'stand', 'p_throws'
]


# --- 3. MAIN TRAINING LOOP ---
def train_dual_optimized():
    print("Loading data...")
    df = pd.read_parquet(INPUT_FILE)    
    # Critical: Convert to category types
    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')

    X = df[FEATURES]
    
    # # --- MODEL 1: ARSENAL (Pitch Type) ---
    # print("\n--- Training Model 1: Pitch Type ---")
    # y_type = df['pitch_type'].astype('category')
    # X_train, X_test, y_train, y_test = train_test_split(X, y_type, test_size=0.2, random_state=42)
    
    # type_model = lgb.LGBMClassifier(
    #     n_estimators=2000,
    #     learning_rate=0.02, 
    #     num_leaves=255,        # Increased from 128 to handle new Archetypes
    #     class_weight='balanced',
    #     objective='multiclass',
    #     n_jobs=-1
    # )
    # type_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
    #                callbacks=[lgb.early_stopping(stopping_rounds=50)])
    # type_model.booster_.save_model("model_type_optimized.txt")


    # --- MODEL 2: LOCATION (Zone) ---
    print("\n--- Training Model 2: Zone ---")
    y_zone = df['zone']
    X_train, X_test, y_train, y_test = train_test_split(X, y_zone, test_size=0.2, random_state=42)
    
    # The "Deep Brain" setup to move past the 5.5% wall
    zone_model = lgb.LGBMClassifier(
        num_leaves=127,            # Reduced from 1023 (HUGE speed boost)
        max_depth=10,              # Limits tree depth to prevent memorization
        min_child_samples=100,     # Higher value prevents noise from relievers
        learning_rate=0.05,        # 10x faster than 0.005; still very precise
        n_estimators=1500,         # 1500 trees is plenty with a 0.05 rate
        subsample=0.8,             # Uses 80% of data per tree (speeds up training)
        colsample_bytree=0.7,      # Same as your feature_fraction
        objective='multiclass',
        n_jobs=-1,                 # Uses all your laptop cores
        random_state=42
    )
    zone_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50),lgb.log_evaluation(period=100)])
    zone_model.booster_.save_model("model_zone_optimized.txt")

    print("\nSUCCESS: Both Optimized Models Saved with Archetype Logic!")

if __name__ == "__main__":
    train_dual_optimized()