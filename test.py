import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 1. CONFIGURATION ---
MODEL_ZONE_PATH = "model_zone_optimized.txt"
MODEL_TYPE_PATH = "model_type_optimized.txt"
DATA_PATH = "statcast_final/final_2024.parquet"

FEATURES = [
    'pitcher', 'batter', 'balls', 'strikes', 'outs_when_up', 
    'stand', 'p_throws', 'on_1b', 'on_2b', 'on_3b',
    'batter_rolling_whiff_rate', 'pitcher_ff_usage',
    'prev_pitch_type', 'prev_zone'
]
CAT_FEATURES = ['pitcher', 'batter', 'prev_pitch_type', 'prev_zone', 'stand', 'p_throws']

def calculate_top_k(probs, y_true, k=3):
    """Calculates if the true value is within the top K predicted probabilities."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    # Convert y_true to a numpy array for faster indexing
    y_true_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
    matches = [y_true_arr[i] in top_k_preds[i] for i in range(len(y_true))]
    return np.mean(matches)

def run_full_test():
    # --- 2. LOAD DATA & MODELS ---
    print("Loading data and models...")
    try:
        df = pd.read_parquet(DATA_PATH)
        zone_model = lgb.Booster(model_file=MODEL_ZONE_PATH)
        type_model = lgb.Booster(model_file=MODEL_TYPE_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure your .txt files and parquet data are in the correct folder.")
        return

    # --- 3. DATA PREPARATION ---
    # Crucial: Must match training categories exactly
    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')

    X = df[FEATURES]
    y_zone = df['zone']
    y_type = df['pitch_type'].astype('category').cat.codes # Convert pitch type strings to codes if necessary

    # Split to get a clean test set (20% of data)
    _, X_test, y_zone_train, y_zone_test = train_test_split(X, y_zone, test_size=0.2, random_state=42)
    _, _, y_type_train, y_type_test = train_test_split(X, y_type, test_size=0.2, random_state=42)

    # --- 4. TEST ZONE MODEL ---
    print("\n" + "="*30)
    print(" TESTING ZONE MODEL (Location)")
    print("="*30)
    
    zone_probs = zone_model.predict(X_test)
    zone_preds = np.argmax(zone_probs, axis=1)
    
    acc_z1 = accuracy_score(y_zone_test, zone_preds)
    acc_z3 = calculate_top_k(zone_probs, y_zone_test, k=3)
    
    print(f"Top-1 Accuracy (Exact Square): {acc_z1:.2%}")
    print(f"Top-3 Accuracy (Neighborhood): {acc_z3:.2%}")

    # --- 5. TEST PITCH TYPE MODEL ---
    print("\n" + "="*30)
    print(" TESTING PITCH TYPE MODEL (Arsenal)")
    print("="*30)
    
    type_probs = type_model.predict(X_test)
    type_preds = np.argmax(type_probs, axis=1)
    
    acc_t1 = accuracy_score(y_type_test, type_preds)
    acc_t2 = calculate_top_k(type_probs, y_type_test, k=2)
    
    print(f"Top-1 Accuracy (Exact Pitch):  {acc_t1:.2%}")
    print(f"Top-2 Accuracy (Pitch Group):  {acc_t2:.2%}")

    # --- 6. SITUATIONAL ACCURACY (0-2 Counts) ---
    print("\n--- Situational: 0-2 Counts ---")
    mask_02 = (X_test['balls'] == 0) & (X_test['strikes'] == 2)
    if mask_02.any():
        preds_02 = np.argmax(zone_model.predict(X_test[mask_02]), axis=1)
        acc_02 = accuracy_score(y_zone_test[mask_02], preds_02)
        print(f"Zone Accuracy on 0-2 Counts: {acc_02:.2%}")
    else:
        print("Not enough 0-2 counts in test sample.")

if __name__ == "__main__":
    run_full_test()