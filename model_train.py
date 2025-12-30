import pandas as pd
import lightgbm as lgb
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Configuration
INPUT_DIR = "statcast_final"
MODEL_PATH = "pitch_zone_model.txt"

# 1. Select the features that actually help predict the zone
# We exclude things like 'game_pk' or 'game_date' which don't help prediction
FEATURES = [
    'balls', 'strikes', 'outs_when_up', 'inning', 'inning_topbot',
    'stand', 'p_throws', 'home_score', 'away_score',
    'on_1b', 'on_2b', 'on_3b',
    'batter_rolling_whiff_rate', 'pitcher_ff_usage',
    'prev_pitch_type', 'prev_zone'
]
TARGET = 'zone'

# Categorical features tell LightGBM to treat them as groups, not numbers
CAT_FEATURES = ['prev_pitch_type', 'prev_zone', 'inning_topbot', 'stand', 'p_throws']

def load_and_prepare_data():
    all_files = sorted(glob.glob(f"{INPUT_DIR}/final_*.parquet"))
    df_list = []
    
    for f in all_files:
        print(f"Loading {os.path.basename(f)}...")
        # Only load the columns we need to save RAM
        df_year = pd.read_parquet(f, columns=FEATURES + [TARGET])
        df_list.append(df_year)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # LightGBM prefers categorical columns to be type 'category'
    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')
        
    return df

def train_pitch_model():
    # Load Data
    df = load_and_prepare_data()
    print(f"Total Rows: {len(df)}")

    # Split into Features (X) and Target (y)
    X = df[FEATURES]
    y = df[TARGET]

    # Split into Training and Testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Starting LightGBM Training...")
    
    # Initialize the LightGBM Classifier
    # n_estimators: Number of trees. 500 is a good start for this much data.
    # learning_rate: Low rate helps accuracy but takes longer.
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        objective='multiclass',
        random_state=42,
        n_jobs=-1 # Use all CPU cores
    )

    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    # 4. Evaluate
    y_pred = model.predict(X_test)
    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Save the model
    model.booster_.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Feature Importance
    importance = pd.DataFrame({'feature': FEATURES, 'importance': model.feature_importances_})
    print("\nTop 5 Most Important Features:")
    print(importance.sort_values(by='importance', ascending=False).head(5))

if __name__ == "__main__":
    train_pitch_model()