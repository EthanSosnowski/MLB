import pandas as pd
import numpy as np
import lightgbm as lgb

def get_filtered_accuracy(model_path, data, target_col):
    # 1. Load Model
    model = lgb.Booster(model_file=model_path)
    expected_features = model.feature_name()
    
    # 2. Get the Universal Category Mapping (Alphabetical)
    all_possible_pitches = sorted(df[target_col].unique().tolist())
    data[target_col] = pd.Categorical(data[target_col], categories=all_possible_pitches)
    y_true = data[target_col].cat.codes.values
    
    # 3. BUILD THE REPERTOIRE MASK
    # We create a mapping of pitcher -> list of pitch codes they actually throw
    print("🧠 Building Repertoire Mask...")
    repertoire_map = df.groupby('pitcher')[target_col].unique().apply(
        lambda x: [all_possible_pitches.index(p) for p in x]
    ).to_dict()

    # 4. Prepare Features
    X = data[expected_features].copy()
    for col in X.columns:
        if X[col].dtype.name == 'category' or X[col].dtype == 'object':
            all_col_vals = sorted(df[col].unique().tolist())
            X[col] = pd.Categorical(X[col], categories=all_col_vals).codes
        X[col] = X[col].fillna(0).astype(np.float32)

    # 5. Get Raw Probabilities
    raw_probs = model.predict(X.values) # Shape: (50000, 18)
    
    # 6. APPLY THE MASK
    # For every row, find the pitcher and zero out pitches they don't throw
    filtered_probs = raw_probs.copy()
    pitcher_ids = data['pitcher'].values
    
    for i, p_id in enumerate(pitcher_ids):
        allowed_indices = repertoire_map.get(p_id, [])
        # Create a mask of zeros
        mask = np.zeros(len(all_possible_pitches))
        mask[allowed_indices] = 1
        # Multiply probabilities by the mask
        filtered_probs[i] = filtered_probs[i] * mask
        
        # Re-normalize so the sum is 1.0 again
        if filtered_probs[i].sum() > 0:
            filtered_probs[i] /= filtered_probs[i].sum()

    # 7. Calculate Accuracy
    top1 = np.mean(np.argmax(filtered_probs, axis=1) == y_true) * 100
    
    top3_idx = np.argsort(filtered_probs, axis=1)[:, -3:]
    top3 = np.any(top3_idx == y_true[:, None], axis=1).mean() * 100
    
    return top1, top3

# --- RUN ---
df = pd.read_parquet("final_data.parquet")
test_sample = df.sample(n=50000, random_state=42).copy()

t1, t3 = get_filtered_accuracy('model_type_optimized.txt', test_sample, 'pitch_type')
print(f"\n✅ Filtered Pitch Type -> Top 1: {t1:.2f}%, Top 3: {t3:.2f}%")