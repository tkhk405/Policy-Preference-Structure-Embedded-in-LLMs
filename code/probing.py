"""
Probing Analysis with Ordinal Logistic Regression

Apply ordinal logistic regression (LogisticAT) to activation vectors
to evaluate how well each layer and head encodes policy stances.

- Method: mord.LogisticAT (All-Threshold, absolute error loss)
- Evaluation: Spearman correlation via 5-fold stratified cross-validation
- Regularization: alpha selected from {0.01, 0.1, 1, 10, 100, 1000}

Usage:
    python extract_activations.py  # Extract activation vectors first
    python probing.py
"""

import os

# Limit numpy/MKL threads to 1 (prevent conflicts with joblib parallelism)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import mord
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)

# ==============================================================================
# Path settings
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(REPO_ROOT, "data")
VEC_DIR = os.path.join(REPO_ROOT, "output", "activation_vectors")
RESULT_DIR = os.path.join(REPO_ROOT, "output", "probing_results")

for subdir in ["rho", "coef", "alpha", "theta", "preds", "full"]:
    os.makedirs(os.path.join(RESULT_DIR, subdir), exist_ok=True)

# ==============================================================================
# Dataset definitions (6 policy issues)
# ==============================================================================
# (CSV filename, vector prefix, save prefix)
DATASETS = {
    "Defense":         ("defense.csv",         "Defense",  "Defense"),
    "Social Welfare":  ("social_welfare.csv",  "Social",   "Social"),
    "Public Works":    ("public_works.csv",    "Public",   "Public"),
    "Fiscal Stimulus": ("fiscal_stimulus.csv", "Fiscal",   "Fiscal"),
    "North Korea":     ("north_korea.csv",     "Nkorea",   "Nkorea"),
    "Security":        ("public_safety.csv",   "Security", "Security"),
}

NUM_LAYERS = 42
NUM_HEADS = 16
ALPHAS = np.logspace(-2, 3, 6)  # [0.01, 0.1, 1, 10, 100, 1000]
N_JOBS = 8

# ==============================================================================
# Spearman correlation
# ==============================================================================
def calc_spearman(y_true, y_pred):
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    c, _ = spearmanr(y_true, y_pred)
    return c if not np.isnan(c) else np.nan

# ==============================================================================
# Process one head (for parallel execution)
# ==============================================================================
def process_one_head(h_idx, layer_data, labels_ordinal, alphas):
    """Run cross-validation and final training for a single head."""
    X = layer_data[:, h_idx, :]
    y = labels_ordinal

    oof_preds_temp = {a: np.zeros_like(y, dtype=float) for a in alphas}
    alpha_cv_scores = {a: [] for a in alphas}

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for alpha in alphas:
            model = mord.LogisticAT(alpha=alpha)
            try:
                model.fit(X_train_scaled, y_train)
                y_pred_val = model.predict(X_val_scaled)
            except Exception:
                continue

            oof_preds_temp[alpha][val_idx] = y_pred_val
            score = calc_spearman(y_val + 1, y_pred_val + 1)
            if not np.isnan(score):
                alpha_cv_scores[alpha].append(score)

    # Select best alpha
    avg_scores = {a: np.mean(s) if s else np.nan for a, s in alpha_cv_scores.items()}
    valid_scores = {k: v for k, v in avg_scores.items() if not np.isnan(v)}

    if valid_scores:
        best_alpha = max(valid_scores, key=valid_scores.get)
        best_score = valid_scores[best_alpha]
    else:
        best_alpha = alphas[0]
        best_score = 0.0

    # Retrain on all data
    scaler_final = StandardScaler()
    X_final = scaler_final.fit_transform(X)
    final_model = mord.LogisticAT(alpha=best_alpha)

    coef = np.zeros(X.shape[1])
    theta = np.zeros(4)  # 5 categories -> 4 thresholds

    try:
        final_model.fit(X_final, y)
        coef = final_model.coef_.flatten()
        if hasattr(final_model, 'theta_') and final_model.theta_ is not None:
            theta = final_model.theta_
    except Exception:
        pass

    oof_preds = oof_preds_temp[best_alpha] + 1  # Convert back to original scale (1-5)

    return h_idx, best_score, best_alpha, coef, theta, oof_preds

# ==============================================================================
# Main loop
# ==============================================================================
dataset_list = list(DATASETS.items())

for ds_idx, (theme_name, (csv_file, vec_prefix, save_prefix)) in enumerate(dataset_list):
    print(f"\n{'=' * 60}")
    print(f"[{ds_idx+1}/{len(dataset_list)}] {theme_name}")
    print(f"{'=' * 60}")

    # Load data
    csv_path = os.path.join(INPUT_DIR, csv_file)
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    valid_mask = df['Stance_Value'].isin([1, 2, 3, 4, 5]).values
    df_filtered = df[valid_mask].copy()

    labels_original = df_filtered['Stance_Value'].astype(int).values
    labels_ordinal = labels_original - 1  # Convert to 0-4

    print(f"Samples: {len(df_filtered)}")

    heatmap_rho = np.zeros((NUM_LAYERS, NUM_HEADS))
    best_alpha_data = np.zeros((NUM_LAYERS, NUM_HEADS))
    heatmap_coef = None
    heatmap_theta = None

    # Layer-wise loop
    for layer_idx in tqdm(range(NUM_LAYERS), desc=f"[{save_prefix}]"):
        vec_path = os.path.join(VEC_DIR, f"{vec_prefix}_layer_{layer_idx:02d}.npy")

        if not os.path.exists(vec_path):
            print(f"Vector not found: {vec_path}")
            continue

        full_layer_data = np.load(vec_path)
        if full_layer_data.shape[0] != len(df):
            print(f"Size mismatch: CSV={len(df)}, NPY={full_layer_data.shape[0]}")
            continue

        layer_data = full_layer_data[valid_mask]

        if heatmap_coef is None:
            heatmap_coef = np.zeros((NUM_LAYERS, NUM_HEADS, layer_data.shape[2]))
            heatmap_theta = np.zeros((NUM_LAYERS, NUM_HEADS, 4))

        layer_oof_preds = np.zeros((len(labels_original), NUM_HEADS))

        # Parallel processing across heads
        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_one_head)(h_idx, layer_data, labels_ordinal, ALPHAS)
            for h_idx in range(NUM_HEADS)
        )

        for h_idx, best_score, best_alpha, coef, theta, oof_preds in results:
            heatmap_rho[layer_idx, h_idx] = best_score
            best_alpha_data[layer_idx, h_idx] = best_alpha
            heatmap_coef[layer_idx, h_idx, :] = coef
            heatmap_theta[layer_idx, h_idx, :] = theta
            layer_oof_preds[:, h_idx] = oof_preds

        # Save per-layer results
        np.save(os.path.join(RESULT_DIR, "rho",   f"{save_prefix}_layer_{layer_idx:02d}_rho.npy"),   heatmap_rho[layer_idx])
        np.save(os.path.join(RESULT_DIR, "coef",  f"{save_prefix}_layer_{layer_idx:02d}_coef.npy"),  heatmap_coef[layer_idx])
        np.save(os.path.join(RESULT_DIR, "theta", f"{save_prefix}_layer_{layer_idx:02d}_theta.npy"), heatmap_theta[layer_idx])
        np.save(os.path.join(RESULT_DIR, "alpha", f"{save_prefix}_layer_{layer_idx:02d}_alpha.npy"), best_alpha_data[layer_idx])
        np.save(os.path.join(RESULT_DIR, "preds", f"{save_prefix}_layer_{layer_idx:02d}_preds.npy"), layer_oof_preds)

    # Save aggregated results
    if heatmap_coef is not None:
        np.save(os.path.join(RESULT_DIR, "full", f"{save_prefix}_heatmap_rho_full.npy"),  heatmap_rho)
        np.save(os.path.join(RESULT_DIR, "full", f"{save_prefix}_coef_full.npy"),         heatmap_coef)
        np.save(os.path.join(RESULT_DIR, "full", f"{save_prefix}_theta_full.npy"),        heatmap_theta)
        np.save(os.path.join(RESULT_DIR, "full", f"{save_prefix}_bestAlpha_full.npy"),    best_alpha_data)
        np.save(os.path.join(RESULT_DIR, "full", f"{save_prefix}_labels.npy"),            labels_original)

    print(f"{theme_name} done")

print("\nAll issues completed")
