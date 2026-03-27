"""
Cross-Domain Transfer Performance Analysis

Evaluate how well probing coefficients learned on one policy issue
transfer to predicting stances on another issue.

- Top 20 heads: Selected by average Spearman correlation across 6 issues
- Transfer metric: Spearman correlation between linear score (X @ W)
  and ground-truth labels (thresholds are not used)

Usage:
    python probing.py             # Run probing analysis first
    python transfer_analysis.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

np.random.seed(42)

# ==============================================================================
# Path settings
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(REPO_ROOT, "data")
VEC_DIR = os.path.join(REPO_ROOT, "output", "activation_vectors")
PROBING_DIR = os.path.join(REPO_ROOT, "output", "probing_results", "full")
RESULT_DIR = os.path.join(REPO_ROOT, "output", "transfer_results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ==============================================================================
# Dataset definitions (6 policy issues)
# ==============================================================================
# (CSV filename, vector prefix, probing result prefix)
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
TOP_N = 20

# ==============================================================================
# Select top 20 heads (by average score across 6 issues)
# ==============================================================================
rho_data = {}
for theme, (_, _, save_prefix) in DATASETS.items():
    rho_path = os.path.join(PROBING_DIR, f"{save_prefix}_heatmap_rho_full.npy")
    rho_data[theme] = np.load(rho_path)

avg_rho = np.mean(list(rho_data.values()), axis=0)  # (42, 16)
flat_indices = np.argsort(avg_rho.flatten())[::-1][:TOP_N]
top_heads = [(idx // NUM_HEADS, idx % NUM_HEADS) for idx in flat_indices]
used_layers = sorted(set(l for l, h in top_heads))

print(f"Top {TOP_N} heads (layers used: {used_layers})")

# ==============================================================================
# Load data
# ==============================================================================
labels_map = {}
masks_map = {}
coef_data = {}

for theme, (csv_file, _, save_prefix) in DATASETS.items():
    df = pd.read_csv(os.path.join(INPUT_DIR, csv_file))
    valid_mask = df['Stance_Value'].isin([1, 2, 3, 4, 5]).values
    masks_map[theme] = valid_mask
    labels_map[theme] = df.loc[valid_mask, 'Stance_Value'].astype(float).values

    coef_path = os.path.join(PROBING_DIR, f"{save_prefix}_coef_full.npy")
    coef_data[theme] = np.load(coef_path)

# ==============================================================================
# Compute cross-domain transfer performance
# ==============================================================================
themes_list = list(DATASETS.keys())
cross_scores = {(src, tgt): [] for src in themes_list for tgt in themes_list}

for layer_idx in tqdm(used_layers, desc="Layers"):
    heads_in_layer = [h for l, h in top_heads if l == layer_idx]

    # Load activation vectors
    layer_vecs = {}
    for theme, (_, vec_prefix, _) in DATASETS.items():
        vec_path = os.path.join(VEC_DIR, f"{vec_prefix}_layer_{layer_idx:02d}.npy")
        full_vec = np.load(vec_path)
        layer_vecs[theme] = full_vec[masks_map[theme]]

    # Apply source coefficients to target vectors
    for src in themes_list:
        W = coef_data[src][layer_idx]

        for tgt in themes_list:
            X = layer_vecs[tgt]
            y_true = labels_map[tgt]

            head_scores = []
            for h in heads_in_layer:
                scaler = StandardScaler()
                X_h_scaled = scaler.fit_transform(X[:, h, :])
                y_pred = X_h_scaled @ W[h]

                if np.std(y_pred) > 0:
                    rho, _ = spearmanr(y_true, y_pred)
                    if not np.isnan(rho):
                        head_scores.append(rho)

            if head_scores:
                cross_scores[(src, tgt)].append(np.mean(head_scores))

# ==============================================================================
# Aggregate and save results
# ==============================================================================
transfer_matrix = pd.DataFrame(index=themes_list, columns=themes_list, dtype=float)

for (src, tgt), scores in cross_scores.items():
    if scores:
        transfer_matrix.loc[src, tgt] = np.mean(scores)

print("\nTransfer performance matrix:")
print(transfer_matrix.to_string())

csv_path = os.path.join(RESULT_DIR, "transfer_matrix.csv")
transfer_matrix.to_csv(csv_path, encoding='utf-8-sig')
print(f"\nSaved: {csv_path}")
