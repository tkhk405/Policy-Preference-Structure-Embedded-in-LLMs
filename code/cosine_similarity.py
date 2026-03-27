"""
Cosine Similarity Analysis of Direction Vectors

Compute cosine similarity between sigma-standardized direction vectors
across policy issues. Direction vectors are defined as the mean difference
between oppose and agree groups in activation space.

- Direction vector: v_i = mean(oppose) - mean(agree)
- Sigma-standardization: each dimension divided by its standard deviation
- Head selection: union of per-issue top-20 heads

Usage:
    python probing.py              # Run probing analysis first
    python cosine_similarity.py
"""

import os
import numpy as np
import pandas as pd
np.random.seed(42)

# ==============================================================================
# Path settings
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

VEC_DIR = os.path.join(REPO_ROOT, "output", "activation_vectors")
PROBING_DIR = os.path.join(REPO_ROOT, "output", "probing_results", "full")
RESULT_DIR = os.path.join(REPO_ROOT, "output", "cosine_results")
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

THEME_ORDER = list(DATASETS.keys())
NUM_LAYERS = 42
NUM_HEADS = 16
TOP_N = 20

# ==============================================================================
# Select heads: union of per-issue top-20
# ==============================================================================
rho_data = {}
for theme, (_, _, save_prefix) in DATASETS.items():
    rho_path = os.path.join(PROBING_DIR, f"{save_prefix}_heatmap_rho_full.npy")
    rho_data[theme] = np.load(rho_path)

top20_by_theme = {}
for theme in THEME_ORDER:
    rho = rho_data[theme]
    flat_indices = np.argsort(rho.flatten())[::-1][:TOP_N]
    top20_by_theme[theme] = [(idx // NUM_HEADS, idx % NUM_HEADS) for idx in flat_indices]

# Union of all per-issue top-20 heads
all_heads = sorted(set(h for heads in top20_by_theme.values() for h in heads))
all_layers = sorted(set(l for l, h in all_heads))

print(f"Union of per-issue top-{TOP_N} heads: {len(all_heads)} heads")
print(f"Layers used: {all_layers}")

# ==============================================================================
# Load labels
# ==============================================================================
labels_map = {}

for theme, (_, _, save_prefix) in DATASETS.items():
    labels = np.load(os.path.join(PROBING_DIR, f"{save_prefix}_labels.npy"))
    labels_map[theme] = labels

# ==============================================================================
# Compute direction vectors (d_S) and sigma
# ==============================================================================
D_STANDARDIZED = {}  # {theme: {(layer, head): standardized_vector}}

for theme, (csv_file, vec_prefix, save_prefix) in DATASETS.items():
    labels = labels_map[theme]
    idx_agree = np.where((labels == 1) | (labels == 2))[0]
    idx_oppose = np.where((labels == 4) | (labels == 5))[0]

    print(f"{theme}: agree={len(idx_agree)}, oppose={len(idx_oppose)}")

    d_std = {}

    for layer in all_layers:
        vec_path = os.path.join(VEC_DIR, f"{vec_prefix}_layer_{layer:02d}.npy")
        vecs = np.load(vec_path)

        for head in range(NUM_HEADS):
            if (layer, head) not in all_heads:
                continue

            v_agree = vecs[idx_agree, head, :]
            v_oppose = vecs[idx_oppose, head, :]

            # Direction vector: oppose - agree
            d = v_oppose.mean(axis=0) - v_agree.mean(axis=0)

            # Sigma: std across agree + oppose samples
            v_all = np.concatenate([v_agree, v_oppose], axis=0)
            sigma = v_all.std(axis=0)
            sigma_safe = np.where(sigma > 1e-10, sigma, 1.0)

            # Sigma-standardized direction vector
            d_std[(layer, head)] = d / sigma_safe

    D_STANDARDIZED[theme] = d_std

# ==============================================================================
# Compute cosine similarity matrix
# ==============================================================================
def cosine_sim(v1, v2):
    """Cosine similarity between two vectors."""
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return np.nan
    return np.dot(v1, v2) / (norm1 * norm2)

n_themes = len(THEME_ORDER)
cos_matrix = np.zeros((n_themes, n_themes))

for i, t1 in enumerate(THEME_ORDER):
    cos_matrix[i, i] = 1.0
    for j, t2 in enumerate(THEME_ORDER):
        if i >= j:
            continue

        sims = [cosine_sim(D_STANDARDIZED[t1][lh], D_STANDARDIZED[t2][lh])
                for lh in all_heads]
        sims = [s for s in sims if not np.isnan(s)]
        cos_matrix[i, j] = np.mean(sims)
        cos_matrix[j, i] = cos_matrix[i, j]

# ==============================================================================
# Save results
# ==============================================================================
df_matrix = pd.DataFrame(cos_matrix, index=THEME_ORDER, columns=THEME_ORDER)

print("\nCosine similarity matrix (sigma-standardized):")
print(df_matrix.round(4).to_string())

csv_path = os.path.join(RESULT_DIR, "cosine_similarity_matrix.csv")
df_matrix.to_csv(csv_path, encoding='utf-8-sig')
print(f"\nSaved: {csv_path}")
