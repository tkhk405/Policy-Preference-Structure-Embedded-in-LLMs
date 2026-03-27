"""
Mantel Test: Comparison of LLM Internal Representations with Parliamentary Survey

Compare matrices via Mantel test (exact enumeration, 6! = 720 permutations):
  1. Probing transfer matrix vs. Cosine similarity matrix
  2. Probing transfer matrix vs. Taniguchi-Asahi survey correlation matrix
  3. Cosine similarity matrix vs. Taniguchi-Asahi survey correlation matrix
  4. Baseline (per-model and combined) vs. Taniguchi-Asahi survey

Usage:
    python probing.py
    python transfer_analysis.py
    python cosine_similarity.py
    python baseline_analysis.py
    python mantel_test.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from itertools import permutations

np.random.seed(42)

# ==============================================================================
# Path settings
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

TRANSFER_PATH = os.path.join(REPO_ROOT, "output", "transfer_results", "transfer_matrix.csv")
COSINE_PATH = os.path.join(REPO_ROOT, "output", "cosine_results", "cosine_similarity_matrix.csv")
BASELINE_DIR = os.path.join(REPO_ROOT, "output", "baseline_results")
RESULT_DIR = os.path.join(REPO_ROOT, "output", "mantel_results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Taniguchi-Asahi survey data is not included in this repository.
# See: https://www.masaki.j.u-tokyo.ac.jp/utas/utasindex_en.html
TANIGUCHI_PATH = "taniguchi_asahi_survey.csv"

# ==============================================================================
# Theme settings
# ==============================================================================
THEMES = ["Defense", "Social", "Public", "Fiscal", "Nkorea", "Security"]
COLS_JP = ["防衛力強化", "小さな政府", "公共事業", "財政出動", "北朝鮮", "治安"]
COL_MAP = dict(zip(THEMES, COLS_JP))
n_themes = len(THEMES)

# Mapping from matrix labels to internal theme names
MATRIX_LABEL_MAP = {
    "Defense": "Defense",
    "Social Welfare": "Social",
    "Public Works": "Public",
    "Fiscal Stimulus": "Fiscal",
    "North Korea": "Nkorea",
    "Security": "Security",
}

# ==============================================================================
# Mantel test function (exact enumeration)
# ==============================================================================
def mantel_test(mat1, mat2, n):
    """Mantel test (Spearman) with exact enumeration of all n! permutations.

    Returns (rho, p_value, n_permutations).
    If mat1 has zero variance in upper triangle, returns (nan, nan, n_perm).
    """
    def upper_tri(mat):
        return np.array([mat[i, j] for i in range(n) for j in range(i + 1, n)])

    ut1 = upper_tri(mat1)
    ut2 = upper_tri(mat2)

    if np.std(ut1) == 0:
        return np.nan, np.nan, 0

    obs, _ = stats.spearmanr(ut1, ut2)

    count = 0
    total = 0
    for perm in permutations(range(n)):
        perm = list(perm)
        mat1_perm = mat1[np.ix_(perm, perm)]
        ut1_perm = upper_tri(mat1_perm)
        if np.std(ut1_perm) == 0:
            total += 1
            continue
        r, _ = stats.spearmanr(ut1_perm, ut2)
        if r >= obs:
            count += 1
        total += 1

    return obs, count / total, total

# ==============================================================================
# Helper: load and reorder matrix by THEMES
# ==============================================================================
def load_and_reorder(csv_path):
    """Load a CSV matrix and reorder rows/columns to match THEMES."""
    df = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
    theme_map = {}
    for label in list(df.index):
        if label in MATRIX_LABEL_MAP:
            theme_map[MATRIX_LABEL_MAP[label]] = label
        else:
            theme_map[label] = label

    matrix = np.array([
        [df.loc[theme_map.get(THEMES[i], THEMES[i]),
                theme_map.get(THEMES[j], THEMES[j])]
         for j in range(n_themes)]
        for i in range(n_themes)
    ], dtype=float)
    return matrix

# ==============================================================================
# Load probing transfer matrix (symmetrize)
# ==============================================================================
probing_matrix = load_and_reorder(TRANSFER_PATH)
probing_sym = (probing_matrix + probing_matrix.T) / 2

# ==============================================================================
# Load cosine similarity matrix
# ==============================================================================
cos_matrix = load_and_reorder(COSINE_PATH)

# ==============================================================================
# Load baseline correlation matrices
# ==============================================================================
baseline_matrices = {}
baseline_files = glob.glob(os.path.join(BASELINE_DIR, "baseline_corr_*.csv"))
for fpath in sorted(baseline_files):
    fname = os.path.basename(fpath)
    # Extract label from filename (e.g., baseline_corr_gpt-5.1.csv -> GPT-5.1)
    label = fname.replace("baseline_corr_", "").replace(".csv", "").replace("_", " ").title()
    df = pd.read_csv(fpath, index_col=0, encoding="utf-8-sig")
    baseline_matrices[label] = df.values.astype(float)
    print(f"Loaded baseline: {label} ({fpath})")

# ==============================================================================
# Load Taniguchi-Asahi survey data and compute Spearman correlation matrix
# ==============================================================================
if not os.path.exists(TANIGUCHI_PATH):
    print(f"\nTaniguchi-Asahi survey data not found: {TANIGUCHI_PATH}")
    print("Please set TANIGUCHI_PATH to your local copy.")
    print("Proceeding without Taniguchi-Asahi comparisons.\n")
    tani_matrix = None
else:
    df = pd.read_csv(TANIGUCHI_PATH, encoding="utf-8-sig")
    for col in COLS_JP:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~df[col].isin({1.0, 2.0, 3.0, 4.0, 5.0}), col] = np.nan

    # Listwise deletion (2004: exclude public safety due to missing question)
    cols_without_safety = [c for c in COLS_JP if c != "治安"]
    mask_2004 = df["調査年"] == 2004
    df_2004 = df[mask_2004].dropna(subset=cols_without_safety)
    df_other = df[~mask_2004].dropna(subset=COLS_JP)
    df_valid = pd.concat([df_2004, df_other], ignore_index=True)

    print(f"Taniguchi-Asahi survey: N = {len(df_valid)}")

    corr_spearman = df_valid[COLS_JP].corr(method="spearman")
    tani_matrix = np.array([
        [corr_spearman.loc[COL_MAP[THEMES[i]], COL_MAP[THEMES[j]]]
         for j in range(n_themes)]
        for i in range(n_themes)
    ], dtype=float)

# ==============================================================================
# Run Mantel tests
# ==============================================================================
print("\n" + "=" * 70)
print(f"Mantel test (exact enumeration, {n_themes}! = 720 permutations)")
print("=" * 70)

comparisons = [
    ("Probing transfer vs. Cosine similarity", probing_sym, cos_matrix),
]

if tani_matrix is not None:
    comparisons.extend([
        ("Probing transfer vs. Taniguchi-Asahi survey", probing_sym, tani_matrix),
        ("Cosine similarity vs. Taniguchi-Asahi survey", cos_matrix, tani_matrix),
    ])
    # Baseline vs. Taniguchi-Asahi
    for label, bl_matrix in baseline_matrices.items():
        comparisons.append((f"Baseline ({label}) vs. Taniguchi-Asahi survey", bl_matrix, tani_matrix))

results = []
for label, mat1, mat2 in comparisons:
    rho, p_rho, n_perm = mantel_test(mat1, mat2, n_themes)
    results.append({"Comparison": label, "Spearman_rho": rho, "p_value": p_rho})
    if np.isnan(rho):
        print(f"\n{label}:")
        print(f"  N/A (zero variance in correlation matrix)")
    else:
        print(f"\n{label}:")
        print(f"  Spearman rho = {rho:.4f},  p = {p_rho:.4f}")

# ==============================================================================
# Save results
# ==============================================================================
df_results = pd.DataFrame(results)
csv_path = os.path.join(RESULT_DIR, "mantel_test_results.csv")
df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\nSaved: {csv_path}")
