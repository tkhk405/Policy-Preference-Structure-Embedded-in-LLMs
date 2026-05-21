"""
Mantel Test: Comparison of LLM Internal Representations with Parliamentary Survey

Compare matrices via Mantel test (exact enumeration, 6! = 720 permutations):
  1. Probing transfer matrix vs. cosine similarity matrix
  2. Probing transfer matrix vs. Taniguchi-Asahi survey correlation matrix
     for elected members
  3. Cosine similarity matrix vs. Taniguchi-Asahi survey correlation matrix
     for elected members
  4. Probing transfer matrix vs. Taniguchi-Asahi survey correlation matrix
     for all candidates
  5. Cosine similarity matrix vs. Taniguchi-Asahi survey correlation matrix
     for all candidates
  6. Baseline final-output matrices vs. elected-member Taniguchi-Asahi matrix,
     when the baseline matrix is complete.

Usage:
    python probing.py
    python transfer_analysis.py
    python cosine_similarity.py
    python baseline_analysis.py
    python mantel_test.py

The Taniguchi-Asahi Survey data is not included in this repository. Set
TANIGUCHI_ELECTED_PATH and TANIGUCHI_ALL_CANDIDATES_PATH to local CSV files.
For backward compatibility, TANIGUCHI_PATH is treated as the elected-member
CSV path when TANIGUCHI_ELECTED_PATH is not set.
"""

import glob
import math
import os
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# Path settings
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

TRANSFER_PATH = os.path.join(REPO_ROOT, "output", "transfer_results", "transfer_matrix.csv")
COSINE_PATH = os.path.join(REPO_ROOT, "output", "cosine_results", "cosine_similarity_matrix.csv")
BASELINE_DIR = os.path.join(REPO_ROOT, "output", "baseline_results")
RESULT_DIR = os.path.join(REPO_ROOT, "output", "mantel_results")

# Taniguchi-Asahi survey data is not included in this repository.
# See: https://www.masaki.j.u-tokyo.ac.jp/utas/utasindex_en.html
TANIGUCHI_PATH = os.environ.get("TANIGUCHI_PATH", "taniguchi_asahi_survey.csv")
TANIGUCHI_ELECTED_PATH = os.environ.get("TANIGUCHI_ELECTED_PATH", TANIGUCHI_PATH)
TANIGUCHI_ALL_CANDIDATES_PATH = os.environ.get(
    "TANIGUCHI_ALL_CANDIDATES_PATH",
    "taniguchi_asahi_survey_all_candidates.csv",
)

# ==============================================================================
# Theme settings
# ==============================================================================
THEMES = ["Defense", "Social", "Public", "Fiscal", "Nkorea", "Security"]
COLS_JP = ["防衛力強化", "小さな政府", "公共事業", "財政出動", "北朝鮮", "治安"]
COL_MAP = dict(zip(THEMES, COLS_JP))
N_THEMES = len(THEMES)

# Mapping from matrix labels to internal theme names
MATRIX_LABEL_MAP = {
    "Defense": "Defense",
    "Social Welfare": "Social",
    "Public Works": "Public",
    "Fiscal Stimulus": "Fiscal",
    "North Korea": "Nkorea",
    "Security": "Security",
}


def upper_tri(mat):
    """Return off-diagonal upper-triangle elements."""
    n = mat.shape[0]
    return np.array([mat[i, j] for i in range(n) for j in range(i + 1, n)])


def matrix_valid_for_mantel(mat):
    """Check whether a matrix can be used in the Mantel test."""
    vals = upper_tri(mat)
    if not np.all(np.isfinite(vals)):
        return False, "upper triangle contains NaN or infinite values"
    if np.std(vals) == 0:
        return False, "upper triangle has zero variance"
    return True, ""


def mantel_test(mat1, mat2):
    """Mantel test (Spearman) with exact enumeration of all 6! permutations.

    Returns (rho, p_value, n_permutations, reason). The p-value is one-sided:
    the proportion of permutations with rho >= observed rho.
    """
    n = mat1.shape[0]
    n_permutations = math.factorial(n)

    ok1, reason1 = matrix_valid_for_mantel(mat1)
    ok2, reason2 = matrix_valid_for_mantel(mat2)
    if not ok1:
        return np.nan, np.nan, n_permutations, f"matrix 1: {reason1}"
    if not ok2:
        return np.nan, np.nan, n_permutations, f"matrix 2: {reason2}"

    ut1 = upper_tri(mat1)
    ut2 = upper_tri(mat2)
    obs, _ = stats.spearmanr(ut1, ut2)

    count = 0
    total = 0
    for perm in permutations(range(n)):
        perm = list(perm)
        mat1_perm = mat1[np.ix_(perm, perm)]
        r, _ = stats.spearmanr(upper_tri(mat1_perm), ut2)
        if r >= obs:
            count += 1
        total += 1

    return obs, count / total, total, ""


def resolve_input_path(path):
    """Resolve relative input paths from cwd or repository root."""
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(REPO_ROOT, path)


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
         for j in range(N_THEMES)]
        for i in range(N_THEMES)
    ], dtype=float)
    return matrix


def load_taniguchi_matrix(csv_path, label):
    """Load UTAS data and compute the six-issue Spearman correlation matrix."""
    resolved_path = resolve_input_path(csv_path)
    if not os.path.exists(resolved_path):
        print(f"\nTaniguchi-Asahi survey data not found for {label}: {csv_path}")
        print(f"Resolved path checked: {resolved_path}")
        return None

    df = pd.read_csv(resolved_path, encoding="utf-8-sig")
    for col in COLS_JP:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~df[col].isin({1.0, 2.0, 3.0, 4.0, 5.0}), col] = np.nan

    # Listwise deletion. The 2004 survey has no public-safety item, so that
    # year is retained when the other five items are complete. Pandas corr()
    # then uses pairwise complete observations for pairs involving public safety.
    cols_without_safety = [c for c in COLS_JP if c != "治安"]
    mask_2004 = df["調査年"] == 2004
    df_2004 = df[mask_2004].dropna(subset=cols_without_safety)
    df_other = df[~mask_2004].dropna(subset=COLS_JP)
    df_valid = pd.concat([df_2004, df_other], ignore_index=True)

    print(f"Taniguchi-Asahi survey ({label}): N = {len(df_valid)}")

    corr_spearman = df_valid[COLS_JP].corr(method="spearman")
    return np.array([
        [corr_spearman.loc[COL_MAP[THEMES[i]], COL_MAP[THEMES[j]]]
         for j in range(N_THEMES)]
        for i in range(N_THEMES)
    ], dtype=float)


def load_baseline_matrices():
    """Load complete baseline matrices and record incomplete ones as skipped."""
    baseline_matrices = {}
    skipped = []
    baseline_files = glob.glob(os.path.join(BASELINE_DIR, "baseline_corr_*.csv"))

    for fpath in sorted(baseline_files):
        fname = os.path.basename(fpath)
        label = fname.replace("baseline_corr_", "").replace(".csv", "").replace("_", " ").title()
        matrix = load_and_reorder(fpath)
        ok, reason = matrix_valid_for_mantel(matrix)
        if ok:
            baseline_matrices[label] = matrix
            print(f"Loaded baseline: {label} ({fpath})")
        else:
            skipped.append((label, reason))
            print(f"Skipped baseline: {label} ({reason})")

    return baseline_matrices, skipped


def run_comparison(label, mat1, mat2):
    """Run a Mantel comparison and return a result row."""
    rho, p_rho, n_perm, reason = mantel_test(mat1, mat2)
    if np.isnan(rho):
        print(f"\n{label}:")
        print(f"  skipped ({reason})")
        status = "skipped"
    else:
        print(f"\n{label}:")
        print(f"  Spearman rho = {rho:.4f},  p = {p_rho:.4f}")
        status = "ok"

    return {
        "Comparison": label,
        "Spearman_rho": rho,
        "p_value": p_rho,
        "n_permutations": n_perm,
        "status": status,
        "reason": reason,
    }


def main():
    np.random.seed(42)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Load probing transfer matrix (symmetrize)
    probing_matrix = load_and_reorder(TRANSFER_PATH)
    probing_sym = (probing_matrix + probing_matrix.T) / 2

    # Load cosine similarity matrix
    cos_matrix = load_and_reorder(COSINE_PATH)

    # Load baseline correlation matrices. Incomplete matrices are not tested.
    baseline_matrices, skipped_baselines = load_baseline_matrices()

    # Load Taniguchi-Asahi survey data
    tani_elected = load_taniguchi_matrix(TANIGUCHI_ELECTED_PATH, "elected members")
    tani_all = load_taniguchi_matrix(TANIGUCHI_ALL_CANDIDATES_PATH, "all candidates")

    print("\n" + "=" * 70)
    print(f"Mantel test (exact enumeration, {N_THEMES}! = 720 permutations)")
    print("=" * 70)

    results = [
        run_comparison("Probing transfer vs. Cosine similarity", probing_sym, cos_matrix)
    ]

    if tani_elected is not None:
        results.extend([
            run_comparison(
                "Probing transfer vs. Taniguchi-Asahi survey (elected members)",
                probing_sym,
                tani_elected,
            ),
            run_comparison(
                "Cosine similarity vs. Taniguchi-Asahi survey (elected members)",
                cos_matrix,
                tani_elected,
            ),
        ])

        for label, bl_matrix in baseline_matrices.items():
            results.append(run_comparison(
                f"Baseline ({label}) vs. Taniguchi-Asahi survey (elected members)",
                bl_matrix,
                tani_elected,
            ))

        for label, reason in skipped_baselines:
            comparison = f"Baseline ({label}) vs. Taniguchi-Asahi survey (elected members)"
            results.append({
                "Comparison": comparison,
                "Spearman_rho": np.nan,
                "p_value": np.nan,
                "n_permutations": math.factorial(N_THEMES),
                "status": "skipped",
                "reason": reason,
            })
            print(f"\n{comparison}:")
            print(f"  skipped ({reason})")
    else:
        print("\nElected-member UTAS data unavailable; skipped UTAS and baseline comparisons.")

    if tani_all is not None:
        results.extend([
            run_comparison(
                "Probing transfer vs. Taniguchi-Asahi survey (all candidates)",
                probing_sym,
                tani_all,
            ),
            run_comparison(
                "Cosine similarity vs. Taniguchi-Asahi survey (all candidates)",
                cos_matrix,
                tani_all,
            ),
        ])
    else:
        print("\nAll-candidate UTAS data unavailable; skipped all-candidate comparisons.")

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(RESULT_DIR, "mantel_test_results.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
