"""
プロービング分析コード（順序ロジスティック回帰）

活性化ベクトルに対して順序ロジスティック回帰（LogisticAT）を適用し、
各層・各ヘッドの政策スタンス予測性能を評価する。

- 手法: mord.LogisticAT（All-Threshold, 絶対誤差損失）
- 評価: 5分割層化交差検証によるスピアマン相関係数
- 正則化: α ∈ {0.01, 0.1, 1, 10, 100, 1000} から交差検証で選択

使用方法:
    git clone https://github.com/tkhk405/Policy-Preference-Structure-Embedded-in-LLMs.git
    cd Policy-Preference-Structure-Embedded-in-LLMs/code
    python extract_activations.py  # 先に活性化ベクトルを抽出
    python probing.py
"""

import os

# numpy/MKL等のスレッド数を1に制限（joblibとの競合を防止）
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
# パス設定
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(REPO_ROOT, "data")
VEC_DIR = os.path.join(REPO_ROOT, "output", "activation_vectors")
RESULT_DIR = os.path.join(REPO_ROOT, "output", "probing_results")

for subdir in ["rho", "coef", "alpha", "theta", "preds", "full"]:
    os.makedirs(os.path.join(RESULT_DIR, subdir), exist_ok=True)

# ==============================================================================
# データセット定義（6テーマ）
# ==============================================================================
# (CSVファイル名, ベクトルプレフィックス, 保存プレフィックス)
DATASETS = {
    "Defense (防衛)":            ("defense.csv",         "Defense",  "Defense"),
    "Social Welfare (社会福祉)": ("social_welfare.csv",  "Social",   "Social"),
    "Public Works (公共事業)":   ("public_works.csv",    "Public",   "Public"),
    "Fiscal Stimulus (財政出動)": ("fiscal_stimulus.csv", "Fiscal",   "Fiscal"),
    "North Korea (北朝鮮)":      ("north_korea.csv",     "Nkorea",   "Nkorea"),
    "Public Safety (治安)":      ("public_safety.csv",   "Security", "Security"),
}

NUM_LAYERS = 42
NUM_HEADS = 16
ALPHAS = np.logspace(-2, 3, 6)  # [0.01, 0.1, 1, 10, 100, 1000]
N_JOBS = 8

# ==============================================================================
# スピアマン相関の計算
# ==============================================================================
def calc_spearman(y_true, y_pred):
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    c, _ = spearmanr(y_true, y_pred)
    return c if not np.isnan(c) else np.nan

# ==============================================================================
# 1ヘッド分の処理（並列化用）
# ==============================================================================
def process_one_head(h_idx, layer_data, labels_ordinal, alphas):
    """1つのヘッドに対する交差検証 + 全データ再学習を実行"""
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

    # 最適αの選択
    avg_scores = {a: np.mean(s) if s else np.nan for a, s in alpha_cv_scores.items()}
    valid_scores = {k: v for k, v in avg_scores.items() if not np.isnan(v)}

    if valid_scores:
        best_alpha = max(valid_scores, key=valid_scores.get)
        best_score = valid_scores[best_alpha]
    else:
        best_alpha = alphas[0]
        best_score = 0.0

    # 全データで再学習
    scaler_final = StandardScaler()
    X_final = scaler_final.fit_transform(X)
    final_model = mord.LogisticAT(alpha=best_alpha)

    coef = np.zeros(X.shape[1])
    theta = np.zeros(4)  # 5カテゴリ → 4しきい値

    try:
        final_model.fit(X_final, y)
        coef = final_model.coef_.flatten()
        if hasattr(final_model, 'theta_') and final_model.theta_ is not None:
            theta = final_model.theta_
    except Exception:
        pass

    oof_preds = oof_preds_temp[best_alpha] + 1  # 元スケール（1-5）に戻す

    return h_idx, best_score, best_alpha, coef, theta, oof_preds

# ==============================================================================
# メインループ
# ==============================================================================
dataset_list = list(DATASETS.items())

for ds_idx, (theme_name, (csv_file, vec_prefix, save_prefix)) in enumerate(dataset_list):
    print(f"\n{'=' * 60}")
    print(f"[{ds_idx+1}/{len(dataset_list)}] {theme_name}")
    print(f"{'=' * 60}")

    # データ読み込み
    csv_path = os.path.join(INPUT_DIR, csv_file)
    if not os.path.exists(csv_path):
        print(f"CSVが見つかりません: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    valid_mask = df['Stance_Value'].isin([1, 2, 3, 4, 5]).values
    df_filtered = df[valid_mask].copy()

    labels_original = df_filtered['Stance_Value'].astype(int).values
    labels_ordinal = labels_original - 1  # 0-4に変換

    print(f"データ数: {len(df_filtered)}")

    heatmap_rho = np.zeros((NUM_LAYERS, NUM_HEADS))
    best_alpha_data = np.zeros((NUM_LAYERS, NUM_HEADS))
    heatmap_coef = None
    heatmap_theta = None

    # 層ごとのループ
    for layer_idx in tqdm(range(NUM_LAYERS), desc=f"[{save_prefix}]"):
        vec_path = os.path.join(VEC_DIR, f"{vec_prefix}_layer_{layer_idx:02d}.npy")

        if not os.path.exists(vec_path):
            print(f"ベクトルが見つかりません: {vec_path}")
            continue

        full_layer_data = np.load(vec_path)
        if full_layer_data.shape[0] != len(df):
            print(f"サイズ不一致: CSV={len(df)}, NPY={full_layer_data.shape[0]}")
            continue

        layer_data = full_layer_data[valid_mask]

        if heatmap_coef is None:
            heatmap_coef = np.zeros((NUM_LAYERS, NUM_HEADS, layer_data.shape[2]))
            heatmap_theta = np.zeros((NUM_LAYERS, NUM_HEADS, 4))

        layer_oof_preds = np.zeros((len(labels_original), NUM_HEADS))

        # ヘッドごとの処理を並列実行
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

        # 層ごとに保存
        np.save(os.path.join(RESULT_DIR, "rho",   f"{save_prefix}_layer_{layer_idx:02d}_rho.npy"),   heatmap_rho[layer_idx])
        np.save(os.path.join(RESULT_DIR, "coef",  f"{save_prefix}_layer_{layer_idx:02d}_coef.npy"),  heatmap_coef[layer_idx])
        np.save(os.path.join(RESULT_DIR, "theta", f"{save_prefix}_layer_{layer_idx:02d}_theta.npy"), heatmap_theta[layer_idx])
        np.save(os.path.join(RESULT_DIR, "alpha", f"{save_prefix}_layer_{layer_idx:02d}_alpha.npy"), best_alpha_data[layer_idx])
        np.save(os.path.join(RESULT_DIR, "preds", f"{save_prefix}_layer_{layer_idx:02d}_preds.npy"), layer_oof_preds)

    # 全層統合データの保存
    if heatmap_coef is not None:
        np.save(os.path.join(RESULT_DIR, "full", f"{save_prefix}_heatmap_rho_full.npy"),  heatmap_rho)
        np.save(os.path.join(RESULT_DIR, "full", f"{save_prefix}_coef_full.npy"),         heatmap_coef)
        np.save(os.path.join(RESULT_DIR, "full", f"{save_prefix}_theta_full.npy"),        heatmap_theta)
        np.save(os.path.join(RESULT_DIR, "full", f"{save_prefix}_bestAlpha_full.npy"),    best_alpha_data)
        np.save(os.path.join(RESULT_DIR, "full", f"{save_prefix}_labels.npy"),            labels_original)

    print(f"{theme_name} 完了")

print("\n全テーマの処理が完了しました")
