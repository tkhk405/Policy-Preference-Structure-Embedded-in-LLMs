"""
活性化ベクトル抽出コード（Gemma-2-Llama-Swallow-9b）

Gemma-2-Llama-Swallow-9b-pt-v0.1 の各層アテンションヘッドから
活性化ベクトルを抽出する。

- モデル: tokyotech-llm/Gemma-2-Llama-Swallow-9b-pt-v0.1
- 構造: 42層 × 16ヘッド × 256次元/ヘッド
- 抽出方法: self_attn.o_proj の入力（線形変換前）をforward hookで取得
- 抽出トークン: シーケンス最後のトークン
- 保存形状: (サンプル数, 16, 256)

使用方法:
    git clone https://github.com/tkhk405/Policy-Preference-Structure-Embedded-in-LLMs.git
    cd Policy-Preference-Structure-Embedded-in-LLMs/code
    python extract_activations.py
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
import gc

# ==============================================================================
# パス設定
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "activation_vectors")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# データセット定義（6テーマ）
# ==============================================================================
DATASETS = {
    "Defense (防衛)":            ("defense.csv",         "Defense"),
    "Social Welfare (社会福祉)": ("social_welfare.csv",  "Social"),
    "Public Works (公共事業)":   ("public_works.csv",    "Public"),
    "Fiscal Stimulus (財政出動)": ("fiscal_stimulus.csv", "Fiscal"),
    "North Korea (北朝鮮)":      ("north_korea.csv",     "Nkorea"),
    "Public Safety (治安)":      ("public_safety.csv",   "Security"),
}

# ==============================================================================
# モデルの読み込み
# ==============================================================================
MODEL_ID = "tokyotech-llm/Gemma-2-Llama-Swallow-9b-pt-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

num_layers = model.config.num_hidden_layers   # 42
num_heads = model.config.num_attention_heads  # 16

# ==============================================================================
# Hook関数の定義
# ==============================================================================
current_layer_activations = []

def get_single_layer_hook(num_heads):
    """o_proj層の入力（線形変換前）をヘッドごとに分割して取得するHook"""

    def hook(module, input, output):
        hidden_states = input[0]
        input_dim = hidden_states.shape[-1]
        head_dim = input_dim // num_heads

        # 最後のトークンのみ抽出し、ヘッドごとに分割
        last_token = hidden_states[:, -1, :]
        batch_size = last_token.shape[0]
        reshaped = last_token.view(batch_size, num_heads, head_dim)
        current_layer_activations.append(reshaped.detach().cpu().numpy())

    return hook

# ==============================================================================
# メインループ（6テーマ × 42層）
# ==============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

for theme_name, (csv_file, save_prefix) in DATASETS.items():
    print(f"\n{'=' * 60}")
    print(f"テーマ: {theme_name}")
    print(f"{'=' * 60}")

    gc.collect()
    torch.cuda.empty_cache()

    # データ読み込み
    file_path = os.path.join(INPUT_DIR, csv_file)
    if not os.path.exists(file_path):
        print(f"ファイルが見つかりません: {file_path}")
        continue

    df = pd.read_csv(file_path)
    texts = df['Generated_Text'].tolist()
    print(f"データ数: {len(texts)} 件")

    # 層ごとにベクトルを抽出
    for layer_idx in tqdm(range(num_layers), desc=f"[{save_prefix}]"):
        save_path = os.path.join(OUTPUT_DIR, f"{save_prefix}_layer_{layer_idx:02d}.npy")

        # 計算済みならスキップ
        if os.path.exists(save_path):
            continue

        # Hook設定
        current_layer_activations = []
        layer_module = model.model.layers[layer_idx].self_attn.o_proj
        handle = layer_module.register_forward_hook(get_single_layer_hook(num_heads))

        # バッチ推論
        for i in range(0, len(texts), 8):
            batch = texts[i:i+8]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(model.device)

            with torch.no_grad():
                model(**inputs)

            del inputs, batch
            torch.cuda.empty_cache()

        handle.remove()

        # 保存: (サンプル数, 16, 256)
        layer_data = np.concatenate(current_layer_activations, axis=0)
        np.save(save_path, layer_data)

        del layer_data, current_layer_activations
        gc.collect()
        torch.cuda.empty_cache()

    print(f"{theme_name} 完了")

print("\n全テーマの処理が完了しました")
