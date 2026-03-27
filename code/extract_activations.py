"""
Activation Vector Extraction (Gemma-2-Llama-Swallow-9b)

Extract activation vectors from each layer's attention heads of
Gemma-2-Llama-Swallow-9b-pt-v0.1.

- Model: tokyotech-llm/Gemma-2-Llama-Swallow-9b-pt-v0.1
- Architecture: 42 layers x 16 heads x 256 dims/head
- Method: Forward hook on self_attn.o_proj input (before linear projection)
- Token: Last token of each sequence
- Output shape: (num_samples, 16, 256)

Usage:
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
# Path settings
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "activation_vectors")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Dataset definitions (6 policy issues)
# ==============================================================================
DATASETS = {
    "Defense":         ("defense.csv",         "Defense"),
    "Social Welfare":  ("social_welfare.csv",  "Social"),
    "Public Works":    ("public_works.csv",    "Public"),
    "Fiscal Stimulus": ("fiscal_stimulus.csv", "Fiscal"),
    "North Korea":     ("north_korea.csv",     "Nkorea"),
    "Security":        ("public_safety.csv",   "Security"),
}

# ==============================================================================
# Load model
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
# Hook function
# ==============================================================================
current_layer_activations = []

def get_single_layer_hook(num_heads):
    """Hook to capture o_proj input (before linear projection), split by head."""

    def hook(module, input, output):
        hidden_states = input[0]
        input_dim = hidden_states.shape[-1]
        head_dim = input_dim // num_heads

        # Extract last token and reshape into per-head vectors
        last_token = hidden_states[:, -1, :]
        batch_size = last_token.shape[0]
        reshaped = last_token.view(batch_size, num_heads, head_dim)
        current_layer_activations.append(reshaped.detach().cpu().numpy())

    return hook

# ==============================================================================
# Main loop (6 issues x 42 layers)
# ==============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

for theme_name, (csv_file, save_prefix) in DATASETS.items():
    print(f"\n{'=' * 60}")
    print(f"Issue: {theme_name}")
    print(f"{'=' * 60}")

    gc.collect()
    torch.cuda.empty_cache()

    # Load data
    file_path = os.path.join(INPUT_DIR, csv_file)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)
    texts = df['Generated_Text'].tolist()
    print(f"Samples: {len(texts)}")

    # Extract vectors layer by layer
    for layer_idx in tqdm(range(num_layers), desc=f"[{save_prefix}]"):
        save_path = os.path.join(OUTPUT_DIR, f"{save_prefix}_layer_{layer_idx:02d}.npy")

        # Skip if already computed
        if os.path.exists(save_path):
            continue

        # Register hook
        current_layer_activations = []
        layer_module = model.model.layers[layer_idx].self_attn.o_proj
        handle = layer_module.register_forward_hook(get_single_layer_hook(num_heads))

        # Batch inference
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

        # Save: (num_samples, 16, 256)
        layer_data = np.concatenate(current_layer_activations, axis=0)
        np.save(save_path, layer_data)

        del layer_data, current_layer_activations
        gc.collect()
        torch.cuda.empty_cache()

    print(f"{theme_name} done")

print("\nAll issues completed")
