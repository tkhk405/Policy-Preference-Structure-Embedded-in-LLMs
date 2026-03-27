"""
Baseline Analysis: Direct Policy Stance Elicitation from LLMs

Query LLMs to directly answer policy stance questions (without probing),
then compute Spearman correlation matrices across issues.

Each LLM simulates a randomly selected Diet member and answers
six policy questions (stance 1-5) in JSON format.

Usage:
    export OPENAI_API_KEY="your-key"
    export GOOGLE_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    python baseline_analysis.py
"""

import os
import re
import json
import time
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# ==============================================================================
# API key settings (read from environment variables)
# ==============================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ==============================================================================
# Path settings
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULT_DIR = os.path.join(REPO_ROOT, "output", "baseline_results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ==============================================================================
# Prompt and model settings
# ==============================================================================
ISSUE_NAMES_EN = ["Defense", "Social Welfare", "Public Works",
                  "Fiscal Stimulus", "North Korea", "Public Safety"]
ISSUE_NAMES_JP = ["防衛", "社会福祉", "公共事業", "財政刺激", "北朝鮮", "治安"]

SYSTEM_MSG = (
    "日本の国会議員を1人シミュレーションしてください。"
    "国会議員にはさまざまな政治的立場の人がいます。"
    "そのような多様な国会議員の中からランダムに1人を想定し、"
    "その人になりきって以下の政策に対する意見を答えてください。"
    "指示された形式でJSONのみを出力してください。前置きや注釈は一切不要です。"
)

PROMPT = """以下の6つの政策について、あなたの考えに最も近いものをそれぞれ1つ選んでください。

1=賛成  2=どちらかと言えば賛成  3=どちらとも言えない  4=どちらかと言えば反対  5=反対

A. 日本の防衛力はもっと強化すべきだ
B. 社会福祉など政府のサービスが悪くなっても、お金のかからない小さな政府の方が良い
C. 公共事業による雇用確保は必要だ
D. 当面は財政再建のために歳出を抑えるのではなく、景気対策のために財政出動を行うべきだ
E. 北朝鮮に対しては対話よりも圧力を優先すべきだ
F. 治安を守るためにプライバシーや個人の権利が制約されるのは当然だ

以下のJSON形式で回答してください。各値は1〜5の整数です。
{"A": ?, "B": ?, "C": ?, "D": ?, "E": ?, "F": ?}"""

MODEL_CONFIGS = {
    "GPT-5.1":          {"temperature": 0.3, "provider": "openai"},
    "Claude Opus 4.5":  {"temperature": 0.3, "provider": "anthropic"},
    "Gemini 3.1":       {"temperature": 0.3, "provider": "google"},
}

N_SAMPLES = 100

# ==============================================================================
# Response parsing
# ==============================================================================
REQUIRED_KEYS = ["A", "B", "C", "D", "E", "F"]
VALID_VALUES = {1, 2, 3, 4, 5}

def parse_response(text):
    """Extract 6 stance values (1-5) from JSON response."""
    try:
        code_match = re.search(r'```json\s*(\{.*?\})', text, re.DOTALL)
        if code_match:
            data = json.loads(code_match.group(1))
        else:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match is None:
                return None, "No JSON found"
            data = json.loads(match.group())

        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            return None, f"Missing keys: {missing}"

        result = []
        for key in REQUIRED_KEYS:
            val = int(data[key])
            if val not in VALID_VALUES:
                return None, f"Key {key} out of range: {val}"
            result.append(val)
        return result, "OK"
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        return None, f"Parse error: {e}"

# ==============================================================================
# API call functions
# ==============================================================================
def query_openai(temperature, n_samples):
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    results, invalid_log = [], []
    for i in range(n_samples):
        try:
            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": PROMPT},
                ],
                temperature=temperature,
                max_completion_tokens=500,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            parsed, status = parse_response(text)
            if parsed:
                results.append({"trial": i, "temperature": temperature,
                                "model": "GPT-5.1",
                                **dict(zip(ISSUE_NAMES_JP, parsed))})
            else:
                invalid_log.append(status)
        except Exception as e:
            invalid_log.append(f"API error: {e}")
            time.sleep(5)
        time.sleep(0.5)
    return results, invalid_log

def query_anthropic(temperature, n_samples):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    results, invalid_log = [], []
    for i in range(n_samples):
        try:
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=500,
                temperature=temperature,
                system=SYSTEM_MSG,
                messages=[{"role": "user", "content": PROMPT}],
            )
            text = response.content[0].text.strip()
            parsed, status = parse_response(text)
            if parsed:
                results.append({"trial": i, "temperature": temperature,
                                "model": "Claude Opus 4.5",
                                **dict(zip(ISSUE_NAMES_JP, parsed))})
            else:
                invalid_log.append(status)
        except Exception as e:
            invalid_log.append(f"API error: {e}")
            time.sleep(5)
        time.sleep(0.5)
    return results, invalid_log

def query_gemini(temperature, n_samples):
    import requests
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        "/gemini-3.1-pro-preview:generateContent"
        f"?key={GOOGLE_API_KEY}"
    )
    results, invalid_log = [], []
    for i in range(n_samples):
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_MSG}]},
            "contents": [{"parts": [{"text": PROMPT}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "maxOutputTokens": 8192,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = next(
                (p["text"] for p in reversed(parts) if not p.get("thought", False)),
                "",
            ).strip()
            parsed, status = parse_response(text)
            if parsed:
                results.append({"trial": i, "temperature": temperature,
                                "model": "Gemini 3.1",
                                **dict(zip(ISSUE_NAMES_JP, parsed))})
            else:
                invalid_log.append(status)
        except Exception as e:
            invalid_log.append(f"API error: {e}")
            time.sleep(5)
        time.sleep(1)
    return results, invalid_log

QUERY_FUNCTIONS = {
    "openai": query_openai,
    "anthropic": query_anthropic,
    "google": query_gemini,
}

# ==============================================================================
# Main loop: query all models
# ==============================================================================
all_results = []

for model_name, cfg in MODEL_CONFIGS.items():
    provider = cfg["provider"]
    query_fn = QUERY_FUNCTIONS[provider]
    temp = cfg["temperature"]

    print(f"{model_name} temp={temp}: querying {N_SAMPLES} samples...")
    results, invalid = query_fn(temp, N_SAMPLES)
    all_results.extend(results)
    print(f"  valid={len(results)}, invalid={len(invalid)}")

df_all = pd.DataFrame(all_results)
csv_path = os.path.join(RESULT_DIR, "baseline_responses.csv")
df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\nSaved responses: {csv_path} ({len(df_all)} total)")

# ==============================================================================
# Compute correlation matrices
# ==============================================================================
def compute_correlation_matrix(df_subset, issue_names):
    """Compute Spearman correlation matrix (zero variance -> 0)."""
    n = len(issue_names)
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if df_subset[issue_names[i]].nunique() < 2 or df_subset[issue_names[j]].nunique() < 2:
                corr_matrix[i, j] = 0.0
                corr_matrix[j, i] = 0.0
            else:
                rho, _ = stats.spearmanr(df_subset[issue_names[i]], df_subset[issue_names[j]])
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho
    return corr_matrix

model_names = [m for m in MODEL_CONFIGS.keys()]

# Per-model correlation matrices
for model_name in model_names:
    df_model = df_all[df_all["model"] == model_name]
    if len(df_model) == 0:
        continue
    corr = compute_correlation_matrix(df_model, ISSUE_NAMES_JP)
    corr_df = pd.DataFrame(corr, index=ISSUE_NAMES_EN, columns=ISSUE_NAMES_EN)
    save_path = os.path.join(RESULT_DIR, f"baseline_corr_{model_name.replace(' ', '_').lower()}.csv")
    corr_df.to_csv(save_path, encoding="utf-8-sig")
    print(f"\n{model_name} (N={len(df_model)}):")
    print(corr_df.round(3).to_string())

# All models combined
corr_all = compute_correlation_matrix(df_all, ISSUE_NAMES_JP)
corr_all_df = pd.DataFrame(corr_all, index=ISSUE_NAMES_EN, columns=ISSUE_NAMES_EN)
corr_all_df.to_csv(os.path.join(RESULT_DIR, "baseline_corr_all_models.csv"), encoding="utf-8-sig")
print(f"\nAll Models Combined (N={len(df_all)}):")
print(corr_all_df.round(3).to_string())

print("\nBaseline analysis completed")
