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

def ensure_result_dir():
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
    "GPT-5.1":          {"temperature": 0.8, "provider": "openai"},
    "Claude Opus 4.5":  {"temperature": 0.8, "provider": "anthropic"},
    "Gemini 3.1":       {"temperature": 0.8, "provider": "google"},
}

N_SAMPLES = 100
STANCE_VALUES = [1, 2, 3, 4, 5]

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
def run_queries():
    """Query all configured models and save raw valid responses."""
    ensure_result_dir()
    all_results = []
    invalid_summary = []

    for model_name, cfg in MODEL_CONFIGS.items():
        provider = cfg["provider"]
        query_fn = QUERY_FUNCTIONS[provider]
        temp = cfg["temperature"]

        print(f"{model_name} temp={temp}: querying {N_SAMPLES} samples...")
        results, invalid = query_fn(temp, N_SAMPLES)
        all_results.extend(results)
        invalid_summary.append({
            "model": model_name,
            "n_valid": len(results),
            "n_invalid": len(invalid),
            "invalid_messages": invalid,
        })
        print(f"  valid={len(results)}, invalid={len(invalid)}")

    df_all = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULT_DIR, "baseline_responses.csv")
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

    invalid_path = os.path.join(RESULT_DIR, "baseline_invalid_responses.json")
    with open(invalid_path, "w", encoding="utf-8") as f:
        json.dump(invalid_summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved responses: {csv_path} ({len(df_all)} total)")
    print(f"Saved invalid-response log: {invalid_path}")
    return df_all

# ==============================================================================
# Compute correlation matrices
# ==============================================================================
def compute_correlation_matrix(df_subset, issue_names):
    """Compute Spearman correlation matrix.

    If either issue has zero variance, Spearman's rho is undefined. The paper
    treats such cases as preventing construction of a complete correlation
    matrix, so the corresponding cells are stored as NaN rather than being
    forced to zero.
    """
    n = len(issue_names)
    corr_matrix = np.ones((n, n))
    undefined_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if df_subset[issue_names[i]].nunique() < 2 or df_subset[issue_names[j]].nunique() < 2:
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan
                undefined_pairs.append((issue_names[i], issue_names[j]))
            else:
                rho, _ = stats.spearmanr(df_subset[issue_names[i]], df_subset[issue_names[j]])
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho
    return corr_matrix, undefined_pairs

def save_response_distribution(df_all):
    """Save response distributions and, if matplotlib is available, Fig. 1."""
    ensure_result_dir()
    long_df = df_all.melt(
        id_vars=["trial", "temperature", "model"],
        value_vars=ISSUE_NAMES_JP,
        var_name="issue_jp",
        value_name="stance",
    )
    issue_map = dict(zip(ISSUE_NAMES_JP, ISSUE_NAMES_EN))
    long_df["issue"] = long_df["issue_jp"].map(issue_map)

    counts = (
        long_df.groupby(["model", "issue", "stance"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(["model", "issue"])["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals

    dist_path = os.path.join(RESULT_DIR, "baseline_response_distribution.csv")
    counts.to_csv(dist_path, index=False, encoding="utf-8-sig")
    print(f"Saved response distribution: {dist_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped figure generation.")
        return

    model_names = list(MODEL_CONFIGS.keys())
    fig, axes = plt.subplots(
        len(model_names),
        len(ISSUE_NAMES_EN),
        figsize=(13.5, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row, model_name in enumerate(model_names):
        for col, (issue_jp, issue_en) in enumerate(zip(ISSUE_NAMES_JP, ISSUE_NAMES_EN)):
            ax = axes[row, col]
            subset = df_all[df_all["model"] == model_name]
            values = subset[issue_jp].value_counts().reindex(STANCE_VALUES, fill_value=0)
            ax.bar(STANCE_VALUES, values.values, color="#4c78a8", width=0.75)
            ax.set_xticks(STANCE_VALUES)
            if row == 0:
                ax.set_title(issue_en, fontsize=9)
            if col == 0:
                ax.set_ylabel(model_name, fontsize=9)
            ax.tick_params(axis="both", labelsize=8)

    fig.supxlabel("Stance (1 = agree, 5 = disagree)", fontsize=10)
    fig.supylabel("Count", fontsize=10)

    png_path = os.path.join(RESULT_DIR, "baseline_random_distribution.png")
    pdf_path = os.path.join(RESULT_DIR, "baseline_random_distribution.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved response distribution figure: {png_path}")
    print(f"Saved response distribution figure: {pdf_path}")

def save_correlation_outputs(df_all):
    """Save per-model correlation matrices and a validity summary."""
    ensure_result_dir()
    model_names = [m for m in MODEL_CONFIGS.keys()]
    validity_rows = []

    for model_name in model_names:
        df_model = df_all[df_all["model"] == model_name]
        if len(df_model) == 0:
            validity_rows.append({
                "model": model_name,
                "n": 0,
                "complete_correlation_matrix": False,
                "undefined_pairs": "No valid responses",
                "mantel_applicable": False,
            })
            continue

        corr, undefined_pairs = compute_correlation_matrix(df_model, ISSUE_NAMES_JP)
        corr_df = pd.DataFrame(corr, index=ISSUE_NAMES_EN, columns=ISSUE_NAMES_EN)
        save_path = os.path.join(RESULT_DIR, f"baseline_corr_{model_name.replace(' ', '_').lower()}.csv")
        corr_df.to_csv(save_path, encoding="utf-8-sig")

        complete = len(undefined_pairs) == 0
        validity_rows.append({
            "model": model_name,
            "n": len(df_model),
            "complete_correlation_matrix": complete,
            "undefined_pairs": "; ".join([f"{a}-{b}" for a, b in undefined_pairs]),
            "mantel_applicable": complete,
        })

        print(f"\n{model_name} (N={len(df_model)}):")
        if complete:
            print(corr_df.round(3).to_string())
        else:
            print(corr_df.round(3).to_string())
            print("  Note: Spearman correlations involving zero-variance issues are undefined (NaN).")
            print("  Mantel test with this incomplete matrix is not applicable.")

    validity_df = pd.DataFrame(validity_rows)
    validity_path = os.path.join(RESULT_DIR, "baseline_matrix_validity.csv")
    validity_df.to_csv(validity_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved baseline matrix validity summary: {validity_path}")

def main():
    df_all = run_queries()
    if df_all.empty:
        print("\nNo valid responses were obtained; skipping analysis.")
        return
    save_response_distribution(df_all)
    save_correlation_outputs(df_all)
    print("\nBaseline analysis completed")

if __name__ == "__main__":
    main()
