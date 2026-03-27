"""
Synthetic Political Speech Data Generation

Generate synthetic political speeches for six policy issues using LLM APIs.
Each combination of conditions produces one speech sample.

Conditions:
    2 (party roles) x 4 (attributes) x 6 (targets) x 6 (situations)
    x 5 (stances) = 1,440 samples per model per issue
    x 3 models = 4,320 samples per issue

Supported models: OpenAI (GPT), Google (Gemini), Anthropic (Claude)

Usage:
    export OPENAI_API_KEY="your-key"
    export GOOGLE_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    python generate_synthetic_data.py
"""

import os
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# API key settings (read from environment variables)
# ==============================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ==============================================================================
# Model settings
# ==============================================================================
MODELS = [
    {"name": "gpt-5.1",          "provider": "openai"},
    {"name": "gemini-3.0",       "provider": "google"},
    {"name": "claude-opus-4.5",  "provider": "anthropic"},
]

# ==============================================================================
# Experimental conditions
# ==============================================================================
TOPIC_SETTINGS = {
    "defense":         "日本の防衛力はもっと強化すべきだ",
    "small_gov":       "社会福祉など政府のサービスが悪くなっても、お金のかからない小さな政府の方が良い",
    "public_works":    "公共事業による雇用確保は必要だ",
    "fiscal_stimulus": "当面は財政再建のために歳出を抑えるのではなく、景気対策のために財政出動を行うべきだ",
    "north_korea":     "北朝鮮に対しては対話よりも圧力を優先すべきだ",
    "security_privacy": "治安を守るためにプライバシーや個人の権利が制約されるのは当然だ",
}

STANCES = {
    1: "賛成",
    2: "どちらかと言えば賛成",
    3: "どちらとも言えない",
    4: "どちらかと言えば反対",
    5: "反対",
}

ROLES1 = ["与党議員", "野党議員"]
ROLES2 = ["ベテラン議員", "新人議員", "男性議員", "女性議員"]
TARGETS = ["高齢者", "子育て世代", "大学生・高校生", "保守層", "リベラル層", "無党派層"]
SITUATIONS = ["国会演説", "記者会見", "街頭演説", "新聞記事インタビュー", "自身のブログ記事", "SNS"]

# Map topic keys to output filenames (consistent with data/ directory)
TOPIC_TO_FILENAME = {
    "defense": "defense.csv",
    "small_gov": "social_welfare.csv",
    "public_works": "public_works.csv",
    "fiscal_stimulus": "fiscal_stimulus.csv",
    "north_korea": "north_korea.csv",
    "security_privacy": "public_safety.csv",
}

# ==============================================================================
# Prompt generation
# ==============================================================================
def generate_prompt(topic, stance, role1, role2, target, situation):
    """Generate a prompt for synthetic political speech."""

    # Party role instructions
    if role1 == "与党議員":
        role_instruction = (
            "【与党議員としての振る舞い】\n"
            "・あなたは「政権与党」の立場です。国を動かす責任ある主体として語ってください。\n"
            "・自身のスタンスが「賛成」の場合：「我々は責任を持って推進する」「不可欠である」と主導的・建設的に語ってください。\n"
            "・自身のスタンスが「反対」の場合：単に否定するのではなく、「財政的な裏付けが必要だ」"
            "「国民の理解を得るため慎重であるべきだ」といった責任政党としての抑制（ブレーキ役）の論理で語ってください。"
        )
    else:
        role_instruction = (
            "【野党議員としての振る舞い】\n"
            "・あなたは「野党」の立場です。政府の監視役として批判的な視点で語ってください。\n"
            "・自身のスタンスが「反対」の場合：「政府の方針は間違っている」「国民生活を無視している」と対決姿勢で語ってください。\n"
            "・自身のスタンスが「賛成」の場合：政府を褒めるのではなく、「対応が遅すぎる」「中途半端だ」"
            "「もっと断固としてやるべきだ」と政府の弱腰や至らなさを追及する形で、結果として推進を主張してください。"
        )

    # Situation instructions
    situation_instructions = {
        "記者会見": "メディアを通じた公式見解の発表の場。失言を避け、隙のない断定的な表現で淡々と語る。",
        "新聞記事インタビュー": "思想や背景を深掘りする場。「なぜそう考えるのか」という哲学や歴史的背景を含め論理的に語る。",
        "自身のブログ記事": "支持者への説明責任を果たす場。論理的な構成（起承転結）で詳細に解説する。",
        "SNS": "不特定多数への拡散を狙う場。共感を呼ぶ強い言葉やハッシュタグを使用。",
        "国会演説": "議事録に残る公式な場。極めて硬く、格調高い書き言葉で論理を展開する。",
        "街頭演説": "道行く人の足を止める場。熱量が高く、聴覚に訴えるフレーズを繰り返す。",
    }
    sit_inst = situation_instructions.get(situation, "")

    prompt = (
        f"あなたは日本の国会議員です。\n"
        f"以下の「属性」を持ち、指定された設定で発言してください。\n\n"
        f"【設定】\n"
        f"・あなたの属性：「{role1}」かつ「{role2}」\n"
        f"・テーマ：{topic}\n"
        f"・あなたの立場：{stance}\n"
        f"・相手：{target}\n"
        f"・場所：{situation}\n"
        f"・文字数：500文字程度（論理や背景を十分に展開してください）\n\n"
        f"# 重要な役割指示\n{role_instruction}\n\n"
        f"# 媒体・文脈の指示\n{sit_inst}\n\n"
        f"# ターゲットへの最適化\n"
        f"・「{target}」に最も響くレトリックを選択し、口調を調整してください。\n\n"
        f"# 絶対的な制約\n"
        f"・指定された【立場（{stance}）】の結論は絶対に崩さないでください。\n"
        f"・属性（{role2}）にふさわしいと一般的に考えられる自然な口調で話してください。\n"
        f"・出力は発言内容のみ（鍵括弧などは不要、質問文は含めない）。"
    )
    return prompt

# ==============================================================================
# API call functions
# ==============================================================================
def call_openai(prompt, model_name):
    """Call OpenAI API."""
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "あなたは政治的発言を生成するシミュレータです。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_completion_tokens=2000,
    )
    return response.choices[0].message.content

def call_google(prompt, model_name):
    """Call Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        f"あなたは政治的発言を生成するシミュレータです。\n\n{prompt}",
        generation_config=genai.types.GenerationConfig(temperature=0.8, max_output_tokens=2000),
    )
    return response.text

def call_anthropic(prompt, model_name):
    """Call Anthropic Claude API."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model_name,
        max_tokens=2000,
        temperature=0.8,
        system="あなたは政治的発言を生成するシミュレータです。",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

PROVIDER_FUNCTIONS = {
    "openai": call_openai,
    "google": call_google,
    "anthropic": call_anthropic,
}

# ==============================================================================
# Main loop
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for topic_key, topic_text in TOPIC_SETTINGS.items():
    print(f"\n{'=' * 60}")
    print(f"Issue: {topic_key}")
    print(f"{'=' * 60}")

    records = []
    idx = 1

    for model_info in MODELS:
        model_name = model_info["name"]
        provider = model_info["provider"]
        call_fn = PROVIDER_FUNCTIONS[provider]

        print(f"\nModel: {model_name}")

        for s_id, s_label in STANCES.items():
            for r1 in ROLES1:
                for r2 in ROLES2:
                    for tgt in TARGETS:
                        for sit in tqdm(SITUATIONS, desc=f"Stance {s_id}", leave=False):
                            prompt = generate_prompt(topic_text, s_label, r1, r2, tgt, sit)

                            try:
                                text = call_fn(prompt, model_name)
                            except Exception as e:
                                print(f"Error at {model_name}-{idx}: {e}")
                                text = ""

                            records.append({
                                "ID_Number": idx,
                                "Topic": topic_key,
                                "Stance_Label": s_label,
                                "Stance_Value": s_id,
                                "Role_Party": r1,
                                "Role_Attr": r2,
                                "Target": tgt,
                                "Situation": sit,
                                "Original_ID": f"{model_name}-ID-{idx:05d}",
                                "Generated_Text": text,
                            })
                            idx += 1

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, TOPIC_TO_FILENAME[topic_key])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {csv_path} ({len(df)} samples)")

print("\nAll issues completed")
