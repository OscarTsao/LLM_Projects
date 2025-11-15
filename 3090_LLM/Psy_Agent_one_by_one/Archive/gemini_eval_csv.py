"""
gemini_eval_csv.py
===================

這支腳本從一個已有的 CSV 檔讀取兩個欄位：英文原文 `post` 和翻譯後的繁體中文
`translated_post`，並使用 Google AI Studio 提供的 Gemini API 對翻譯的準確度與
語意相似性進行評分。評分結果會以 JSON 字串寫入第三個欄位 `gemini_eval`，
並將結果存回 CSV 檔案。

使用前請完成下列準備：

1. **取得 API 金鑰**：官方文件指出，使用 Gemini API 必須先在 Google AI Studio
   建立 API 金鑰【850486610150670†L260-L268】。金鑰可透過設定環境變數
   `GEMINI_API_KEY` 或 `GOOGLE_API_KEY` 提供給 SDK，自動被讀取【850486610150670†L275-L279】。

2. **安裝依賴**：
   ```bash
   pip install -U google-genai pandas tqdm pydantic
   ```
   - `google-genai` 是 Google 官方的 SDK，用於呼叫 Gemini API【930035816769667†L332-L344】。
   - `pydantic` 用來定義評估結果的資料結構，並與 `response_schema` 配合讓
     模型輸出結構化 JSON【625196986970676†L267-L297】。

使用方式：

```bash
python gemini_eval_csv.py --input output.csv --output output_with_eval.csv \
    --model gemini-2.5-pro --api_key <YOUR_GEMINI_API_KEY>
```

參數：
    --input/-i     需要評估的 CSV 檔案路徑，必須包含 `post` 和 `translated_post` 兩欄。
    --output/-o    輸出 CSV 檔案路徑。若未指定則覆寫原檔案。
    --model/-m     使用的 Gemini 模型名稱，例如 `gemini-2.5-flash` 或 `gemini-2.5-pro`，預設為
                   `gemini-2.5-flash`。
    --api_key/-k   可選，顯式提供 API 金鑰。如果未傳入，SDK 會自動讀取環境變數。

產出的 CSV 檔會新增一個欄位 `gemini_eval`，內含以下 JSON 欄位：
    - accuracy：0-100 的整數分數，表示翻譯的忠實度。
    - similarity：0.0-1.0 的浮點數，表示語意相似度。
    - verdict：對品質的簡短評價（例如 "poor"、"fair"、"good"、"excellent"）。
    - comment：簡短中文評語。

注意：為保護敏感資訊，請勿將 API 金鑰直接寫入程式碼或共享倉庫，建議透過環境
變數或安全的配置管理方式提供金鑰【850486610150670†L451-L461】。
"""

import argparse
import json
import time
from typing import Optional

import pandas as pd
from tqdm import tqdm

try:
    from google import genai
    from pydantic import BaseModel, Field
    from typing import Literal
except ImportError as e:
    raise ImportError(
        "請先安裝 google-genai、pydantic 等依賴，使用 pip install -U google-genai pydantic"
    ) from e


# 提示語設定
EVAL_SYSTEM_PROMPT = (
    "You are a bilingual evaluator specializing in English → Traditional Chinese (Taiwan)"
    " translation for social media. Assess the translation for faithfulness, meaning preservation,"
    " adequacy, fluency, and Taiwan-style colloquial appropriateness."
)

EVAL_USER_PROMPT_TEMPLATE = """Evaluate the translation quality.

Source (EN):
{source}

Translation (ZH-TW, colloquial):
{translation}

Instructions:
1) Accuracy (0–100): how faithful the translation is to the source's meaning.
2) Similarity (0.0–1.0): semantic similarity between source and translation.
3) Verdict: one of [\"poor\",\"fair\",\"good\",\"excellent\"].
4) Comment: brief rationale (Traditional Chinese, ≤60 chars).

Return ONLY JSON following the response schema.
"""


# Pydantic 模型描述 JSON 結構
class EvalResult(BaseModel):
    accuracy: int = Field(..., ge=0, le=100)
    similarity: float = Field(..., ge=0.0, le=1.0)
    verdict: Literal["poor", "fair", "good", "excellent"]
    comment: str


def build_client(api_key: Optional[str] = None) -> genai.Client:
    """建立 Gemini API 客戶端。若提供 api_key 則使用之，否則自動讀取環境變數。"""
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def eval_one(
    client: genai.Client,
    model: str,
    source_en: str,
    zh_tw_translation: str,
    max_retries: int = 3,
    timeout_backoff_sec: float = 2.0,
) -> dict:
    """呼叫 Gemini API 評估單一翻譯，返回 dict。"""
    contents = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": EVAL_USER_PROMPT_TEMPLATE.format(
                source=source_en.strip(), translation=zh_tw_translation.strip()
            ),
        },
    ]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": EvalResult,
                },
            )
            # 若 SDK 能解析則使用 parsed
            if getattr(response, "parsed", None):
                parsed: EvalResult = response.parsed  # type: ignore
                return {
                    "accuracy": parsed.accuracy,
                    "similarity": parsed.similarity,
                    "verdict": parsed.verdict,
                    "comment": parsed.comment,
                }
            # 否則嘗試讀取原始 JSON
            text = (response.text or "").strip()
            if text:
                return json.loads(text)
            # 無結果
            return {
                "accuracy": None,
                "similarity": None,
                "verdict": None,
                "comment": "",
            }
        except Exception as e:
            if attempt >= max_retries:
                return {
                    "accuracy": None,
                    "similarity": None,
                    "verdict": "error",
                    "comment": f"Gemini error: {e.__class__.__name__}",
                }
            time.sleep(timeout_backoff_sec * attempt)
    return {
        "accuracy": None,
        "similarity": None,
        "verdict": None,
        "comment": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use Gemini API to evaluate translation quality in an existing CSV file."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default="output.csv",
        help="Path to the CSV file containing columns 'post' and 'translated_post'.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Path to save the CSV with evaluation results. If omitted, the input file is overwritten."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gemini-2.5-flash",
        help=(
            "Gemini model name for evaluation, e.g., gemini-2.5-flash or gemini-2.5-pro."
        ),
    )
    parser.add_argument(
        "--api_key",
        "-k",
        default=None,
        help="Optional Gemini API key. If not provided, the SDK uses GEMINI_API_KEY from the environment.",
    )
    args = parser.parse_args()
    out_path = args.output or args.input
    # 讀取 CSV
    df = pd.read_csv(args.input)
    if "post" not in df.columns or "translated_post" not in df.columns:
        raise ValueError(
            "Input CSV must contain columns 'post' and 'translated_post'."
        )
    # 建立客戶端
    client = build_client(api_key=args.api_key)
    evaluations = []
    for src, tgt in tqdm(
        zip(df["post"].astype(str), df["translated_post"].astype(str)),
        total=len(df),
        desc="Evaluating with Gemini",
    ):
        if not src.strip() or not tgt.strip():
            evaluations.append(
                json.dumps(
                    {
                        "accuracy": None,
                        "similarity": None,
                        "verdict": "skip",
                        "comment": "empty source/translation",
                    },
                    ensure_ascii=False,
                )
            )
            continue
        result = eval_one(client, args.model, src, tgt)
        evaluations.append(json.dumps(result, ensure_ascii=False))
    # 將結果寫入新欄位
    df["gemini_eval"] = evaluations
    df.to_csv(out_path, index=False)
    print(f"Finished. Results saved to: {out_path}")


if __name__ == "__main__":
    main()