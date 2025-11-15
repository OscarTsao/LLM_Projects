"""
taide_translate_v2.py
======================

本腳本是在第一版的基礎上擴充功能，除了使用 TAIDE 模型將英文社群媒體貼文翻譯成
符合台灣口語習慣的繁體中文外，還會呼叫 Google AI Studio 提供的 Gemini API 來評估
翻譯結果的準確度與語意相似性。評估分數與簡短評語會存入同一份輸出的 CSV
檔案中。

在使用之前請先完成下列準備：

1. 取得 TAIDE 模型的使用權並準備 Hugging Face 的存取權杖。
2. 依照 Google AI Studio 的指引建立 Gemini API 金鑰。官方文件指出
   「要使用 Gemini API，需要一把 API 金鑰。可以在 Google AI Studio 中免費建立」
   【850486610150670†L260-L268】【930035816769667†L260-L268】。
   取得金鑰後可以將其設為環境變數 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`，Gemini
   API 的 Python 客戶端會自動讀取該值【850486610150670†L275-L279】；也可以直接在程式中
   傳入 `api_key` 參數。
3. 安裝必要的依賴：
   ```bash
   pip install -U google-genai pandas transformers accelerate pydantic
   ```
   - `google-genai` 是官方的 Gemini API Python SDK，可簡易呼叫生成式模型
     【930035816769667†L332-L343】。
   - `pydantic` 用來定義評估回傳的資料結構，方便解析 JSON。

使用方式：
    python taide_translate_v2.py --input input_posts.csv --output evaluated_posts.csv \
       --model taide/TAIDE-LX-7B-Chat --hf_token <HF_TOKEN> \
       --gemini_model gemini-2.5-pro --source_col post --translation_col translated_post \
       --evaluation_col evaluation

參數說明：
    --input/-i         原始 CSV 檔案路徑。
    --output/-o        含翻譯與評估結果的輸出 CSV 檔案路徑。
    --model/-m         Hugging Face 上的 TAIDE 模型名稱，預設為 "taide/TAIDE-LX-7B-Chat"。
    --hf_token/-t      Hugging Face 存取權杖，用於讀取 gated 模型。
    --gemini_model     要用於評估的 Gemini 模型名稱，例如 "gemini-2.5-pro" 或
                       "gemini-2.5-flash"，預設為 "gemini-2.5-pro"。
    --source_col/-s    原始英文貼文欄位名稱，預設為 "post"。
    --translation_col  翻譯後欄位名稱，預設為 "translated_post"。
    --evaluation_col   評估結果欄位名稱，預設為 "evaluation"。
    --no-quant         如果指定，則不啟用 4bit 量化載入模型。
    --max_tokens       每條翻譯的最大輸出 token 數，預設 256。
    --gemini_key       如果沒有預先設定環境變數，可透過此參數顯式傳入
                       Gemini API 金鑰。

輸出的 CSV 會包含三個欄位：原文 (source_col)、翻譯後文本 (translation_col) 以及
Gemini API 回傳的評估結果 (evaluation_col)。評估結果為 JSON 字串，內含
`accuracy`、`similarity` 兩個 0~100 的分數以及 `feedback` 評語。

注意：請勿將金鑰寫死在原始碼中並公開至版本控制系統，應使用環境變數或外部
設定檔來管理敏感資訊【850486610150670†L451-L461】。
"""

import argparse
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    BitsAndBytesConfig,
)
import torch

# Google Gemini SDK
try:
    from google import genai
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "本腳本需要安裝 google-genai 和 pydantic。請先執行 pip install -U google-genai pydantic"
    ) from e


def load_taide_pipeline(
    model_name: str, hf_token: str, use_quantization: bool = True
) -> Tuple[TextGenerationPipeline, AutoTokenizer]:
    """載入 TAIDE 模型並返回 text-generation pipeline 與 tokenizer。"""
    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        token=hf_token,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )

    pipe = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        do_sample=False,
    )
    return pipe, tokenizer


def translate_text(
    pipe: TextGenerationPipeline,
    tokenizer: AutoTokenizer,
    text: str,
    system_prompt: str = "你是一位翻譯專家，擅長將英文社群媒體貼文翻譯成符合台灣口語習慣的繁體中文。請使用口語化的字詞，避免過於正式的語氣。",
    max_new_tokens: int = 256,
) -> str:
    """使用 TAIDE 模型翻譯單句英文文字。"""
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"請翻譯以下內容：{text}"},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    output = pipe(
        prompt, max_new_tokens=max_new_tokens, return_full_text=False
    )[0]["generated_text"]
    return output.strip()


class EvaluationResult(BaseModel):
    """定義評估結果的結構，用於 Gemini API 的結構化輸出。"""

    accuracy: float
    similarity: float
    feedback: str


def evaluate_translation(
    client: genai.Client,
    original: str,
    translated: str,
    gemini_model: str = "gemini-2.5-pro",
) -> EvaluationResult:
    """使用 Gemini API 評估翻譯的準確度與相似性。

    會要求模型輸出包含 `accuracy`、`similarity` 以及 `feedback` 的 JSON。數值
    範圍為 0–100，feedback 為簡短中文評語。
    """
    # 建立提示詞：system 提示描述任務與輸出格式，user 提供原文與翻譯
    system_prompt = (
        "你是一位雙語審核員，專門評估英文與翻譯成繁體中文的貼文。"
        "請根據翻譯與原文的語意相符程度，以及翻譯是否符合台灣口語化表達，"
        "給出兩個 0 到 100 的分數：accuracy 表示翻譯內容是否忠於原文，"
        "similarity 表示語意相似度。請再提供一行簡短中文評語 (feedback)。"
        "最後請以 JSON 格式輸出，結構為 {\"accuracy\": <數字>, \"similarity\": <數字>, \"feedback\": \"評語\"}。"
    )
    user_prompt = f"原文：{original}\n翻譯：{translated}"
    # Gemini SDK 接受單一字串內容；包含 system 與 user 角色無法直接使用於 generate_content
    # 因此將兩段文字合併為一個 prompt
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    # 透過 response_mime_type 指定輸出為 JSON，並用 Pydantic 模型描述 schema
    response = client.models.generate_content(
        model=gemini_model,
        contents=full_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": EvaluationResult,
        },
    )
    # 使用 parsed 屬性自動解析為 Pydantic 模型
    try:
        result: EvaluationResult = response.parsed  # type: ignore
        return result
    except Exception:
        # 如果無法解析，則嘗試自行解析 JSON 字串
        try:
            data: Dict[str, Any] = json.loads(response.text)
            return EvaluationResult(
                accuracy=float(data.get("accuracy", 0)),
                similarity=float(data.get("similarity", 0)),
                feedback=str(data.get("feedback", "")),
            )
        except Exception as e:
            # 當解析失敗時回傳預設值
            return EvaluationResult(accuracy=0.0, similarity=0.0, feedback=f"解析失敗: {e}")


def translate_and_evaluate_csv(
    input_path: str,
    output_path: str,
    model_name: str,
    hf_token: str,
    gemini_model: str = "gemini-2.5-pro",
    gemini_key: Optional[str] = None,
    source_col: str = "post",
    translation_col: str = "translated_post",
    evaluation_col: str = "evaluation",
    use_quantization: bool = True,
    max_new_tokens: int = 256,
) -> None:
    """讀取 CSV，使用 TAIDE 模型翻譯並透過 Gemini API 評估，最後寫入新的 CSV。"""
    pipe, tokenizer = load_taide_pipeline(
        model_name, hf_token, use_quantization=use_quantization
    )
    # 初始化 Gemini 用戶端。若環境中有 GEMINI_API_KEY 或 GOOGLE_API_KEY，
    # 則可直接呼叫 Client()；否則需傳入 api_key
    if gemini_key:
        client = genai.Client(api_key=gemini_key)
    else:
        client = genai.Client()
    # 讀取 CSV
    df = pd.read_csv(input_path)
    if source_col not in df.columns:
        raise ValueError(f"輸入 CSV 中找不到欄位 {source_col}")
    translations = []
    evaluations = []
    for content in df[source_col].astype(str):
        # 翻譯
        translation = translate_text(
            pipe,
            tokenizer,
            content,
            max_new_tokens=max_new_tokens,
        )
        translations.append(translation)
        # 評估
        evaluation_result = evaluate_translation(
            client, content, translation, gemini_model=gemini_model
        )
        # 儲存為 JSON 字串
        evaluations.append(
            json.dumps(
                {
                    "accuracy": evaluation_result.accuracy,
                    "similarity": evaluation_result.similarity,
                    "feedback": evaluation_result.feedback,
                },
                ensure_ascii=False,
            )
        )
    # 新增欄位並寫出 CSV
    df[translation_col] = translations
    df[evaluation_col] = evaluations
    df.to_csv(output_path, index=False)
    print(f"翻譯與評估完成，輸出檔案已寫入 {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Translate English posts into colloquial Taiwanese Mandarin with TAIDE and "
            "evaluate the translations using Gemini API."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input CSV file containing English posts",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output CSV file with translations and evaluations",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="taide/TAIDE-LX-7B-Chat",
        help=(
            "Name of the TAIDE model to use (e.g., taide/TAIDE-LX-7B-Chat or "
            "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1)"
        ),
    )
    parser.add_argument(
        "--hf_token",
        "-t",
        required=True,
        help="Your Hugging Face access token for gated models",
    )
    parser.add_argument(
        "--gemini_model",
        default="gemini-2.5-pro",
        help="Gemini model to use for evaluation (e.g., gemini-2.5-pro or gemini-2.5-flash)",
    )
    parser.add_argument(
        "--gemini_key",
        default=None,
        help="Gemini API key (optional). If not provided, the client will try to read from environment variables",
    )
    parser.add_argument(
        "--source_col",
        "-s",
        default="post",
        help="Column name of English posts in the input CSV",
    )
    parser.add_argument(
        "--translation_col",
        default="translated_post",
        help="Column name for translated posts in the output CSV",
    )
    parser.add_argument(
        "--evaluation_col",
        default="evaluation",
        help="Column name for Gemini evaluation results in the output CSV",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable 4bit quantization when loading the TAIDE model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate for each translation",
    )
    args = parser.parse_args()

    translate_and_evaluate_csv(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        hf_token=args.hf_token,
        gemini_model=args.gemini_model,
        gemini_key=args.gemini_key,
        source_col=args.source_col,
        translation_col=args.translation_col,
        evaluation_col=args.evaluation_col,
        use_quantization=not args.no_quant,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()