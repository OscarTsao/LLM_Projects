"""
taide_translate.py
===================

此腳本用於讀取包含英文社群媒體貼文的 CSV 檔，利用 TAIDE 模型將貼文翻譯成符合
台灣口語習慣的繁體中文，並將結果輸出到新的 CSV 檔案。請先確定您已獲得
TAIDE 模型的授權並擁有 Hugging Face 存取權杖。

使用方式：
    python taide_translate.py --input input_posts.csv --output translated_posts.csv \
       --model taide/TAIDE-LX-7B-Chat --hf_token <YOUR_TOKEN> --source_col post \
       --target_col translated_post

參數：
    --input/-i      原始 CSV 檔案路徑。
    --output/-o     翻譯後輸出 CSV 檔案路徑。
    --model/-m      Hugging Face 模型名稱，預設為 "taide/TAIDE-LX-7B-Chat"。
    --hf_token/-t   Hugging Face 存取權杖，用於存取 gated 模型。
    --source_col/-s 原始英文貼文欄位名稱，預設為 "post"。
    --target_col/-d 翻譯後欄位名稱，預設為 "translated_post"。
    --no-quant      如果指定，則不使用 4bit 量化模型。
"""

import argparse
from typing import Tuple

import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    BitsAndBytesConfig,
)
import torch
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_taide_pipeline(model_name: str, hf_token: str, use_quantization: bool = True) -> Tuple[TextGenerationPipeline, AutoTokenizer]:
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
    output = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False)[0]["generated_text"]
    return output.strip()


def translate_csv(
    input_path: str,
    output_path: str,
    model_name: str,
    hf_token: str,
    source_col: str = "post",
    target_col: str = "translated_post",
    use_quantization: bool = True,
    system_prompt: str = "你是一位翻譯專家，擅長將英文社群媒體貼文翻譯成符合台灣口語習慣的繁體中文。請使用口語化的字詞，避免過於正式的語氣。",
    max_new_tokens: int = 256,
) -> None:
    """處理整個 CSV：逐筆翻譯並存成新的 CSV。"""
    pipe, tokenizer = load_taide_pipeline(model_name, hf_token, use_quantization)
    df = pd.read_csv(input_path)
    if source_col not in df.columns:
        raise ValueError(f"輸入 CSV 中找不到欄位 {source_col}")
    translations = []
    for content in tqdm(df[source_col].astype(str)):
        translation = translate_text(
            pipe,
            tokenizer,
            content,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        translations.append(translation)
    df[target_col] = translations
    df.to_csv(output_path, index=False)
    print(f"翻譯完成，輸出檔案已寫入 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Translate English social media posts into colloquial Taiwanese Mandarin using TAIDE.")
    parser.add_argument("--input", "-i", default="Data/DR_post.csv", help="Path to input CSV file containing English posts")
    parser.add_argument("--output", "-o", default="output.csv", help="Path to output CSV file")
    parser.add_argument(
        "--model",
        "-m",
        default="taide/Llama-3.1-TAIDE-LX-8B-Chat",
        help="Name of the TAIDE model to use (e.g., taide/TAIDE-LX-7B-Chat or taide/Llama3-TAIDE-LX-8B-Chat-Alpha1)",
    )
    parser.add_argument("--hf_token", "-t", default=os.getenv("HF_TOKEN"), help="Your Hugging Face access token for gated models")
    parser.add_argument("--source_col", "-s", default="post", help="Column name of English posts in the input CSV")
    parser.add_argument("--target_col", "-d", default="translated_post", help="Column name for translated posts in the output CSV")
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable 4bit quantization when loading the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate for each translation",
    )
    args = parser.parse_args()

    translate_csv(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        hf_token=args.hf_token,
        source_col=args.source_col,
        target_col=args.target_col,
        use_quantization=not args.no_quant,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()