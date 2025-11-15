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
from typing import Tuple, Optional, List, Dict

import pandas as pd
import re
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


# Regular expressions for protecting special tokens (URLs, mentions, hashtags, inline code)
_URL_RE = re.compile(r"https?://\S+")
_TAG_RE = re.compile(r"(?:^|[\s])#\w+")
_MEN_RE = re.compile(r"(?:^|[\s])@\w+")
_CODE_RE = re.compile(r"`[^`]+`")


def _protect(text: str) -> Tuple[str, dict]:
    """
    Replace special patterns in the input text with placeholders to prevent the model from
    modifying them during translation. Returns the protected text and a mapping of
    placeholders back to the original segments.

    Args:
        text: Original text to protect.

    Returns:
        A tuple of (protected_text, placeholder_mapping).
    """
    maps = {}
    idx = 0

    def _repl(regex: re.Pattern, prefix: str, s: str) -> str:
        nonlocal idx

        def _inner(m: re.Match) -> str:
            nonlocal idx
            key = f"__{prefix}{idx}__"
            maps[key] = m.group(0)
            idx += 1
            return key

        return regex.sub(_inner, s)

    s = text
    for regex, prefix in [(_CODE_RE, "CODE"), (_URL_RE, "URL"), (_MEN_RE, "MEN"), (_TAG_RE, "TAG")]:
        s = _repl(regex, prefix, s)
    return s, maps


def _restore(text: str, maps: dict) -> str:
    """
    Restore placeholders back to their original strings after translation.

    Args:
        text: Text containing placeholders.
        maps: Mapping from placeholders to original strings.

    Returns:
        Restored text.
    """
    for k, v in maps.items():
        text = text.replace(k, v)
    return text


def _default_system_prompt() -> str:
    """
    Return a verbose system prompt that enforces faithful translation, preserving all
    important tokens and formatting while producing natural Taiwanese colloquial
    Mandarin. The prompt emphasises completeness and forbids deletion or
    addition of content.
    """
    return (
        "你是一位翻譯專家，將英文社群貼文完整翻成符合台灣口語習慣的繁體中文。\n"
        "嚴格規則：\n"
        "1) 不得刪減或新增資訊；逐句對應，不要合併句子。\n"
        "2) 保留所有數字/單位/日期/時間/網址/@mention/#hashtag/emoji/括號與引號結構；\n"
        "3) 專有名詞/品牌/帳號名以原文保留；必要時可在後方補中文註釋。\n"
        "4) 風格自然台灣口語、避免過度書面或中國用語；\n"
        "5) 不確定的縮寫請保留原文並加上中文括號輕註；\n"
        "6) 僅輸出翻譯，不要加說明。"
    )


def translate_text(
    pipe: TextGenerationPipeline,
    tokenizer: AutoTokenizer,
    text: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    stop_ids: Optional[List[int]] = None,
) -> str:
    """
    Translate a single English sentence or paragraph into colloquial Taiwanese Mandarin
    using the TAIDE model. This function protects special tokens (URLs, hashtags,
    mentions, code blocks) before translation and restores them afterwards. It also
    dynamically adjusts the number of tokens to generate based on the input length
    unless the user provides a fixed `max_new_tokens`.

    Args:
        pipe: A Hugging Face TextGenerationPipeline configured for translation.
        tokenizer: The corresponding AutoTokenizer.
        text: The English text to translate.
        system_prompt: Optional system prompt. If None, a default prompt is used.
        max_new_tokens: Maximum number of tokens to generate. If None, this is
            computed dynamically based on the length of `text`.
        temperature: Sampling temperature for generation (lower for more
            deterministic output).
        top_p: Top-p sampling value.
        repetition_penalty: Penalty for repeated tokens.
        stop_ids: Optional list of EOS token IDs to terminate generation early.

    Returns:
        A translated string with special tokens restored.
    """
    # Use the default system prompt if none is provided
    if system_prompt is None:
        system_prompt = _default_system_prompt()

    # Protect URLs, mentions, hashtags and code blocks
    protected_text, maps = _protect(text)

    # Build chat prompt
    chat = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "把以下英文逐句對應翻成台灣口語繁中，完整保留所有結構與資訊：\n"
                f"{protected_text}"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)

    # Determine token limit: scale with input length if not provided
    if max_new_tokens is None:
        # Base on the number of characters: overshoot by 40% plus some buffer.
        # Limit to 1024 new tokens to avoid excessively long generations.
        base = max(128, int(len(protected_text) * 1.4) + 32)
        max_new_tokens = min(base, 1024)

    # Generation arguments
    gen_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "return_full_text": False,
    }
    if stop_ids is not None:
        gen_args["eos_token_id"] = stop_ids

    # Generate translation
    output = pipe(prompt, **gen_args)[0]["generated_text"].strip()

    # Restore placeholders to original
    restored = _restore(output, maps)
    return restored

    
#--- Evaluation Blocks ---#


def _extract_numbers(text: str) -> list[str]:
    """Extract all numeric-like substrings (including decimals) from text."""
    return re.findall(r"\d+(?:\.\d+)?", text)


def _paren_counts(text: str) -> dict:
    """Count various types of parentheses in text."""
    counts = {}
    for p in ["(", ")", "[", "]", "{", "}"]:
        counts[p] = text.count(p)
    return counts


def _alpha_ratio(text: str) -> float:
    """Return the proportion of A-Za-z letters in text."""
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha() and c.lower() >= 'a' and c.lower() <= 'z')
    return alpha / len(text)


def _evaluate_translation(src: str, tgt: str, len_threshold: float = 0.8, alpha_threshold: float = 0.15) -> dict:
    """
    Evaluate the quality of a translation based on several heuristics.

    Args:
        src: Source English text.
        tgt: Target translated text.
        len_threshold: Minimum acceptable ratio of target length to source length.
        alpha_threshold: Maximum acceptable proportion of English letters in target.

    Returns:
        A dictionary with boolean flags for each check and computed metrics.
    """
    # Compute length ratio
    len_ratio = (len(tgt) + 1e-6) / (len(src) + 1e-6)
    flag_len_short = len_ratio < len_threshold

    # Compare numbers
    src_numbers = _extract_numbers(src)
    tgt_numbers = _extract_numbers(tgt)
    flag_number_mismatch = len(src_numbers) != len(tgt_numbers)

    # Compare parentheses counts
    src_paren = _paren_counts(src)
    tgt_paren = _paren_counts(tgt)
    flag_paren_mismatch = any(src_paren[p] != tgt_paren[p] for p in src_paren)

    # English alphabet proportion
    tgt_alpha_pct = _alpha_ratio(tgt)
    flag_alpha_high = tgt_alpha_pct > alpha_threshold

    return {
        "len_ratio": len_ratio,
        "flag_len_short": flag_len_short,
        "flag_number_mismatch": flag_number_mismatch,
        "flag_paren_mismatch": flag_paren_mismatch,
        "tgt_alpha_pct": tgt_alpha_pct,
        "flag_alpha_high": flag_alpha_high,
    }

#--- Main Translation Function ---#

def translate_csv(
    input_path: str,
    output_path: str,
    model_name: str,
    hf_token: str,
    source_col: str = "post",
    target_col: str = "translated_post",
    use_quantization: bool = True,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    evaluate: bool = True,
    len_threshold: float = 0.8,
    alpha_threshold: float = 0.15,
    run_second_pass: bool = True,
) -> None:
    """
    Translate an entire CSV file column by column using the TAIDE model. This
    function reads an input CSV, translates each entry in the specified source
    column, optionally evaluates the translation for quality issues, and writes
    the results to a new CSV.

    Args:
        input_path: Path to the input CSV containing English posts.
        output_path: Path where the translated CSV will be saved.
        model_name: Hugging Face model name to load.
        hf_token: Hugging Face access token.
        source_col: Name of the column containing English posts.
        target_col: Name of the column where translated posts will be stored.
        use_quantization: Whether to enable 4bit quantization when loading the model.
        system_prompt: Optional custom system prompt. If None, a default prompt
            enforcing completeness and colloquial Taiwanese will be used.
        max_new_tokens: Maximum tokens for generation. If None, dynamic
            length will be used per entry.
        temperature, top_p, repetition_penalty: Generation parameters.
        evaluate: Whether to run heuristic QA checks on each translation.
        len_threshold: Threshold for minimum acceptable length ratio.
        alpha_threshold: Threshold for maximum acceptable English letter proportion.
        run_second_pass: If True and evaluate is True, translations that
            trigger flags will be re-translated using a corrective prompt.
    """
    # Load the model and tokenizer
    pipe, tokenizer = load_taide_pipeline(model_name, hf_token, use_quantization)

    # Read input CSV
    df = pd.read_csv(input_path)
    if source_col not in df.columns:
        raise ValueError(f"輸入 CSV 中找不到欄位 {source_col}")

    # Prepare output list and evaluation columns
    translations = []
    eval_records = []

    # Iterate through each post and translate
    for content in tqdm(df[source_col].astype(str), desc="Translating"):
        # First translation
        first_translation = translate_text(
            pipe,
            tokenizer,
            content,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Evaluate translation quality
        record = None
        if evaluate:
            record = _evaluate_translation(
                content, first_translation, len_threshold=len_threshold, alpha_threshold=alpha_threshold
            )

            # If flags indicate issues and second pass is enabled
            if run_second_pass and (
                record["flag_len_short"]
                or record["flag_number_mismatch"]
                or record["flag_paren_mismatch"]
                or record["flag_alpha_high"]
            ):
                # Construct corrective system prompt (use the default if none provided)
                corrective_prompt = (
                    system_prompt if system_prompt is not None else _default_system_prompt()
                )
                # Build user message instructing to fix missing parts
                user_msg = (
                    "上面這段中文翻譯疑似有遺漏或改寫，請逐句比對原文補齊，保持所有數字與括號完全一致，不要刪減任何句子。\n"
                    f"原文：{content}\n"
                    f"第一次翻譯：{first_translation}\n"
                    "請輸出修正版。"
                )
                chat = [
                    {"role": "system", "content": corrective_prompt},
                    {"role": "user", "content": user_msg},
                ]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False)
                # Use a generous token limit for corrective generation
                second_max_tokens = max_new_tokens if max_new_tokens is not None else None
                second_translation = pipe(
                    prompt,
                    max_new_tokens=second_max_tokens or 1024,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    return_full_text=False,
                )[0]["generated_text"].strip()
                # Use second translation as final result
                translation = second_translation
            else:
                translation = first_translation
        else:
            translation = first_translation

        translations.append(translation)
        # Append evaluation record if available
        if record is not None:
            eval_records.append(record)

    # Attach translations to DataFrame
    df[target_col] = translations

    # Optionally write evaluation metrics to extra columns
    if evaluate and eval_records:
        # Flatten evaluation dict into DataFrame columns
        eval_df = pd.DataFrame(eval_records)
        for col in eval_df.columns:
            df[f"eval_{col}"] = eval_df[col]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"翻譯完成，輸出檔案已寫入 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Translate English social media posts into colloquial Taiwanese Mandarin using TAIDE.")
    parser.add_argument("--input", "-i", default="Data/mental_health_posts.csv", help="Path to input CSV file containing English posts")
    parser.add_argument("--output", "-o", default="output.csv", help="Path to output CSV file")
    parser.add_argument(
        "--model",
        "-m",
        default="taide/Llama-3.1-TAIDE-LX-8B-Chat",
        help="Name of the TAIDE model to use (e.g., taide/TAIDE-LX-7B-Chat or taide/Llama3-TAIDE-LX-8B-Chat-Alpha1)",
    )
    parser.add_argument("--hf_token", "-t", default=os.getenv("HF_TOKEN"), help="Your Hugging Face access token for gated models")
    parser.add_argument("--source_col", "-s", default="post_context", help="Column name of English posts in the input CSV")
    parser.add_argument("--target_col", "-d", default="translated_post", help="Column name for translated posts in the output CSV")
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable 4bit quantization when loading the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of new tokens to generate for each translation. "
            "If not provided, the script will dynamically determine an appropriate length."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for translation generation",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Penalty for repeated tokens during generation",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help=(
            "Custom system prompt to steer translations. If omitted, a default prompt "
            "enforcing completeness and colloquial Taiwanese will be used."
        ),
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable heuristic QA evaluation and second pass correction",
    )
    parser.add_argument(
        "--len_threshold",
        type=float,
        default=0.8,
        help="Minimum acceptable length ratio (target/source) before triggering a second pass",
    )
    parser.add_argument(
        "--alpha_threshold",
        type=float,
        default=0.15,
        help="Maximum acceptable proportion of English letters in the translation",
    )
    parser.add_argument(
        "--no-second-pass",
        action="store_true",
        help="Disable the second pass correction even if evaluation flags issues",
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
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        evaluate=not args.no_eval,
        len_threshold=args.len_threshold,
        alpha_threshold=args.alpha_threshold,
        run_second_pass=not args.no_second_pass,
    )


if __name__ == "__main__":
    main()