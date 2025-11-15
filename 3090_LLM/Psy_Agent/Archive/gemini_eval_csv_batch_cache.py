#!/usr/bin/env python3
"""
gemini_eval_csv_batch_cache.py

這個腳本展示如何同時利用 Gemini API 的批次模式（Batch Mode）以及顯式上下文快取
（Context Caching）來評估大量翻譯請求。批次模式在處理大量非即時任務時成本僅為
即時模式的一半【714182217970536†L266-L270】【790941564634588†L760-L770】；上下文快取
允許將固定的提示內容預先快取起來，多次引用可降低重複 Token 的成本【667530173770751†L264-L286】。

用途：
 讀取含有 `post`（英文原文）與 `translated_post`（繁體翻譯）的 CSV，對每行使用
 Gemini 模型產生評分結果（accuracy, similarity, verdict, comment），並將結果寫入
 第三欄 `gemini_eval`。當開啟 `--batch` 選項時，程式會組合所有請求為單一
 Batch Job 送出，大幅降低成本；當開啟 `--cache` 選項時，會建立一個上下文
 快取，將評估的系統提示與說明存入快取並於每次呼叫時引用，以減少重複 token。

注意事項：
 * 批次模式為非即時：需輪詢直到批次任務完成後才能取得結果，處理時間可能長達
   24 小時，但實務上通常更快【790941564634588†L760-L770】。
 * 缓存功能需要版本後綴 `-001` 的模型名稱。例如 `gemini-2.5-flash-001`。若傳入
   的模型名不含版本後綴但開啟了 `--cache`，腳本會自動加上 `-001`。若指定的
   型號不支援快取，則建立快取會失敗。
 * 由於 SDK 目前對批次模式的結構化輸出支援有限，本腳本於解析批次結果時直接
   讀取文本 JSON 並以 `json.loads` 解析。

用法範例：

    export GEMINI_API_KEY="YOUR_API_KEY"
    python gemini_eval_csv_batch_cache.py -i output.csv -o output_batch.csv \
      --model gemini-2.5-flash --batch --cache --cache_ttl 3600

"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd
from google import genai
from pydantic import BaseModel, Field


# ------------------- Schema 定義 -------------------
class EvalResult(BaseModel):
    """定義結構化輸出的模式，用於解析 JSON 字串。"""
    accuracy: Optional[int]
    similarity: Optional[float]
    verdict: Optional[str]
    comment: Optional[str]


# ------------------- 常數與模板 -------------------
# 評估用的系統提示與用戶提示。這些會被存入上下文快取以減少重複 token。
SYSTEM_PROMPT = (
    "You are a bilingual evaluator. Assess the quality of English to Traditional Chinese (Taiwan) translations for social media posts."
)

USER_INSTRUCTIONS = (
    "Return a JSON object with these keys: accuracy (0-100 integer, faithfulness), "
    "similarity (0.0-1.0 float, semantic similarity), verdict ('poor','fair','good','excellent'), "
    "and comment (brief Traditional Chinese explanation, ≤60 chars)."
)

def build_client(api_key: Optional[str] = None) -> genai.Client:
    """初始化 Gemini Client。若未提供 api_key，將讀取環境變數。"""
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def create_cache(
    client: genai.Client, model: str, display_name: str, ttl: str = "3600s"
) -> str:
    """建立上下文快取，將系統提示與評估說明存入，返回快取名稱。"""
    # 使用帶版本的模型名稱，例如 gemini-2.5-flash-001；如果沒有版本後綴，自動加上 -001
    versioned_model = model if model.endswith("-001") else f"{model}-001"
    from google.genai import types

    # cache contents 包含系統提示與指令，這些將作為上下文前綴
    contents = [SYSTEM_PROMPT, USER_INSTRUCTIONS]

    cache = client.caches.create(
        model=versioned_model,
        config=types.CreateCachedContentConfig(
            display_name=display_name,
            system_instruction=SYSTEM_PROMPT,
            contents=contents,
            ttl=ttl,
        ),
    )
    return cache.name


def delete_cache(client: genai.Client, cache_name: str) -> None:
    """刪除快取以釋放儲存資源。"""
    try:
        client.caches.delete(name=cache_name)
    except Exception:
        pass


def prepare_inline_requests(
    rows: List[Dict[str, str]], cache_name: Optional[str] = None
) -> List[Dict]:
    """將每一行翻譯資料轉換為批次模式的 InlinedRequest 格式。"""
    requests: List[Dict] = []
    for idx, row in enumerate(rows):
        source = str(row["post"]).strip() if pd.notna(row["post"]) else ""
        translation = str(row["translated_post"]).strip() if pd.notna(row["translated_post"]) else ""
        # 若有任何欄位為空，加入佔位結果
        if not source or not translation:
            # 我們仍然傳送請求讓模型處理空內容，稍後解析時會標記為 skip
            user_text = "Source: \nTranslation: "
        else:
            user_text = f"Source (EN): {source}\nTranslation (ZH-TW): {translation}\n"
        # 組合請求物件
        req: Dict[str, any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_text},
                        {"text": USER_INSTRUCTIONS},
                    ],
                }
            ],
        }
        # 若有快取，加入 generation_config
        if cache_name:
            req["generation_config"] = {
                "cached_content": cache_name,
                "response_mime_type": "application/json",
                # batch 暫不支援 response_schema 直接解析，待解析時用 json.loads
            }
        else:
            req["generation_config"] = {
                "response_mime_type": "application/json",
            }
        requests.append(req)
    return requests


def parse_batch_inline_responses(
    inline_responses: List[Dict], num_rows: int
) -> List[str]:
    """解析批次模式的 inlineResponses，回傳 JSON 字串列表長度與資料列相同。"""
    results: List[str] = [""] * num_rows
    for idx, resp in enumerate(inline_responses):
        # inline response 可能有 error 或 response
        if resp.get("error"):
            error = resp["error"]
            results[idx] = json.dumps(
                {
                    "accuracy": None,
                    "similarity": None,
                    "verdict": "error",
                    "comment": f"Batch error: {error.get('message', 'unknown')}",
                },
                ensure_ascii=False,
            )
        else:
            response = resp.get("response")
            if not response:
                results[idx] = json.dumps(
                    {
                        "accuracy": None,
                        "similarity": None,
                        "verdict": None,
                        "comment": "no response",
                    },
                    ensure_ascii=False,
                )
            else:
                # 嘗試解析 response.text 為 JSON
                try:
                    # response.text 可能包含模型輸出的全部文本
                    text = response.get("text") or response.get("candidates", [{}])[0].get(
                        "content", {}
                    ).get("parts", [{}])[0].get("text", "")
                    text = text.strip()
                    if text:
                        parsed = json.loads(text)
                        # 確保缺失欄位補上 None
                        result_dict = {
                            "accuracy": parsed.get("accuracy"),
                            "similarity": parsed.get("similarity"),
                            "verdict": parsed.get("verdict"),
                            "comment": parsed.get("comment"),
                        }
                        results[idx] = json.dumps(result_dict, ensure_ascii=False)
                    else:
                        results[idx] = json.dumps(
                            {
                                "accuracy": None,
                                "similarity": None,
                                "verdict": None,
                                "comment": "empty output",
                            },
                            ensure_ascii=False,
                        )
                except Exception as e:
                    # 解析失敗
                    results[idx] = json.dumps(
                        {
                            "accuracy": None,
                            "similarity": None,
                            "verdict": "error",
                            "comment": f"parse error: {e.__class__.__name__}",
                        },
                        ensure_ascii=False,
                    )
    return results


def batch_evaluate(
    client: genai.Client,
    model: str,
    rows: List[Dict[str, str]],
    use_cache: bool = False,
    cache_ttl: str = "3600s",
    display_name: str = "batch-eval-job",
) -> List[str]:
    """使用批次模式對所有資料列進行評分並返回 JSON 字串列表。"""
    cache_name: Optional[str] = None
    if use_cache:
        cache_name = create_cache(
            client, model=model, display_name=f"eval-cache-{display_name}", ttl=cache_ttl
        )
    try:
        # 準備批次請求
        inline_requests = prepare_inline_requests(rows, cache_name=cache_name)
        # 提交批次工作
        batch_job = client.batches.create(
            model=model,
            src=inline_requests,
            config={"display_name": display_name},
        )
        # 等待工作完成
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_PAUSED",
            "JOB_STATE_EXPIRED",
        }
        job = batch_job
        while job.state not in completed_states:
            print(f"Batch job state: {job.state}")
            time.sleep(30)
            job = client.batches.get(name=batch_job.name)
        if job.state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Batch job did not succeed: {job.state}")
        # 解析 inline responses
        inline_responses = job.get("inline_responses") or job.get(
            "inlined_responses"
        )
        if inline_responses is None:
            # 根據 SDK 實際返回格式調整
            inline_responses = getattr(job, "inline_responses", None)
        if inline_responses is None:
            # 若沒有返回 inline responses，拋出例外
            raise RuntimeError("No inline responses found in batch job result.")
        results = parse_batch_inline_responses(inline_responses, len(rows))
        return results
    finally:
        if cache_name:
            # 批次完成後刪除快取
            delete_cache(client, cache_name)


def interactive_evaluate_with_cache(
    client: genai.Client,
    model: str,
    rows: List[Dict[str, str]],
    cache_ttl: str = "3600s",
) -> List[str]:
    """使用即時模式結合上下文快取對每行進行評分，返回 JSON 字串列表。"""
    # 建立快取並取得名稱
    cache_name = create_cache(
        client, model=model, display_name="eval-cache-interactive", ttl=cache_ttl
    )
    results: List[str] = []
    try:
        for row in rows:
            source = str(row["post"]).strip() if pd.notna(row["post"]) else ""
            translation = str(row["translated_post"]).strip() if pd.notna(row["translated_post"]) else ""
            if not source or not translation:
                results.append(
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
            prompt = f"Source (EN): {source}\nTranslation (ZH-TW): {translation}\n"
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "cached_content": cache_name,
                        "response_mime_type": "application/json",
                    },
                )
                text = (resp.text or "").strip()
                if text:
                    try:
                        parsed = json.loads(text)
                        result_dict = {
                            "accuracy": parsed.get("accuracy"),
                            "similarity": parsed.get("similarity"),
                            "verdict": parsed.get("verdict"),
                            "comment": parsed.get("comment"),
                        }
                        results.append(json.dumps(result_dict, ensure_ascii=False))
                    except Exception as e:
                        results.append(
                            json.dumps(
                                {
                                    "accuracy": None,
                                    "similarity": None,
                                    "verdict": "error",
                                    "comment": f"parse error: {e.__class__.__name__}",
                                },
                                ensure_ascii=False,
                            )
                        )
                else:
                    results.append(
                        json.dumps(
                            {
                                "accuracy": None,
                                "similarity": None,
                                "verdict": None,
                                "comment": "empty output",
                            },
                            ensure_ascii=False,
                        )
                    )
            except Exception as exc:
                results.append(
                    json.dumps(
                        {
                            "accuracy": None,
                            "similarity": None,
                            "verdict": "error",
                            "comment": f"Gemini error: {exc.__class__.__name__}",
                        },
                        ensure_ascii=False,
                    )
                )
    finally:
        delete_cache(client, cache_name)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate translations using Gemini batch mode and/or context caching."
    )
    parser.add_argument("--input", "-i", default="output.csv", help="Input CSV file path")
    parser.add_argument(
        "--output", "-o", default=None, help="Output CSV path. Default overwrites input."
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gemini-2.5-flash",
        help="Gemini model name. For caching, a '-001' suffix will be appended automatically.",
    )
    parser.add_argument(
        "--api_key",
        "-k",
        default=None,
        help="Optional API key. If omitted, reads from GEMINI_API_KEY or GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch mode (process all rows in a single asynchronous batch job).",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use explicit context caching to reduce repeated token costs.",
    )
    parser.add_argument(
        "--cache_ttl",
        default="3600s",
        help="TTL for the context cache (e.g., '900s' for 15 minutes, '3600s' for 1 hour).",
    )
    parser.add_argument(
        "--display_name",
        default="translation-eval-batch",
        help="Display name for the batch job (for identification in AI Studio).",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path
    df = pd.read_csv(input_path)
    if not {"post", "translated_post"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'post' and 'translated_post' columns.")

    client = build_client(api_key=args.api_key)

    rows = df[["post", "translated_post"]].to_dict(orient="records")
    if args.batch:
        # 使用批次模式
        print("Submitting batch job (this may take a while to complete)...")
        results = batch_evaluate(
            client,
            model=args.model,
            rows=rows,
            use_cache=args.cache,
            cache_ttl=args.cache_ttl,
            display_name=args.display_name,
        )
    elif args.cache:
        # 使用即時模式 + 上下文快取
        results = interactive_evaluate_with_cache(
            client, model=args.model, rows=rows, cache_ttl=args.cache_ttl
        )
    else:
        # 若未指定 batch 或 cache，回退到普通即時模式（不快取）；此處直接調用 interactive 評估
        # reusing interactive evaluation without cache for each row
        results = []
        for row in rows:
            src = str(row["post"]).strip() if pd.notna(row["post"]) else ""
            tgt = str(row["translated_post"]).strip() if pd.notna(row["translated_post"]) else ""
            if not src or not tgt:
                results.append(
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
            prompt = f"Source (EN): {src}\nTranslation (ZH-TW): {tgt}\n{USER_INSTRUCTIONS}"
            try:
                resp = client.models.generate_content(
                    model=args.model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                text = (resp.text or "").strip()
                if text:
                    try:
                        parsed = json.loads(text)
                        res_dict = {
                            "accuracy": parsed.get("accuracy"),
                            "similarity": parsed.get("similarity"),
                            "verdict": parsed.get("verdict"),
                            "comment": parsed.get("comment"),
                        }
                        results.append(json.dumps(res_dict, ensure_ascii=False))
                    except Exception as e:
                        results.append(
                            json.dumps(
                                {
                                    "accuracy": None,
                                    "similarity": None,
                                    "verdict": "error",
                                    "comment": f"parse error: {e.__class__.__name__}",
                                },
                                ensure_ascii=False,
                            )
                        )
                else:
                    results.append(
                        json.dumps(
                            {
                                "accuracy": None,
                                "similarity": None,
                                "verdict": None,
                                "comment": "empty output",
                            },
                            ensure_ascii=False,
                        )
                    )
            except Exception as exc:
                results.append(
                    json.dumps(
                        {
                            "accuracy": None,
                            "similarity": None,
                            "verdict": "error",
                            "comment": f"Gemini error: {exc.__class__.__name__}",
                        },
                        ensure_ascii=False,
                    )
                )

    df["gemini_eval"] = results
    df.to_csv(output_path, index=False)
    print(f"Done. Wrote results to {output_path}")


if __name__ == "__main__":
    main()