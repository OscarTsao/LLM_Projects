#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAIDE DSM-5 Criteria Filtering — RAG vs Baseline (HF Hub, robust, audit-friendly)

What this script does
- Loads a TAIDE chat model from the Hugging Face Hub (no local repo path needed).
- Loads your DSM-5 criteria JSON shaped like:
  [
    {"diagnosis": "Major Depressive Disorder",
     "criteria": [{"id":"A","text":"..."}, {"id":"B","text":"..."}, ...]},
    ...
  ]
- Flattens to (disorder, criterion_id, text), cleans whitespace/line breaks.
- Builds a TF-IDF retriever; compares Baseline (no retrieval) vs RAG (stuff top-k criteria).
- Saves results (JSONL + CSV) with timings & retrieved snippets for human review.
- Optional schema inspection: --schema_debug (prints how many disorders/criteria detected).

Quick start
-----------
pip install -U transformers accelerate bitsandbytes scikit-learn torch

# inspect JSON only
python taide_dsm5_rag_hf.py --dsm5_json /path/to/DSM_Criteria_Array_Fixed.json --schema_debug

# single context
python taide_dsm5_rag_hf.py \
  --model_id taide/Llama-3.1-TAIDE-LX-8B-Chat \
  --dsm5_json /path/to/DSM_Criteria_Array_Fixed.json \
  --context "Patient reports persistent low mood, anhedonia, insomnia..." \
  --k 12 \
  --out_dir ./rag_results

# batch (CSV with a 'context' column)
python taide_dsm5_rag_hf.py \
  --model_id taide/Llama-3.1-TAIDE-LX-8B-Chat \
  --dsm5_json /path/to/DSM_Criteria_Array_Fixed.json \
  --contexts_csv /path/to/notes.csv \
  --context_col context \
  --k 12 \
  --out_dir ./rag_results
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


# ======================
# Model loader (HF Hub)
# ======================

def load_taide_hf_model(
    model_id: str,
    quantize_4bit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
):
    """
    Return a HF text-generation pipeline for a TAIDE model on the Hub.
    Example model_id:
      - "taide/Llama-3.1-TAIDE-LX-8B-Chat"
      - "taide/TAIDE-LX-7B"
    """
    bnb_config = None
    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
    )

    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.1,
        do_sample=False,
        max_new_tokens=512,
    )


# ======================
# DSM-5 JSON loader
# ======================

@dataclass
class Criterion:
    disorder: str
    criterion_id: str
    text: str

_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\s*\n\s*")

def _clean_text(s: str) -> str:
    # fix hyphenated line breaks: "su-\nicide" -> "suicide"
    s = re.sub(r"-\s*\n\s*", "", s)
    # collapse newlines & spaces
    s = _NL.sub("\n", s)
    s = _WS.sub(" ", s)
    return s.strip()

def load_dsm5_json(path: Union[str, Path]) -> List[Criterion]:
    """
    Loader tailored to files shaped like:
    [
      {"diagnosis": "X", "criteria": [{"id":"A","text":"..."}, {"id":"B","text":"..."}]},
      ...
    ]
    - Accepts duplicate ids per disorder (keeps them; retrieval doesn’t require uniqueness).
    - Cleans whitespace and broken line-hyphenations.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected top-level list of disorder objects in DSM-5 JSON.")

    out: List[Criterion] = []
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        disorder = obj.get("diagnosis") or obj.get("name") or obj.get("title")
        items = obj.get("criteria")
        if not disorder or not isinstance(items, list):
            continue

        disorder_name = str(disorder).strip()
        for idx, it in enumerate(items):
            if isinstance(it, dict):
                cid = (it.get("id") or f"C{idx+1}").strip()
                txt = str(it.get("text", "")).strip()
            elif isinstance(it, str):
                cid, txt = f"C{idx+1}", it
            else:
                continue

            txt = _clean_text(txt)
            if not txt:
                continue
            out.append(Criterion(disorder=disorder_name, criterion_id=cid, text=txt))

    if not out:
        raise ValueError("No criteria found; expected keys 'diagnosis' and 'criteria'.")
    return out

def schema_debug_report(path: Union[str, Path]) -> None:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    print("=== SCHEMA DEBUG ===")
    print(f"Top-level type: {type(raw).__name__}")
    if isinstance(raw, list):
        print(f"Top-level length: {len(raw)}")
        if raw and isinstance(raw[0], dict):
            print(f"First item keys: {list(raw[0].keys())}")
    else:
        print("Expected a list at top level for this loader.")
    # Count criteria quickly
    try:
        crits = load_dsm5_json(path)
        by_dis = {}
        for c in crits:
            by_dis.setdefault(c.disorder, 0)
            by_dis[c.disorder] += 1
        print(f"Total criteria parsed: {len(crits)}")
        for d, n in list(by_dis.items())[:10]:
            print(f"  - {d}: {n} criteria")
    except Exception as e:
        print(f"Parser error: {e}")
    print("=== END SCHEMA DEBUG ===")


# ======================
# TF-IDF retriever
# ======================

class TfidfRetriever:
    def __init__(self, criteria: List[Criterion]):
        self.criteria = criteria
        self.texts = [c.text for c in criteria]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_df=0.9, min_df=1, stop_words=None
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]   # dense ndarray
        sims = np.asarray(sims).squeeze()              # robust; no .toarray()
        idxs = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idxs]


# ======================
# Prompt builders
# ======================

BASELINE_SYS = (
    "You are a careful psychiatric assistant. "
    "Given a patient's context, identify DSM-5 diagnostic criteria that are met. "
    "Respond in JSON with a list 'matches', where each item is: "
    "{'disorder': str, 'criterion_id': str, 'evidence': str, 'confidence': float}. "
    "If unsure, return an empty list."
)

RAG_SYS = (
    "You are a careful psychiatric assistant. "
    "Use ONLY the provided DSM-5 criteria context to decide which criteria are met. "
    "Return JSON with 'matches' as a list of objects: "
    "{'disorder': str, 'criterion_id': str, 'evidence': str, 'confidence': float}. "
    "Evidence must be a short quote or paraphrase drawn from the patient context (not copied from criteria text). "
    "If none match, return an empty list."
)

def build_baseline_prompt(context: str) -> str:
    return (
        f"<|system|>\n{BASELINE_SYS}\n</|system|>\n"
        f"<|user|>\nPatient context:\n{context}\n</|user|>\n"
        f"<|assistant|>\nReturn ONLY valid JSON.\n"
    )

def build_rag_prompt(context: str, retrieved: List[Criterion]) -> str:
    crit_str = "\n".join([f"- [{c.disorder}::{c.criterion_id}] {c.text}" for c in retrieved])
    return (
        f"<|system|>\n{RAG_SYS}\n</|system|>\n"
        f"<|user|>\nPatient context:\n{context}\n\n"
        f"Relevant DSM-5 criteria (only select from these):\n{crit_str}\n"
        f"</|user|>\n<|assistant|>\nReturn ONLY valid JSON.\n"
    )


# ======================
# LLM helpers
# ======================

JSON_REGEX = re.compile(r"\{[\s\S]*\}$", re.MULTILINE)

def run_pipe_text(llm_pipe, prompt: str, max_new_tokens: int = 512) -> str:
    out = llm_pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
    m = JSON_REGEX.search(out.strip())
    return m.group(0) if m else out

def safe_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


# ======================
# Orchestration
# ======================

def process_single(
    llm_pipe,
    retriever: TfidfRetriever,
    all_criteria: List[Criterion],
    context: str,
    k: int = 12,
) -> dict:
    # Baseline
    t0 = time.time()
    baseline_text = run_pipe_text(llm_pipe, build_baseline_prompt(context))
    baseline_obj = safe_parse_json(baseline_text)
    t1 = time.time()

    # RAG
    hits = retriever.search(context, k=k)
    retrieved = [all_criteria[i] for i, _ in hits]
    rag_text = run_pipe_text(llm_pipe, build_rag_prompt(context, retrieved))
    rag_obj = safe_parse_json(rag_text)
    t2 = time.time()

    return {
        "context": context,
        "baseline": baseline_obj,
        "rag": rag_obj,
        "timing": {
            "baseline_sec": round(t1 - t0, 4),
            "retrieval_sec": 0.0,  # retrieval is milliseconds; omit or keep as 0.0
            "rag_gen_sec": round(t2 - t1, 4),
            "total_sec": round(t2 - t0, 4),
        },
        "retrieved": [
            {
                "disorder": all_criteria[i].disorder,
                "criterion_id": all_criteria[i].criterion_id,
                "text": all_criteria[i].text,
                "score": float(score),
            }
            for (i, score) in hits
        ],
    }

def write_results(out_dir: Path, rows: List[dict]) -> Tuple[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / "results.jsonl"
    with jl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    csvp = out_dir / "results.csv"
    with csvp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "index", "baseline_time_s", "rag_time_s", "total_time_s",
            "baseline_json_or_raw", "rag_json_or_raw"
        ])
        for i, r in enumerate(rows):
            w.writerow([
                i,
                r["timing"]["baseline_sec"],
                r["timing"]["rag_gen_sec"],
                r["timing"]["total_sec"],
                json.dumps(r["baseline"], ensure_ascii=False)[:1000],
                json.dumps(r["rag"], ensure_ascii=False)[:1000],
            ])
    return str(jl), str(csvp)


# ======================
# CLI
# ======================

def read_contexts_from_csv(csv_path: str, col: str) -> List[str]:
    vals = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals.append(row[col])
    return vals

def main():
    ap = argparse.ArgumentParser(description="TAIDE DSM-5 RAG vs Baseline (HF Hub)")
    ap.add_argument("--model_id", default="taide/Llama-3.1-TAIDE-LX-8B-Chat", help="HF model id, e.g. taide/Llama-3.1-TAIDE-LX-8B-Chat")
    ap.add_argument("--dsm5_json", default="DSM-5/DSM_Criteria_Array_Fixed.json", help="Path to DSM_Criteria_Array_Fixed.json")
    ap.add_argument("--context", help="Single context text")
    ap.add_argument("--contexts_csv", default="output.csv", help="CSV with a column of contexts")
    ap.add_argument("--context_col", default="translated_post", help="Column name for contexts in CSV")
    ap.add_argument("--k", type=int, default=12, help="Top-k criteria to retrieve for RAG")
    ap.add_argument("--out_dir", default="./rag_results", help="Output directory")
    ap.add_argument("--no-4bit", dest="no_4bit", action="store_true", help="Disable 4-bit quantization")
    ap.add_argument("--schema_debug", action="store_true", help="Print schema info and exit")
    args = ap.parse_args()

    # Inspect JSON only (no model required)
    if args.schema_debug:
        schema_debug_report(args.dsm5_json)
        return

    # Load DSM-5 criteria
    criteria = load_dsm5_json(args.dsm5_json)
    if not criteria:
        schema_debug_report(args.dsm5_json)
        raise ValueError("No criteria parsed. See schema debug.")

    # Prepare contexts
    contexts: List[str] = []
    if args.context:
        contexts.append(args.context)
    if args.contexts_csv:
        contexts.extend(read_contexts_from_csv(args.contexts_csv, args.context_col))
    if not contexts:
        raise SystemExit("No context given. Use --context or --contexts_csv, or run --schema_debug first.")

    # Load model
    if not args.model_id:
        raise SystemExit("Provide --model_id (e.g. taide/Llama-3.1-TAIDE-LX-8B-Chat).")
    llm_pipe = load_taide_hf_model(
        model_id=args.model_id,
        quantize_4bit=(not args.no_4bit),
        dtype=torch.bfloat16
    )

    # Retriever
    retriever = TfidfRetriever(criteria)

    # Run
    rows = [process_single(llm_pipe, retriever, criteria, ctx, k=args.k)
            for ctx in contexts]

    # Save
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / ts
    jl_path, csv_path = write_results(out_dir, rows)
    print(f"[✓] Saved JSONL to: {jl_path}")
    print(f"[✓] Saved CSV   to: {csv_path}")


if __name__ == "__main__":
    main()
