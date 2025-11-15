"""
taide_dsm5_rag.py
===================

This module demonstrates how to build a simple Retrieval‑Augmented Generation
(RAG) pipeline around a locally deployed TAIDE large language model (LLM) using
PyTorch and HuggingFace’s `transformers` library.  The goal of this script is
to filter and rank DSM‑5 diagnostic criteria based on an input clinical
description (the **context**) and compare the model’s behaviour with and
without retrieval.  In the RAG configuration the script first searches the
DSM‑5 criteria for those most relevant to the input description using a
bag‑of‑words TF‑IDF similarity measure and then augments the prompt to the
model with those retrieved criteria.  For the baseline configuration the
model is asked directly about the context without additional information.  The
execution time for each method is recorded so users can evaluate the trade‑off
between retrieval overhead and model reasoning.

The DSM‑5 data is expected to be supplied as a JSON file.  Its structure can
vary (for example nested dictionaries or lists), so the loader flattens all
string values under each top‑level disorder into individual criteria entries.
Each criterion is stored along with its originating disorder name and a key
indicating where it was found in the JSON hierarchy.  When computing TF‑IDF
scores the script treats each criterion as a separate document.

The RAG approach is beneficial because it explicitly grounds the model in
external knowledge rather than relying solely on its parameters.  As
Hugging Face’s cookbook notes, RAG “works by providing an LLM with additional
context that is retrieved from relevant data so that it can generate a
better‑informed response”【436455639175163†L122-L139】.  The external data is
encoded into embeddings (here represented by TF‑IDF vectors) and stored in a
searchable index.  At query time the top matching entries are retrieved and
passed to the generative model as part of the prompt.  This avoids the need
for expensive fine‑tuning and allows the same LLM to be used across multiple
datasets【436455639175163†L122-L139】.

Dependencies
------------
To run this script you need the following Python packages:

* `torch` – for loading and running the TAIDE model.
* `transformers` – to load the tokenizer and model weights.
* `scikit-learn` – provides the `TfidfVectorizer` used for simple document
  embeddings and cosine similarity.
* (Optional) `pandas` – for exporting results as CSV.  If pandas is not
  installed the script will still run but will not write CSV files.

You can install these via pip:

```
pip install torch transformers scikit-learn pandas
```

Usage
-----
This script can be run as a standalone program.  Pass the path to your
DSM‑5 JSON file, the local HuggingFace model directory and optionally a list
of contexts separated by the `--context` argument.  If no contexts are
specified, a few example descriptions will be used.  For example:

```
python taide_dsm5_rag.py --dsm_path dsm5.json --model_path ./Llama3-TAIDE-LX-8B-Chat --context "He feels sad and hopeless for months" "She has difficulty paying attention"
```

The script prints the average execution time for the baseline and RAG
configurations and writes the detailed outputs into JSON files for later
inspection.  If pandas is available, a combined CSV file will also be
produced for easier review in spreadsheet tools.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import pandas as pd  # Optional; used for exporting CSV
except ImportError:
    pd = None

# Only import transformers when needed to avoid heavy import time if the user
# solely wants to use the retrieval component.  The try/except ensures the
# script fails gracefully if transformers is not installed.
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


def load_dsm5_criteria(file_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Load DSM‑5 disorders and criteria from a JSON file.

    The loader flattens all string values beneath each top‑level key (assumed
    to represent a disorder) into individual criterion entries.  Each entry
    consists of the text itself and associated metadata: the disorder name and
    the JSON key path where the text was found.  Non‑string values (e.g. lists
    or nested dictionaries) are traversed recursively.

    Parameters
    ----------
    file_path : str
        Path to the DSM‑5 JSON file.

    Returns
    -------
    criteria_texts : list of str
        A list of flattened criterion texts.

    metadata : list of dict
        A list of metadata dictionaries corresponding to each criterion.  Each
        dictionary contains the disorder name, the key path within the JSON,
        and the index of the criterion within its disorder.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    criteria_texts: List[str] = []
    metadata: List[Dict[str, str]] = []

    def _flatten(node: Any, disorder: str, key_path: List[str]) -> None:
        """Recursively traverse the JSON tree collecting string leaves."""
        if isinstance(node, dict):
            for key, value in node.items():
                _flatten(value, disorder, key_path + [str(key)])
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                _flatten(item, disorder, key_path + [str(idx)])
        elif isinstance(node, str):
            # Clean up whitespace and skip empty strings
            text = node.strip()
            if text:
                criteria_texts.append(text)
                metadata.append(
                    {
                        "disorder": disorder,
                        "path": "/".join(key_path) if key_path else "root",
                        "criterion_index": len(criteria_texts) - 1,
                    }
                )
        # Ignore other types (e.g. numbers, booleans)

    # Top level of the JSON is assumed to map disorder names to definitions
    if isinstance(data, dict):
        for disorder_name, node in data.items():
            _flatten(node, disorder_name, [])
    else:
        # Fallback: treat entire file as one disorder
        _flatten(data, "unknown", [])

    return criteria_texts, metadata


class CriteriaRetriever:
    """Simple semantic retriever based on TF‑IDF and cosine similarity.

    This class converts each criterion into a TF‑IDF vector and precomputes
    its norm so that cosine similarities with a new query can be computed
    efficiently.  It does not require external dependencies like FAISS and
    therefore works in constrained environments.
    """

    def __init__(self, criteria: List[str]):
        if not criteria:
            raise ValueError("No criteria provided for retrieval")
        # We strip whitespace and enforce lowercase to reduce variance
        processed = [c.lower().strip() for c in criteria]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(processed)
        # Precompute norms for cosine similarity; add small epsilon to avoid div by zero
        dense_matrix = self.matrix.toarray().astype(np.float32)
        self.doc_norms = np.linalg.norm(dense_matrix, axis=1) + 1e-10
        self.criteria = criteria

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Retrieve the top_k most relevant criteria for a given query.

        Parameters
        ----------
        query : str
            The input text describing symptoms or clinical context.
        top_k : int, optional
            The number of criteria to return, by default 5.

        Returns
        -------
        List of tuples (index, score, text)
            The index of the criterion in the original list, the cosine
            similarity score, and the criterion text itself.
        """
        if not query:
            return []
        q_vec = self.vectorizer.transform([query.lower().strip()])
        q_dense = q_vec.toarray().astype(np.float32)
        # Compute dot products with all documents
        dot_products = self.matrix @ q_dense.T  # shape (n_docs, 1)
        dot_products = dot_products.toarray().squeeze()
        q_norm = np.linalg.norm(q_dense) + 1e-10
        similarities = dot_products / (self.doc_norms * q_norm)
        # Get indices of top_k scores in descending order
        top_indices = np.argsort(-similarities)[: top_k]
        results = []
        for idx in top_indices:
            results.append((idx, float(similarities[idx]), self.criteria[idx]))
        return results


class LocalLLM:
    """Wrapper for a locally hosted HuggingFace causal language model.

    This class manages the tokenizer and model and provides a `generate` method
    that produces completions for a given prompt.  It supports CPU and GPU
    inference transparently.  If the `transformers` package is missing, the
    class raises a runtime error at initialization.
    """

    def __init__(self, model_path: str, device: str = None):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers is not installed. Please install transformers to load the model."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load the model in half precision if CUDA is available to save memory
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # Automatically use half precision on CUDA for faster inference
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype
        )
        self.model.to(self.device)
        # Some models require special tokens to be appended; the user can override this later
        self.max_length = 2048

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate text from the model given a prompt.

        Parameters
        ----------
        prompt : str
            The input prompt fed to the model.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate, by default 256.
        temperature : float, optional
            Sampling temperature; lower values make the output more deterministic.

        Returns
        -------
        str
            The decoded model output, with special tokens stripped.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        # The first token sequence contains the prompt plus the generated continuation
        generated = outputs[0]
        # Decode and remove the prompt portion
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        return decoded


def build_prompt_with_retrieved(context: str, retrieved: List[str]) -> str:
    """Construct a prompt for the model that includes retrieved criteria.

    The prompt explicitly lists the retrieved DSM‑5 criteria and instructs the
    model to decide which criteria are met given the context.  Feel free to
    customise this function to better match your model’s instruction format.

    Parameters
    ----------
    context : str
        The input description of patient behaviour or symptoms.
    retrieved : list of str
        The retrieved criteria texts.

    Returns
    -------
    str
        A formatted prompt suitable for generation.
    """
    criteria_block = "\n".join(
        [f"Criterion {i+1}: {text}" for i, text in enumerate(retrieved)]
    )
    prompt = (
        "You are a diagnostic assistant referencing the DSM-5 criteria.\n"
        "Below are selected DSM-5 diagnostic criteria that may relate to the patient's description.\n"
        f"{criteria_block}\n\n"
        "Context: "
        f"{context}\n\n"
        "Based on the above criteria and context, identify which criteria are satisfied and suggest the most likely disorders. "
        "Explain your reasoning in a concise manner."
    )
    return prompt


def build_prompt_baseline(context: str) -> str:
    """Construct a prompt for the model without any retrieved criteria.

    This baseline prompt asks the model to identify relevant DSM‑5 criteria and
    disorders solely based on its internal knowledge.  It does not supply
    external context beyond the patient description.

    Parameters
    ----------
    context : str
        The input description of patient behaviour or symptoms.

    Returns
    -------
    str
        A formatted prompt suitable for generation.
    """
    prompt = (
        "You are a diagnostic assistant knowledgeable about the DSM-5.\n"
        "Context: "
        f"{context}\n\n"
        "Identify any DSM-5 diagnostic criteria that appear to be met and suggest potential disorders. "
        "Explain your reasoning in a concise manner."
    )
    return prompt


def run_rag(
    llm: LocalLLM,
    retriever: CriteriaRetriever,
    contexts: Iterable[str],
    top_k: int = 5,
    max_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """Execute RAG inference for a list of contexts.

    Each context is processed by retrieving the top_k criteria, constructing a
    prompt, generating a response via the LLM and timing the process.  The
    results, including the retrieved criteria, the model’s output and the
    elapsed time, are returned for later analysis.

    Parameters
    ----------
    llm : LocalLLM
        The language model wrapper.
    retriever : CriteriaRetriever
        The retriever used to select relevant criteria.
    contexts : iterable of str
        A collection of patient descriptions.
    top_k : int, optional
        Number of criteria to retrieve for each context, by default 5.
    max_tokens : int, optional
        Maximum number of tokens the model should generate, by default 256.

    Returns
    -------
    list of dict
        Each dictionary contains the context, retrieved indices and texts,
        generated output and the time taken in seconds.
    """
    results = []
    for ctx in contexts:
        start = time.perf_counter()
        retrieved_results = retriever.retrieve(ctx, top_k=top_k)
        retrieved_texts = [text for _, _, text in retrieved_results]
        prompt = build_prompt_with_retrieved(ctx, retrieved_texts)
        output = llm.generate(prompt, max_new_tokens=max_tokens)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "context": ctx,
                "retrieved_indices": [idx for idx, _, _ in retrieved_results],
                "retrieved_scores": [score for _, score, _ in retrieved_results],
                "retrieved_texts": retrieved_texts,
                "output": output.strip(),
                "time_sec": elapsed,
            }
        )
    return results


def run_baseline(
    llm: LocalLLM,
    contexts: Iterable[str],
    max_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """Execute baseline inference for a list of contexts using the raw LLM.

    Each context is passed directly to the LLM without retrieval.  The prompt
    instructs the model to identify DSM‑5 criteria and disorders based solely
    on its built‑in knowledge.  The time required to generate the response is
    recorded.

    Parameters
    ----------
    llm : LocalLLM
        The language model wrapper.
    contexts : iterable of str
        A collection of patient descriptions.
    max_tokens : int, optional
        Maximum number of tokens the model should generate, by default 256.

    Returns
    -------
    list of dict
        Each dictionary contains the context, generated output and the time
        taken in seconds.
    """
    results = []
    for ctx in contexts:
        start = time.perf_counter()
        prompt = build_prompt_baseline(ctx)
        output = llm.generate(prompt, max_new_tokens=max_tokens)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "context": ctx,
                "output": output.strip(),
                "time_sec": elapsed,
            }
        )
    return results


def save_results_json(filename: str, data: List[Dict[str, Any]]) -> None:
    """Write results to a JSON file.

    Parameters
    ----------
    filename : str
        Output filename.  Directories are created if necessary.
    data : list of dict
        The results data to save.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_results_csv(filename: str, rag_results: List[Dict[str, Any]], baseline_results: List[Dict[str, Any]]) -> None:
    """Write combined RAG and baseline results to a CSV file if pandas is available.

    Each row contains the context, RAG output, baseline output and timing
    statistics.  Retrieval information is omitted from the CSV to keep it
    concise; refer to the JSON file for complete details.

    Parameters
    ----------
    filename : str
        Output CSV filename.
    rag_results : list of dict
        Results from RAG inference.
    baseline_results : list of dict
        Results from baseline inference.
    """
    if pd is None:
        print(
            "pandas is not available; skipping CSV export. Install pandas to enable this feature."
        )
        return
    rows = []
    for rag, base in zip(rag_results, baseline_results):
        rows.append(
            {
                "context": rag["context"],
                "rag_output": rag["output"],
                "rag_time": rag["time_sec"],
                "baseline_output": base["output"],
                "baseline_time": base["time_sec"],
            }
        )
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple DSM-5 RAG pipeline with a TAIDE LLM and compare it with a baseline."
        )
    )
    parser.add_argument(
        "--dsm_path",
        type=str,
        default="DSM-5/DSM_Criteria_Array_Fixed.json",
        help="Path to the DSM-5 JSON file containing disorders and criteria.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="taide/Llama-3.1-TAIDE-LX-8B-Chat",
        help=(
            "Path to the locally stored HuggingFace model directory (e.g., Llama3-TAIDE-LX-8B-Chat)."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of criteria to retrieve for each context.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--context",
        nargs="*",
        default=None,
        help=(
            "One or more patient description strings. If omitted, example contexts will be used."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to write JSON and CSV outputs.",
    )
    args = parser.parse_args()

    # Load DSM-5 criteria and metadata
    criteria_texts, metadata = load_dsm5_criteria(args.dsm_path)
    print(f"Loaded {len(criteria_texts)} criteria from {args.dsm_path}")

    # Build the retriever
    retriever = CriteriaRetriever(criteria_texts)

    # Load the LLM
    llm = LocalLLM(args.model_path)

    # Use provided contexts or fall back to sample ones
    if args.context:
        contexts = args.context
    else:
        contexts = [
            "The patient reports feeling sad and hopeless most of the day nearly every day for the past two months, has lost interest in activities, and has trouble sleeping.",
            "A child shows persistent inattention, is easily distracted, and has difficulty remaining seated at school, often interrupting others.",
        ]
        print("No contexts provided; using built-in example contexts.")

    # Run RAG and baseline
    rag_results = run_rag(
        llm=llm,
        retriever=retriever,
        contexts=contexts,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    baseline_results = run_baseline(
        llm=llm,
        contexts=contexts,
        max_tokens=args.max_tokens,
    )

    # Compute average times
    avg_rag_time = np.mean([r["time_sec"] for r in rag_results]) if rag_results else 0.0
    avg_baseline_time = np.mean([r["time_sec"] for r in baseline_results]) if baseline_results else 0.0
    print(f"Average RAG response time: {avg_rag_time:.3f} seconds")
    print(f"Average baseline response time: {avg_baseline_time:.3f} seconds")

    # Save results to JSON
    rag_path = os.path.join(args.output_dir, "rag_results.json")
    base_path = os.path.join(args.output_dir, "baseline_results.json")
    save_results_json(rag_path, rag_results)
    save_results_json(base_path, baseline_results)
    print(f"Saved RAG results to {rag_path}")
    print(f"Saved baseline results to {base_path}")

    # Save combined CSV if possible
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    save_results_csv(csv_path, rag_results, baseline_results)
    if pd is not None:
        print(f"Saved comparison CSV to {csv_path}")


if __name__ == "__main__":
    main()