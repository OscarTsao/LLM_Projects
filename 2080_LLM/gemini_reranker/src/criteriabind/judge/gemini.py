"""Gemini judge provider with caching, retries, and optional Vertex support."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

from ..config_schemas import AppConfig
from ..judge.mock_judge import MockJudge
from ..schemas import Candidate, EvidenceSpan, Judgment, JudgingJob, Preference


LOGGER = logging.getLogger(__name__)


class GeminiMissingDependencyError(ImportError):
    """Raised when google-generativeai or vertexai is unavailable."""


class GeminiCache:
    """Lightweight SQLite-backed cache for Gemini responses."""

    def __init__(self, uri: Optional[Path]) -> None:
        self._enabled = bool(uri)
        if not self._enabled:
            self._conn = None
            return
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path.as_posix(), check_same_thread=False)
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judgments (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    prompt_version TEXT NOT NULL
                )
                """
            )

    def get(self, key: str) -> Optional[dict[str, object]]:
        if not self._enabled or self._conn is None:
            return None
        with self._lock:
            cursor = self._conn.execute("SELECT payload FROM judgments WHERE cache_key=?", (key,))
            row = cursor.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def put(self, key: str, payload: dict[str, object], prompt_version: str) -> None:
        if not self._enabled or self._conn is None:
            return
        serialised = json.dumps(payload, ensure_ascii=False)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO judgments(cache_key, payload, created_at, prompt_version)
                VALUES (?, ?, ?, ?)
                """,
                (key, serialised, time.time(), prompt_version),
            )


class GeminiJudge:
    """Gemini-powered judge that produces pairwise/listwise preferences."""

    _PROMPT_VERSION = "v1"

    def __init__(self, app_cfg: AppConfig, api_key: Optional[str] = None) -> None:
        self.app_cfg = app_cfg
        self.cfg = app_cfg.judge
        self.dataset_id = app_cfg.data.name
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or self.cfg.api_key or os.environ.get("GEMINI_API_KEY")

        self.random = random.Random(app_cfg.seed)
        self.cache = GeminiCache(self.cfg.cache_uri)
        self.mock_fallback = MockJudge(model_name="gemini-fallback")
        self.total_cost = 0.0
        self.request_times: deque[float] = deque()
        self.cost_lock = threading.Lock()

        self._init_client()

    def _init_client(self) -> None:
        if self.cfg.vertex_enabled:
            try:
                import vertexai  # type: ignore
                from vertexai.preview.generative_models import GenerativeModel as VertexGenerativeModel  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise GeminiMissingDependencyError("vertexai package is required for Vertex support.") from exc

            if not self.cfg.vertex_project or not self.cfg.vertex_location:
                raise RuntimeError("judge.vertex.project and judge.vertex.location must be set when vertex_enabled=true.")
            vertexai.init(project=self.cfg.vertex_project, location=self.cfg.vertex_location)
            model_name = self.cfg.vertex_model or self.cfg.model
            self._client = VertexGenerativeModel(model_name)
            self._client_type = "vertex"
        else:
            try:
                import google.generativeai as genai  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise GeminiMissingDependencyError("google-generativeai package is required for Gemini judging.") from exc

            if not self.api_key:
                raise RuntimeError("Gemini API key not provided. Set GEMINI_API_KEY or judge.api_key.")
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.cfg.model)
            self._client_type = "generativeai"

    # --------------------------------------------------------------------- API #
    def batch(self, jobs: Iterable[JudgingJob]) -> list[Judgment]:
        job_list = list(jobs)
        if not job_list:
            return []
        max_workers = max(1, int(self.cfg.max_concurrency))
        if max_workers == 1:
            return [self.score(job) for job in job_list]
        results: list[Optional[Judgment]] = [None] * len(job_list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.score, job): idx for idx, job in enumerate(job_list)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
        return [judgment for judgment in results if judgment is not None]

    def score(self, job: JudgingJob) -> Judgment:
        cache_key = self._cache_key(job)
        cached = self.cache.get(cache_key)
        if cached is not None:
            cached.setdefault("provider", "gemini")
            cached.setdefault("model", self.cfg.model)
            cached.setdefault("meta", {}).update({"cached": True})
            return Judgment.model_validate(cached)

        if self._should_skip(job):
            self.logger.info("Skipping Gemini call for job %s due to sample_rate", job.job_id)
            return self.mock_fallback.score(job)

        if not self._budget_allows():
            self.logger.warning("Gemini budget exceeded; falling back to mock for job %s", job.job_id)
            return self.mock_fallback.score(job)

        attempt = 0
        strict_mode = False
        last_error: Optional[Exception] = None
        start_time = time.perf_counter()
        while attempt <= self.cfg.max_retries:
            try:
                payload = self._call_gemini(job, strict=strict_mode)
                latency = time.perf_counter() - start_time
                judgment = self._build_judgment(job, payload, latency)
                self.cache.put(cache_key, judgment.to_dict(), self._PROMPT_VERSION)
                return judgment
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                self.logger.warning(
                    "Gemini call failed for job %s (attempt %s/%s): %s",
                    job.job_id,
                    attempt + 1,
                    self.cfg.max_retries + 1,
                    exc,
                )
                delay = (self.cfg.retry_base ** attempt) + self.random.random() * 0.5
                time.sleep(min(delay, self.cfg.timeout_s))
                attempt += 1
                strict_mode = True  # tighten prompt for retries

        self.logger.error("Gemini judge failed after retries for job %s: %s", job.job_id, last_error)
        if self.cfg.fallback_to_mock:
            return self.mock_fallback.score(job)
        raise RuntimeError(f"Gemini judge failed for job {job.job_id}: {last_error}") from last_error

    # ----------------------------------------------------------------- Helpers #
    def _should_skip(self, job: JudgingJob) -> bool:
        if self.cfg.dry_run:
            return True
        if self.cfg.sample_rate >= 1.0:
            return False
        return self.random.random() > self.cfg.sample_rate

    def _budget_allows(self) -> bool:
        if self.cfg.max_requests_per_minute:
            now = time.time()
            window = self.cfg.max_requests_per_minute
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            if len(self.request_times) >= window:
                return False
            self.request_times.append(now)
        if self.cfg.max_total_cost_usd is not None and self.total_cost >= self.cfg.max_total_cost_usd:
            return False
        return True

    def _call_gemini(self, job: JudgingJob, *, strict: bool) -> dict[str, Any]:
        prompt = self._build_prompt(job, strict=strict)
        if self.cfg.dry_run:
            self.logger.info("Dry-run Gemini prompt for job %s:\n%s", job.job_id, prompt)
            return self._mock_payload(job)

        if self._client_type == "generativeai":
            return self._call_generativeai(prompt)
        return self._call_vertex(prompt)

    def _call_generativeai(self, prompt: str) -> dict[str, Any]:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise GeminiMissingDependencyError("google-generativeai package missing.") from exc

        generation_config = {
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_output_tokens": self.cfg.max_output_tokens,
        }
        if self.cfg.top_k is not None:
            generation_config["top_k"] = self.cfg.top_k

        response = self._client.generate_content(
            [prompt],
            generation_config=generation_config,
            request_options={"timeout": self.cfg.timeout_s},
            response_mime_type=self.cfg.response_mime,
        )
        text = getattr(response, "text", None)
        if not text:
            candidates = getattr(response, "candidates", None)
            if candidates:
                text = "".join(part.text for c in candidates for part in getattr(c, "content", {}).parts)
        if not text:
            raise RuntimeError("Gemini response did not contain text content.")

        payload = self._parse_payload(text)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            tokens_in = usage.get("prompt_token_count") or usage.get("prompt_tokens", 0)
            tokens_out = usage.get("candidates_token_count") or usage.get("completion_token_count", 0)
            payload.setdefault("_usage", {"input_tokens": tokens_in, "output_tokens": tokens_out})
            self._record_cost(tokens_in, tokens_out)
        return payload

    def _call_vertex(self, prompt: str) -> dict[str, Any]:
        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.cfg.temperature,
                    "top_p": self.cfg.top_p,
                    "max_output_tokens": self.cfg.max_output_tokens,
                },
                safety_settings=None,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"VertexAI request failed: {exc}") from exc
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("VertexAI response missing text content.")
        payload = self._parse_payload(text)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            tokens_in = usage.get("prompt_token_count", 0)
            tokens_out = usage.get("candidates_token_count", 0)
            payload.setdefault("_usage", {"input_tokens": tokens_in, "output_tokens": tokens_out})
            self._record_cost(tokens_in, tokens_out)
        return payload

    def _parse_payload(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini response was not valid JSON: {text}") from exc

    def _mock_payload(self, job: JudgingJob) -> dict[str, Any]:
        indexes = list(range(len(job.candidates)))
        return {
            "ranking": indexes,
            "pairs": [{"winner": indexes[0], "loser": idx} for idx in indexes[1:]],
            "evidence_spans": [],
            "rationale": "dry-run placeholder",
            "_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def _build_judgment(self, job: JudgingJob, payload: dict[str, Any], latency: float) -> Judgment:
        num_candidates = len(job.candidates)
        ranking = payload.get("ranking") or list(range(num_candidates))
        pairs = payload.get("pairs") or []
        evidence_spans_payload = payload.get("evidence_spans") or []
        rationale = payload.get("rationale")

        if not ranking:
            ranking = list(range(num_candidates))
        best_idx = ranking[0] if ranking else 0

        preferences: list[Preference] = []
        for item in pairs:
            try:
                winner = int(item["winner"])
                loser = int(item["loser"])
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= winner < num_candidates and 0 <= loser < num_candidates and winner != loser:
                preferences.append(Preference(winner_idx=winner, loser_idx=loser, weight=float(item.get("weight", 1.0))))

        if not preferences:
            # derive pairwise preferences from ranking
            for i, winner_idx in enumerate(ranking):
                for loser_idx in ranking[i + 1 :]:
                    preferences.append(Preference(winner_idx=winner_idx, loser_idx=loser_idx))

        evidence_spans: list[EvidenceSpan] = []
        for span in evidence_spans_payload:
            try:
                evidence_spans.append(
                    EvidenceSpan(
                        candidate_index=int(span["candidate_index"]),
                        start=int(span["start"]),
                        end=int(span["end"]),
                        confidence=span.get("confidence"),
                        metadata=span.get("metadata", {}),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue

        usage = payload.get("_usage", {})
        token_usage = {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }

        meta = {
            **job.meta,
            "prompt_version": self._PROMPT_VERSION,
            "cached": False,
            "cost_usd": usage.get("cost_usd", 0.0),
        }

        return Judgment(
            job_id=job.job_id,
            note_id=job.note_id,
            criterion_id=job.criterion_id,
            criterion=job.criterion,
            criterion_text=job.criterion_text,
            note_text=job.note_text,
            candidates=job.candidates,
            best_idx=best_idx,
            preferences=preferences,
            ranking=ranking,
            evidence_spans=evidence_spans,
            rationale=rationale,
            provider="gemini",
            model=self.cfg.model,
            latency_s=latency,
            token_usage=token_usage,
            meta=meta,
        )

    def _record_cost(self, input_tokens: int, output_tokens: int) -> None:
        # Heuristic pricing for Gemini 1.5 Pro (approximate, USD per 1K tokens)
        IN_COST = 0.0007
        OUT_COST = 0.0021
        cost = (input_tokens / 1000.0) * IN_COST + (output_tokens / 1000.0) * OUT_COST
        with self.cost_lock:
            self.total_cost += cost

    def _cache_key(self, job: JudgingJob) -> str:
        import hashlib

        payload = {
            "dataset": self.dataset_id,
            "split": job.meta.get("split"),
            "criterion_id": job.criterion_id,
            "model": self.cfg.model,
            "prompt_version": self._PROMPT_VERSION,
            "candidates": [candidate.text for candidate in job.candidates],
        }
        serialised = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    def _build_prompt(self, job: JudgingJob, *, strict: bool) -> str:
        instructions = (
            "You are a clinical adjudicator. Given a diagnostic criterion and a set of candidate "
            "snippets extracted from a patient note, identify the best supporting snippet and return "
            "STRICT JSON following the schema:\n"
            "{\n"
            '  "ranking": [int,...],\n'
            '  "pairs": [{"winner": int, "loser": int}, ...],\n'
            '  "evidence_spans": [{"candidate_index": int, "start": int, "end": int}],\n'
            '  "rationale": "optional short string"\n'
            "}\n"
            "Ranking must order candidates from best to worst. Pairs should include winner/loser indices "
            "consistent with the ranking. Use deterministic tie-breaking by preferring lower candidate indices.\n"
        )
        if strict:
            instructions += (
                "Respond with JSON only (no markdown, no commentary). "
                "Ensure all integers correspond to valid candidate indices.\n"
            )

        candidate_block = "\n".join(
            f"{idx}. {cand.text.strip()}" for idx, cand in enumerate(job.candidates)
        )
        note_section = job.note_text.strip()
        criterion_section = job.criterion_text.strip()

        prompt = (
            f"{instructions}\n"
            f"Criterion ID: {job.criterion_id}\n"
            f"Criterion Definition: {criterion_section}\n\n"
            f"Patient Note:\n{note_section}\n\n"
            "Candidates:\n"
            f"{candidate_block}\n"
            "Return JSON only."
        )
        return prompt
