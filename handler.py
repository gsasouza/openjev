"""Runpod Serverless handler for OpenJev Phase 1 decision scoring.

One causal model is loaded at cold start and reused for every job. The
model and its pinned revision come from the environment so the same image
can serve either baseline: the direct/serial/shared modes expect a causal
model such as Qwen/Qwen3.5-4B, the reranker mode expects the native
reranker such as Qwen/Qwen3-Reranker-4B.
"""

from __future__ import annotations

import os

import runpod

from openjev_phase1.core import load_causal_model, validate_row
from openjev_phase1.direct import score as direct_score
from openjev_phase1.reranker import score as reranker_score
from openjev_phase1.serial import SerialPrefixScorer
from openjev_phase1.shared import score_shared
from worker_cache import DEFAULT_CACHE_ROOT, resolve_cached_snapshot

MODEL_SOURCE = os.environ["OPENJEV_MODEL"]
MODEL_REVISION = os.environ["OPENJEV_REVISION"]
MAX_TOKENS = int(os.environ.get("OPENJEV_MAX_TOKENS", "4096"))
MAX_ROWS = int(os.environ.get("OPENJEV_MAX_ROWS", "64"))
DEFAULT_MODE = os.environ.get("OPENJEV_MODE", "direct")
CACHE_ROOT = os.environ.get("OPENJEV_CACHE_ROOT", DEFAULT_CACHE_ROOT)
MODES = ("direct", "serial", "shared", "reranker")

MODEL_PATH = resolve_cached_snapshot(MODEL_SOURCE, MODEL_REVISION, CACHE_ROOT)
MODEL, TOKENIZER, METADATA = load_causal_model(MODEL_PATH, MODEL_REVISION)
# load_causal_model reports the path it was handed; keep the Hub id in the
# provenance that ships with every scored row.
METADATA = {**METADATA, "source": MODEL_SOURCE, "cache_path": MODEL_PATH}


def collect_rows(payload: dict) -> list[dict]:
    """Accept either a single `row` or a `rows` list and validate every entry."""
    if "rows" in payload and "row" in payload:
        raise ValueError("Send either row or rows, not both")
    rows = payload.get("rows", [payload["row"]] if "row" in payload else None)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input needs a nonempty rows array, or a single row object")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"{len(rows)} rows exceed the {MAX_ROWS}-row limit for one job")
    for row in rows:
        validate_row(row)
    return rows


def run(mode: str, rows: list[dict], max_tokens: int) -> dict:
    if mode == "shared":
        results, timing = score_shared(MODEL, TOKENIZER, rows, METADATA, max_tokens)
        return {"results": results, "shared_timing": timing}
    if mode == "serial":
        scorer = SerialPrefixScorer(MODEL, TOKENIZER, METADATA, max_tokens)
        return {"results": [scorer.score(row) for row in rows]}
    scorer = direct_score if mode == "direct" else reranker_score
    return {"results": [scorer(MODEL, TOKENIZER, row, METADATA, max_tokens) for row in rows]}


def handler(job: dict) -> dict:
    payload = job.get("input") or {}
    try:
        mode = payload.get("mode", DEFAULT_MODE)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        max_tokens = int(payload.get("max_tokens", MAX_TOKENS))
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        rows = collect_rows(payload)
        return {"mode": mode, "model": METADATA, **run(mode, rows, max_tokens)}
    except (ValueError, KeyError, TypeError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


runpod.serverless.start({"handler": handler})
