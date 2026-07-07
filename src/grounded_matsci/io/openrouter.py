"""
OpenRouter chat-completions clients.

The experiment runners in the original bundle carried two DISTINCT call families,
duplicated across files. They are deduplicated here but deliberately NOT unified —
they differ in retry codes, timeout, temperature handling, and return shape, and the
paper's runs depend on those exact behaviors:

1. `make_generate` (from the closed-loop driver, `llm_driver.py`): content-only,
   requests usage/cost accounting, 4 attempts, retries on {429, 502, 503},
   timeout 120 s, RAISES on exhaustion, and appends per-call usage records to `log`.
2. `generate_text` (the `_gen` helper shared by the arms/selfcheck/inline-confidence/
   CODATA/end-task runners): returns reasoning + content concatenated, 5 attempts,
   retries on {400, 429, 500, 502, 503}, timeout 180 s, returns "" on failure.
3. `generate_result` (from the holdout harness): like `generate_text` but returns a
   dict carrying text/content/reasoning/usage or an error marker, because the harness
   records usage and error fields per row.

The API key is read from OPENROUTER_API_KEY and never logged.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import requests

_URL = "https://openrouter.ai/api/v1/chat/completions"

Message = dict[str, str]


def _key() -> str:
    return os.environ["OPENROUTER_API_KEY"]


def make_generate(
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 900,
    log: list[dict[str, Any]] | None = None,
) -> Callable[..., str]:
    """Driver for the closed-loop experiment (spec Section 3.1).

    Returns a `generate(messages, sample=?)` callable that loop.py consumes. Records
    the exact model string and every call's token usage for the reproducibility log
    (spec Section 5)."""
    key = os.environ["OPENROUTER_API_KEY"]
    headers = {"Authorization": f"Bearer {key}"}

    def generate(messages: list[Message], sample: bool = False) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if sample else 0.0,
            # ask OpenRouter to return token accounting AND real USD cost
            "usage": {"include": True},
        }
        for attempt in range(4):
            try:
                t0 = time.time()
                r = requests.post(_URL, headers=headers, json=payload, timeout=120)
                if r.status_code == 200:
                    j = r.json()
                    choice = j["choices"][0]
                    txt = choice["message"].get("content")
                    finish = choice.get("finish_reason")
                    # Some models return content=None -- a content-filter refusal
                    # (finish_reason='content_filter') or an empty completion.
                    # Treat as an empty trace so the pipeline never crashes, and
                    # record the refusal as a distinct outcome (not a hallucination).
                    refused = txt is None
                    if txt is None:
                        txt = ""
                    if log is not None:
                        u = j.get("usage", {}) or {}
                        log.append({"model": model, "sample": sample,
                                    "prompt_tokens": u.get("prompt_tokens", 0),
                                    "completion_tokens": u.get("completion_tokens", 0),
                                    "total_tokens": u.get("total_tokens", 0),
                                    "cost_usd": u.get("cost", 0.0),
                                    "wall_s": time.time() - t0,
                                    "finish_reason": finish,
                                    "refused": refused,
                                    "ts": time.time()})
                    return str(txt)
                if r.status_code in (429, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:200]}")
            except requests.RequestException:
                time.sleep(2 ** attempt)
        raise RuntimeError("OpenRouter: exhausted retries")

    return generate


def generate_text(
    model: str,
    messages: list[Message],
    sample: bool = False,
    max_tokens: int = 1200,
    timeout: int = 180,
) -> str:
    """The `_gen` helper shared byte-identically by the arms/selfcheck/inline-confidence/
    CODATA/end-task runners: reasoning + content concatenated, "" on failure. Callers
    that hardcoded temperature 0.0 pass sample=False (identical payload)."""
    for a in range(5):
        try:
            r = requests.post(_URL,
                headers={"Authorization": f"Bearer {_key()}"},
                json={"model": model, "messages": messages, "max_tokens": max_tokens,
                      "temperature": 0.7 if sample else 0.0}, timeout=timeout)
            if r.status_code == 200:
                m = r.json()["choices"][0]["message"]
                c = m.get("content") or ""
                rz = m.get("reasoning") or ""
                return str((rz + "\n\n" + c).strip() if rz else c)
            if r.status_code in (400, 429, 500, 502, 503):
                time.sleep(2 ** a)
                continue
            return ""
        except Exception:
            time.sleep(2 ** a)
    return ""


def make_text_generate(model: str, max_tokens: int = 1200) -> Callable[..., str]:
    """Closure form of `generate_text` (the `make_generate` in the original arms/end-task
    runners); exp2/exp3 import this instead of reaching into the arms runner."""

    def g(messages: list[Message], sample: bool = False) -> str:
        return generate_text(model, messages, sample=sample, max_tokens=max_tokens)

    return g


def generate_result(
    model: str,
    messages: list[Message],
    sample: bool = False,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """Holdout-harness variant: returns {text, content, reasoning, usage} or an error
    marker, because the harness records usage and error fields per checkpoint row."""
    for a in range(5):
        try:
            r = requests.post(_URL,
                headers={"Authorization": f"Bearer {_key()}"},
                json={"model": model, "messages": messages,
                      "max_tokens": max_tokens,
                      "temperature": 0.7 if sample else 0.0}, timeout=180)
            if r.status_code == 200:
                j = r.json()
                msg = j["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning") or ""
                # the verified trace = reasoning trace + committed answer (confabulations
                # occur in the reasoning; the paper's whole premise). Grade over both.
                text = (reasoning + "\n\n" + content).strip() if reasoning else content
                return {"text": text, "content": content, "reasoning": reasoning,
                        "usage": j.get("usage", {})}
            if r.status_code in (400, 429, 500, 502, 503):
                time.sleep(2 ** a)
                continue
            return {"text": "", "error": f"http{r.status_code}"}
        except Exception:
            time.sleep(2 ** a)
    return {"text": "", "error": "retries_exhausted"}
