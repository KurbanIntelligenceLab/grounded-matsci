"""
Holdout run harness — checkpointed, resumable, per-(prompt,model,condition,replicate).

Registered configuration: prompt corpus sha b61516df, holdout ground truth sha 9171f2e4,
extractor sha 42e03ae2 -- all frozen before this pass was run.
Generations are checkpointed to a JSONL so the run resumes after any interruption.

Model list and system prompt come from `configs/harness.yaml`; the OpenRouter call
lives in `io/openrouter.generate_result` (usage/error-recording variant).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from grounded_matsci.io import openrouter
from grounded_matsci.io.checkpoint import load_done_records

logger = logging.getLogger(__name__)


def cell_id(prompt_idx, model, condition, rep):
    return f"{prompt_idx}|{model.split('/')[-1]}|{condition}|r{rep}"


def run_baseline_pass(corpus, ckpt_path, models, system, rep=0, limit=None, log_every=50):
    """Condition 1 (unguarded): one generation per (prompt,model). Appends to JSONL."""
    done = load_done_records(ckpt_path)
    todo = []
    for i, c in enumerate(corpus):
        if limit and i >= limit:
            break
        for m in models:
            cid = cell_id(i, m, "unguarded", rep)
            if cid not in done:
                todo.append((i, c, m, cid))
    logger.info("baseline pass: %d cells to run (%d already done)", len(todo), len(done))
    f = Path(ckpt_path).open("a")  # noqa: SIM115 - handle shared across worker threads
    for n, (i, c, m, cid) in enumerate(todo):
        res = openrouter.generate_result(
            m, [{"role": "system", "content": system}, {"role": "user", "content": c["prompt"]}]
        )
        rec = {
            "cell": cid,
            "prompt_idx": i,
            "model": m,
            "condition": "unguarded",
            "rep": rep,
            "stratum": c["stratum"],
            "claim_type": c["claim_type"],
            "subject": c["subject"],
            "text": res["text"],
            "usage": res.get("usage", {}),
            "error": res.get("error"),
        }
        f.write(json.dumps(rec) + "\n")
        f.flush()
        if (n + 1) % log_every == 0:
            logger.info("  %d/%d done", n + 1, len(todo))
    f.close()
    return len(todo)


def run_baseline_concurrent(
    corpus, ckpt_path, models, system, rep=0, limit=None, workers=16, log_every=200
):
    """Concurrent condition-1 pass. Thread pool over (prompt,model) cells; JSONL append
    is serialized by a lock so the checkpoint stays consistent and resumable."""
    import concurrent.futures as cf
    import threading

    done = load_done_records(ckpt_path)
    todo = []
    for i, c in enumerate(corpus):
        if limit and i >= limit:
            break
        for m in models:
            cid = cell_id(i, m, "unguarded", rep)
            if cid not in done:
                todo.append((i, c, m, cid))
    logger.info(
        "concurrent baseline: %d cells (%d done), %d workers", len(todo), len(done), workers
    )
    lock = threading.Lock()
    f = Path(ckpt_path).open("a")  # noqa: SIM115 - handle shared across worker threads
    n_done = [0]

    def work(args):
        i, c, m, cid = args
        res = openrouter.generate_result(
            m, [{"role": "system", "content": system}, {"role": "user", "content": c["prompt"]}]
        )
        rec = {
            "cell": cid,
            "prompt_idx": i,
            "model": m,
            "condition": "unguarded",
            "rep": rep,
            "stratum": c["stratum"],
            "claim_type": c["claim_type"],
            "subject": c["subject"],
            "text": res["text"],
            "usage": res.get("usage", {}),
            "error": res.get("error"),
        }
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_done[0] += 1
            if n_done[0] % log_every == 0:
                logger.info("  %d/%d done", n_done[0], len(todo))
        return cid

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    f.close()
    return len(todo)
