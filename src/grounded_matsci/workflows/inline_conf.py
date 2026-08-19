"""Inline verbalized-confidence detector baseline. Re-runs the unguarded baseline with a
confidence
elicitation appended to the SAME generation call, so the model commits an answer and a 0-100
confidence together (contrast with the offline post-hoc elicitation). Separate calibration-only arm;
graded against frozen GT. Core verifier untouched.

Model list, system prompt, and the confidence-suffix string come from
`configs/inline_conf.yaml`; the OpenRouter call lives in `io/openrouter.generate_text`
(these cells are greedy: sample=False == temperature 0.0)."""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import re
import threading
from pathlib import Path

from grounded_matsci.io.checkpoint import load_done_cells
from grounded_matsci.io.openrouter import generate_text

logger = logging.getLogger(__name__)


def extract_conf(text):
    m = re.findall(r"[Cc]onfidence\s*[:=]\s*(\d{1,3}(?:\.\d+)?)", text or "")
    return min(float(m[-1]), 100) / 100 if m else None


def run_inline_conf(
    prompts, ckpt_path, models, system, conf_suffix, rep=0, workers=16, log_every=100
):
    """prompts: [{subject, prompt, claim_type, stratum}]. Stores final_text + inline confidence."""
    done = load_done_cells(ckpt_path)
    todo = [
        (i, p, m, f"{i}|{m.split('/')[-1]}|r{rep}")
        for i, p in enumerate(prompts)
        for m in models
        if f"{i}|{m.split('/')[-1]}|r{rep}" not in done
    ]
    logger.info("inline-conf: %d cells (%d done)", len(todo), len(done))
    lock = threading.Lock()
    f = Path(ckpt_path).open("a")  # noqa: SIM115 - handle shared across worker threads
    n = [0]

    def work(args):
        i, p, m, cid = args
        try:
            txt = generate_text(
                m,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": p["prompt"] + conf_suffix},
                ],
                sample=False,
            )
            rec = {"final_text": txt, "confidence": extract_conf(txt)}
        except Exception as e:
            rec = {"final_text": "", "confidence": None, "error": str(e)[:200]}
        row = {
            "cell": cid,
            "prompt_idx": i,
            "model": m,
            "rep": rep,
            "claim_type": p["claim_type"],
            "stratum": p["stratum"],
            "subject": p["subject"],
            **rec,
        }
        with lock:
            f.write(json.dumps(row) + "\n")
            f.flush()
            n[0] += 1
            if n[0] % log_every == 0:
                logger.info("  %d/%d", n[0], len(todo))
        return cid

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    f.close()
    return len(todo)
