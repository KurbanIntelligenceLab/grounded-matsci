"""Item 2: sampling-consistency detection (SelfCheckGPT-style, Manakul et al. 2023). Runs Mode B
(8 samples) and stores the extracted checkable object from EACH sample so cross-sample agreement can
be computed as a detection signal. Low agreement = flag. The frozen extractor produces each sample's
object; no core changes. Compared against the deterministic verifier and the LLM judge in the §2.7
detection table.

Model list and system prompt come from `configs/selfcheck.yaml`; the OpenRouter call lives in
`io/openrouter.generate_text`."""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import re as _re
import threading
from pathlib import Path

from grounded_matsci.io.checkpoint import load_done_cells
from grounded_matsci.io.openrouter import generate_text

logger = logging.getLogger(__name__)

_F = _re.compile(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*){1,})\b")
_SG = _re.compile(r"(?:space group\s*(?:No\.?|number|#)?\s*|No\.?\s*|#|\(#?)\s*(\d{1,3})")
_EF = _re.compile(r"(-?\d+\.?\d*)\s*(?:eV\s*/\s*atom|eV\s*per\s*atom)")


def _sample_objects(text, gt_kwargs, claim_type):
    """Extract the committed object of the task's claim_type from one sample (for cross-sample
    agreement only). Direct regex mirrors the frozen extractor's targets; agreement is over the
    LAST committed value in the sample."""
    if not text:
        return []
    if claim_type == "molecular":
        m = [x for x in _F.findall(text) if any(ch.isdigit() for ch in x) and len(x) > 2]
        return [m[-1]] if m else []
    if claim_type == "crystalline":
        m = _SG.findall(text)
        return [m[-1]] if m else []
    if claim_type == "property_ef":
        m = _EF.findall(text)
        return [f"{float(m[-1]):.3g}"] if m else []
    return []


def run_selfcheck(
    prompts, ckpt_path, gt_kwargs_for, models, system, n=8, rep=0, workers=12, log_every=100
):
    """Store per-sample extracted objects for cross-sample agreement (SelfCheckGPT)."""
    done = load_done_cells(ckpt_path)
    todo = [
        (i, p, m, f"{i}|{m.split('/')[-1]}|r{rep}")
        for i, p in enumerate(prompts)
        for m in models
        if f"{i}|{m.split('/')[-1]}|r{rep}" not in done
    ]
    logger.info("selfcheck: %d cells (%d done)", len(todo), len(done))
    lock = threading.Lock()
    f = Path(ckpt_path).open("a")  # noqa: SIM115 - handle shared across worker threads
    n_done = [0]

    def work(args):
        i, p, m, cid = args
        gk = gt_kwargs_for(p)
        ct = p["claim_type"]
        try:
            samples = []
            for _ in range(n):
                txt = generate_text(
                    m,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": p["prompt"]},
                    ],
                    sample=True,
                )
                samples.append(_sample_objects(txt, gk, ct))
            rec = {"sample_objects": samples}
        except Exception as e:
            rec = {"sample_objects": [], "error": str(e)[:200]}
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
            n_done[0] += 1
            if n_done[0] % log_every == 0:
                logger.info("  %d/%d", n_done[0], len(todo))
        return cid

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    f.close()
    return len(todo)
