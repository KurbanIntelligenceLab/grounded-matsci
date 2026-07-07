"""Cross-domain transfer (manuscript item 3): physical-constants domain, Mode-A runner half.

The grading/parsing/sig-fig policy half lives in `domain/codata.py`. Model list and system
prompt come from `configs/codata.yaml`; the OpenRouter call lives in
`io/openrouter.generate_text` (greedy)."""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import threading
from pathlib import Path

from grounded_matsci.domain.codata import grade_const
from grounded_matsci.io.checkpoint import load_done_cells
from grounded_matsci.io.openrouter import generate_text

logger = logging.getLogger(__name__)


class _ConstFail:
    """Minimal fail object mirroring the materials verifier's claim (.raw, .detail) so loop's
    _fail_feedback can format the correction message unchanged."""

    def __init__(self, raw, detail):
        self.raw = raw
        self.detail = detail


def run_mode_a_const(
    prompt, generate, subject, gt_value, unit, rel_tol=0.01, max_rounds=3, system=None
):
    """Gated Mode A for the constants domain: same detect->inject-reference->repair loop as the
    materials run_mode_a, but the check is grade_const against the frozen CODATA value."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    trajectory = []
    text = generate(messages)
    for rnd in range(max_rounds):
        correct, extracted = grade_const(text, gt_value, rel_tol)
        flagged = correct is False
        trajectory.append({"round": rnd, "text": text, "extracted": extracted, "flagged": flagged})
        if not flagged:
            break
        fb = (
            "A reference tool (CODATA 2018) flagged a physical-constant value in your previous "
            "answer as incorrect. Provide a corrected answer that fixes ONLY this value using the "
            "tool feedback and introduces no new claims.\n\nFlagged:\n"
            f'  - "{subject}": stated {extracted}, CODATA reference value is {gt_value} {unit}'
        )
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": fb})
        text = generate(messages)
    if not (text or "").strip():
        prior = [t["text"] for t in trajectory if (t["text"] or "").strip()]
        if prior:
            text = prior[-1]
    return {
        "final_text": text,
        "rounds": len(trajectory),
        "flagged": bool(trajectory and trajectory[0]["flagged"]),
        "trajectory": trajectory,
    }


def run_codata(prompts, ckpt_path, models, system, rep=0, workers=16, rel_tol=0.01, log_every=50):
    """baseline + gated Mode A over the constants prompts. Stores final_text + flag for grading."""
    done = load_done_cells(ckpt_path)
    todo = []
    for i, p in enumerate(prompts):
        for m in models:
            for cond in ("baseline", "mode_a"):
                cid = f"{i}|{m.split('/')[-1]}|{cond}|r{rep}"
                if cid not in done:
                    todo.append((i, p, m, cond, cid))
    logger.info("codata arms: %d cells (%d done), %d workers", len(todo), len(done), workers)
    lock = threading.Lock()
    f = Path(ckpt_path).open("a")  # noqa: SIM115 - handle shared across worker threads
    n = [0]

    def work(args):
        i, p, m, cond, cid = args

        def gen(msgs, sample=False):
            return generate_text(m, msgs, sample=False)

        try:
            if cond == "baseline":
                txt = gen(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": p["prompt"]},
                    ]
                )
                rec = {"final_text": txt, "flagged": None}
            else:
                res = run_mode_a_const(
                    p["prompt"],
                    gen,
                    p["subject"],
                    p["gt_value"],
                    p["unit"],
                    rel_tol=rel_tol,
                    system=system,
                )
                rec = {
                    "final_text": res["final_text"],
                    "rounds": res["rounds"],
                    "flagged": res["flagged"],
                }
        except Exception as e:
            rec = {"final_text": "", "error": str(e)[:200]}
        row = {
            "cell": cid,
            "prompt_idx": i,
            "model": m,
            "condition": cond,
            "rep": rep,
            "stratum": p["stratum"],
            "claim_type": "phys_const",
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
