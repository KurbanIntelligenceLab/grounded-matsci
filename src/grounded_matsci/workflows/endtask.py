"""End-task propagation runner. Each task's final answer depends on a
checkable object (formula -> molar mass; formula/EF -> comparative selection). Runs baseline +
gated Mode A + Mode B over 4 models, triplicate; stores final_text for answer-grading and the
in-loop flag for propagation analysis. Reuses loop.run_mode_a / run_mode_b and the OpenRouter
driver. Core verifier config untouched; end-task answer grading is orchestration-layer.

Model list and system prompt come from `configs/endtask.yaml`; the OpenRouter call lives in
`io/openrouter.generate_text` with the end-task token budget (max_tokens 1400)."""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import threading
from pathlib import Path

from grounded_matsci.io.checkpoint import load_done_cells
from grounded_matsci.io.openrouter import generate_text

logger = logging.getLogger(__name__)


def _make_generate(model):
    def g(messages, sample=False):
        return generate_text(model, messages, sample=sample, max_tokens=1400)

    return g


def run_endtask(
    tasks,
    ckpt_path,
    gt_kwargs_for,
    models,
    system,
    conditions=("baseline", "mode_a", "mode_b"),
    rep=0,
    workers=12,
    log_every=50,
):
    """gt_kwargs_for(task)->ground_trace kwargs for the objects the task depends on
    (for in-loop verify)."""
    from . import loop

    done = load_done_cells(ckpt_path)
    todo = []
    for i, t in enumerate(tasks):
        for m in models:
            for cond in conditions:
                cid = f"{i}|{m.split('/')[-1]}|{cond}|r{rep}"
                if cid not in done:
                    todo.append((i, t, m, cond, cid))
    logger.info("end-task arms: %d cells (%d done), %d workers", len(todo), len(done), workers)
    lock = threading.Lock()
    f = Path(ckpt_path).open("a")  # noqa: SIM115 - handle shared across worker threads
    n = [0]

    def work(args):
        i, t, m, cond, cid = args
        gen = _make_generate(m)
        gk = gt_kwargs_for(t)
        try:
            if cond == "baseline":
                txt = gen(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": t["prompt"]},
                    ]
                )
                rec = {"final_text": txt, "flagged": None}
            elif cond == "mode_a":
                res = loop.run_mode_a(t["prompt"], gen, max_rounds=3, gt_kwargs=gk, system=system)
                traj = res.get("trajectory", [])
                rec = {
                    "final_text": res["final_text"],
                    "rounds": res.get("rounds"),
                    "flagged": bool(traj and traj[0]["n_fail"] > 0),
                }
            elif cond == "mode_b":
                res = loop.run_mode_b(
                    t["prompt"], gen, n=8, gt_kwargs=gk, system=system, matched_baseline=True
                )
                rec = {
                    "final_text": res["best"]["text"],
                    "matched_text": res["matched_baseline"]["text"],
                }
            else:
                rec = {"error": f"unknown {cond}"}
        except Exception as e:
            rec = {"final_text": "", "error": str(e)[:200]}
        row = {
            "cell": cid,
            "task_idx": i,
            "model": m,
            "condition": cond,
            "rep": rep,
            "type": t["type"],
            "stratum": t["stratum"],
            "subject": t["subject"],
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
