"""
Intervention-arms runner (conditions 2-5) over the frozen v2 corpus. Checkpointed to JSONL,
resumable, concurrent. Each cell = (prompt, model, condition, rep). Stores the final text +
in-loop metadata (rounds, in-loop fails, retrieved facts) for grading + detection/flag-rate.

Model list and system prompt come from `configs/arms.yaml`; the OpenRouter call lives in
`io/openrouter` (the ""-on-failure text variant).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import threading
from pathlib import Path

from grounded_matsci.io.checkpoint import load_done_cells
from grounded_matsci.io.openrouter import make_text_generate

logger = logging.getLogger(__name__)


def build_fact_lookup(prompt_to_facts):
    def f(prompt):
        return prompt_to_facts.get(prompt, "")

    return f


def run_arms(
    corpus,
    ckpt_path,
    conditions,
    gt_kwargs_for,
    fact_lookup,
    models,
    system,
    rep=0,
    workers=12,
    log_every=100,
    limit=None,
):
    """conditions: list subset of ['self_critique','rag_in_prompt','mode_a','mode_b'].
    gt_kwargs_for(prompt_rec) -> dict of ground_trace kwargs (named_sg_lookup etc.) for
    in-loop verify."""
    from . import loop

    done = load_done_cells(ckpt_path)
    todo = []
    for i, c in enumerate(corpus):
        if limit and i >= limit:
            break
        for m in models:
            for cond in conditions:
                cid = f"{i}|{m.split('/')[-1]}|{cond}|r{rep}"
                if cid not in done:
                    todo.append((i, c, m, cond, cid))
    logger.info("intervention arms: %d cells (%d done), %d workers", len(todo), len(done), workers)
    lock = threading.Lock()
    f = Path(ckpt_path).open("a")  # noqa: SIM115 - handle shared across worker threads
    n = [0]

    def work(args):
        i, c, m, cond, cid = args
        gen = make_text_generate(m)
        gk = gt_kwargs_for(c)
        sysmsg = system
        try:
            if cond == "self_critique":
                res = loop.run_self_critique(c["prompt"], gen, gt_kwargs=gk, system=sysmsg)
                rec = {
                    "final_text": res["final_text"],
                    "rounds": res.get("rounds"),
                    "inloop_fail": res.get("final_n_fail"),
                }
            elif cond == "rag_in_prompt":
                res = loop.run_rag_in_prompt(
                    c["prompt"], gen, fact_lookup, gt_kwargs=gk, system=sysmsg
                )
                rec = {
                    "final_text": res["final_text"],
                    "retrieved": bool(res.get("retrieved_facts")),
                    "inloop_fail": res.get("final_n_fail"),
                }
            elif cond == "mode_a":
                res = loop.run_mode_a(c["prompt"], gen, max_rounds=3, gt_kwargs=gk, system=sysmsg)
                traj = res.get("trajectory", [])
                rec = {
                    "final_text": res["final_text"],
                    "rounds": res.get("rounds"),
                    "inloop_fail_round0": traj[0]["n_fail"] if traj else None,
                    "inloop_fail_final": res.get("final_n_fail"),
                    "flagged": bool(traj and traj[0]["n_fail"] > 0),
                }
            elif cond == "mode_b":
                res = loop.run_mode_b(
                    c["prompt"], gen, n=8, gt_kwargs=gk, system=sysmsg, matched_baseline=True
                )
                rec = {
                    "final_text": res["best"]["text"],
                    "matched_text": res["matched_baseline"]["text"],
                    "best_inloop_fail": res["best"]["n_fail"],
                    "matched_inloop_fail": res["matched_baseline"]["n_fail"],
                }
            else:
                rec = {"error": f"unknown {cond}"}
        except Exception as e:
            rec = {"final_text": "", "error": str(e)[:200]}
        row = {
            "cell": cid,
            "prompt_idx": i,
            "model": m,
            "condition": cond,
            "rep": rep,
            "stratum": c["stratum"],
            "claim_type": c["claim_type"],
            "subject": c["subject"],
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
