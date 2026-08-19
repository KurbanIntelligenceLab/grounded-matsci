"""Derived-quantity verification tier. Extends grounding to the DERIVED
quantity the end-task answer actually depends on — molar mass computed from the verified formula
(atomic-weight sum), stability comparison from verified MP formation energies — not just the
object identity.
Tests whether closing the scope gap restores the end-task lift that object-only grounding
did not deliver. Orchestration layer only; core frozen shas untouched. Registered ex ante; both
outcomes (lift / null) reportable.

Molar mass is computed from the verified molecular formula by summing standard IUPAC atomic weights
(`domain/composition.py`); stability comparison uses the verified Materials Project formation
energies directly. No SMILES parsing is needed since the verified object is already the formula.

Model list and system prompt come from `configs/endtask_derived.yaml`; the OpenRouter call lives
in `io/openrouter.generate_text` (greedy, max_tokens 1400)."""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import re
import threading
from pathlib import Path

from grounded_matsci.domain.composition import molar_mass_from_formula
from grounded_matsci.io.checkpoint import load_done_cells
from grounded_matsci.io.openrouter import generate_text

logger = logging.getLogger(__name__)


def _gen(model, messages):
    return generate_text(model, messages, sample=False, max_tokens=1400)


def _parse_answer(txt, ttype):
    if not txt:
        return None
    if ttype == "molar_mass":
        m = re.findall(r"[Aa]nswer\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", txt)
        if m:
            return float(m[-1])
        m2 = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*g\s*/\s*mol", txt)
        return float(m2[-1]) if m2 else None
    m = re.findall(r"[Aa]nswer\s*[:=]\s*\(?([AB])\b", txt)
    return m[-1].upper() if m else None


def _derived_truth(task):
    """Correct answer computed from the VERIFIED objects in task.depends_on."""
    if task["type"] == "molar_mass":
        return molar_mass_from_formula(task["depends_on"]["formula"])
    if task["type"] == "compare_molar":
        (_na, fa), (_nb, fb) = list(task["depends_on"].items())
        ma, mb = molar_mass_from_formula(fa), molar_mass_from_formula(fb)
        return "A" if ma > mb else "B"
    if task["type"] == "compare_stability":
        (_na, va), (_nb, vb) = list(task["depends_on"].items())
        return "A" if va < vb else "B"  # more negative = more stable
    return None


def _derived_feedback(task, stated, truth):
    if task["type"] == "molar_mass":
        f = task["depends_on"]["formula"]
        return (
            f"A derived-quantity verification tool computed the molar mass directly from the "
            f"verified molecular formula {f} and obtained {truth} g/mol, which differs from the "
            f"value in your answer. Provide a corrected final answer using this computed value; "
            f"change nothing else."
        )
    kind = (
        "molar masses computed from the verified formulas"
        if task["type"] == "compare_molar"
        else "formation energies per atom from the Materials Project"
    )
    return (
        f"A derived-quantity verification tool compared the two candidates using {kind} and "
        f"determined the correct choice is {truth}, which differs from your answer. Provide a "
        f"corrected final answer (just the letter) consistent with this; change nothing else."
    )


def run_endtask_derived(
    tasks, ckpt_path, models, system, rep=0, workers=16, tol=0.01, log_every=50
):
    """Mode A with the derived-quantity tier: detect a wrong final answer against the derived truth,
    inject the computed value, re-generate. Stores final_text + flag."""
    done = load_done_cells(ckpt_path)
    todo = [
        (i, t, m, f"{i}|{m.split('/')[-1]}|r{rep}")
        for i, t in enumerate(tasks)
        for m in models
        if f"{i}|{m.split('/')[-1]}|r{rep}" not in done
    ]
    logger.info("endtask-derived: %d cells (%d done)", len(todo), len(done))
    lock = threading.Lock()
    f = Path(ckpt_path).open("a")  # noqa: SIM115 - handle shared across worker threads
    n = [0]

    def work(args):
        i, t, m, cid = args
        truth = _derived_truth(t)
        try:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": t["prompt"]}]
            txt = _gen(m, msgs)
            flagged = False
            for _ in range(2):  # up to 2 correction rounds
                ans = _parse_answer(txt, t["type"])
                if ans is None:
                    break
                if t["type"] == "molar_mass":
                    wrong = truth is not None and abs(ans - truth) / truth > tol
                else:
                    wrong = ans != truth
                if not wrong:
                    break
                flagged = True
                msgs += [
                    {"role": "assistant", "content": txt},
                    {"role": "user", "content": _derived_feedback(t, ans, truth)},
                ]
                txt = _gen(m, msgs)
            rec = {"final_text": txt, "flagged": flagged}
        except Exception as e:
            rec = {"final_text": "", "flagged": None, "error": str(e)[:200]}
        row = {
            "cell": cid,
            "task_idx": i,
            "model": m,
            "condition": "mode_a_derived",
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
