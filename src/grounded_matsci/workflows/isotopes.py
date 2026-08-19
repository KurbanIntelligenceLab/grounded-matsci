"""Isotope half-life domain: Mode A with the isotope verifier tier.

Exercises the orchestration-layer isotope tier (`domain/isotope.py`) and grades against
the frozen IAEA ground truth committed at `data/reference/isotope_gt.json`. This is the
second transfer domain: neither the extractor nor the verifier was tuned on it.

Inputs and the system prompt come from `configs/isotopes.yaml`; the generate callable is
`io/openrouter.make_generate`, the usage-logging driver family. The output file is opened
with truncation."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from grounded_matsci.domain import isotope
from grounded_matsci.io.openrouter import make_generate

logger = logging.getLogger(__name__)


def human_hl(sec):
    # produce a readable reference for the correction feedback
    for u, s in [
        ("Gy", 31557600e9),
        ("My", 31557600e6),
        ("ky", 31557600e3),
        ("y", 31557600),
        ("d", 86400),
        ("h", 3600),
        ("s", 1),
    ]:
        if sec >= s:
            return f"{sec / s:.4g} {u}"
    return f"{sec:.4g} s"


def isotope_verify(text, gt_sec, ref_name):
    """orchestration-layer isotope tier: flag if stated half-life outside tolerance;
    feed reference."""
    g = isotope.grade_halflife(text, gt_sec)
    if g["correct"] is None:  # no commit -> no fail (abstain)
        return []
    if g.get("any_correct"):
        return []

    class _F:
        raw = f"half-life of {ref_name}"
        detail = (
            f"the stated half-life ({g['extracted_sec']:.4g} s) disagrees with the reference value "
            f"({human_hl(gt_sec)} = {gt_sec:.4g} s) by more than 10%. Use the reference value."
        )

    return [_F()]


def run_mode_a_iso(prompt, generate, gt_sec, ref_name, system, max_rounds=3):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    traj = []
    text = generate(messages)
    for rnd in range(max_rounds):
        fails = isotope_verify(text, gt_sec, ref_name)
        traj.append({"round": rnd, "text": text, "n_fail": len(fails)})
        if not fails:
            break
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": "A nuclear-data verification tool flagged your answer:\n  - "
                + fails[0].detail
                + "\nProvide a corrected answer using the reference value; "
                "change only the half-life.",
            }
        )
        text = generate(messages)
    if not (text or "").strip():
        prior = [t["text"] for t in traj if (t["text"] or "").strip()]
        if prior:
            text = prior[-1]
    return {"final_text": text, "rounds": len(traj), "traj_fail": [t["n_fail"] for t in traj]}


def run(gt_path, prompts_path, out_path, models, system, reps=(1, 2, 3), workers=12):
    gt = json.loads(Path(gt_path).read_text())["isotopes"]
    prompts = json.loads(Path(prompts_path).read_text())
    gt_by_name = {v["name"]: v for v in gt.values()}

    def run_cell(args):
        rec, model, rep, arm = args
        ms = model.split("/")[-1]
        g = gt_by_name[rec["subject"]]
        gt_sec = g["half_life_sec"]
        gen = make_generate(model)
        out = {
            "subject": rec["subject"],
            "nucid": rec["nucid"],
            "model": ms,
            "stratum": rec["stratum"],
            "rep": rep,
            "arm": arm,
            "gt_sec": gt_sec,
        }
        if arm == "baseline":
            txt = gen(
                [{"role": "system", "content": system}, {"role": "user", "content": rec["prompt"]}]
            )
            out["final_text"] = txt
            out["rounds"] = 1
            out["traj_fail"] = []
        else:
            res = run_mode_a_iso(rec["prompt"], gen, gt_sec, rec["subject"], system)
            out.update(
                final_text=res["final_text"], rounds=res["rounds"], traj_fail=res["traj_fail"]
            )
        return out

    tasks = [
        (rec, m, rep, arm) for rep in reps for arm in ["mode_a"] for m in models for rec in prompts
    ]
    t0 = time.time()
    done = 0
    with Path(out_path).open("w") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_cell, t) for t in tasks]
        for fu in as_completed(futs):
            try:
                fh.write(json.dumps(fu.result()) + "\n")
                fh.flush()
            except Exception as e:
                fh.write(json.dumps({"error": str(e)[:200]}) + "\n")
            done += 1
            if done % 100 == 0:
                logger.info("%d/%d %.0fs", done, len(tasks), time.time() - t0)
    logger.info("DONE %d/%d in %.0fs", done, len(tasks), time.time() - t0)
    return done
