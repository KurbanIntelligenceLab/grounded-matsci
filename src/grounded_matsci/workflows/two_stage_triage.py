"""Compute-aware two-stage triage detector over the formation-energy prompts.

The deterministic verifier tier fires first; the sampling-consistency signal (threshold
0.25, frozen on the development half before the holdout run) is invoked ONLY on units the
deterministic tier does not flag, so the expensive signal is spent where the cheap one is
silent (`verification/two_stage.py`).

`two_stage_patch` monkey-patches `loop._verify` for the duration of a cell. Giving
`loop.py` a `verify_fn` parameter would change a frozen scientific core, so the patch is
the least-invasive composition point.

Inputs come from `configs/two_stage_triage.yaml`. The output file is opened with
truncation, unlike the resumable append-mode runners — rerunning overwrites it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from grounded_matsci.io.openrouter import make_generate
from grounded_matsci.verification import two_stage
from grounded_matsci.workflows import loop

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def two_stage_patch(cons):
    orig = loop._verify
    loop._verify = two_stage.make_cell_verify(orig, cons)
    try:
        yield
    finally:
        loop._verify = orig


def run(
    corpus_path,
    ef_lookup_path,
    selfcheck_detection_path,
    out_path,
    models,
    system,
    reps=(1, 2, 3),
    workers=12,
):
    ef_lookup = json.loads(Path(ef_lookup_path).read_text())
    det = json.loads(Path(selfcheck_detection_path).read_text())
    cons_lookup = {}
    for r in det:
        if r["claim_type"] == "property_ef":
            cons_lookup[(r["subject"].lower(), r["model"].split("/")[-1])] = r["consistency"]

    corpus = json.loads(Path(corpus_path).read_text())
    ef_prompts = [
        c
        for c in (corpus if isinstance(corpus, list) else corpus.get("prompts", corpus))
        if (c.get("claim_type") or c.get("type")) == "property_ef"
    ]

    def gk_for(rec):
        return {
            "enable_physics": True,
            "ef_lookup": ef_lookup,
            "extra_known_names": [rec["subject"].lower()],
        }

    def run_cell(args):
        rec, model, rep = args
        ms = model.split("/")[-1]
        cons = cons_lookup.get((rec["subject"].lower(), ms))
        gen = make_generate(model)
        gk = gk_for(rec)
        out = {
            "prompt_idx": rec.get("prompt_idx"),
            "subject": rec["subject"],
            "model": ms,
            "stratum": rec["stratum"],
            "claim_type": "property_ef",
            "rep": rep,
            "consistency": cons,
        }
        # two-stage Mode A
        with two_stage_patch(cons):
            res = loop.run_mode_a(rec["prompt"], gen, max_rounds=3, gt_kwargs=gk, system=system)
        out["twostage_final_text"] = res["final_text"]
        out["twostage_rounds"] = res["rounds"]
        out["twostage_n_fail"] = res["final_n_fail"]
        out["twostage_traj_fail"] = [t["n_fail"] for t in res["trajectory"]]
        return out

    tasks = [(rec, m, rep) for rep in reps for m in models for rec in ef_prompts]
    done = 0
    t0 = time.time()
    with Path(out_path).open("w") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_cell, t) for t in tasks]
        for fu in as_completed(futs):
            try:
                fh.write(json.dumps(fu.result()) + "\n")
                fh.flush()
            except Exception as e:
                fh.write(json.dumps({"error": str(e)[:200]}) + "\n")
            done += 1
            if done % 50 == 0:
                logger.info("%d/%d %.0fs", done, len(tasks), time.time() - t0)
    logger.info("DONE %d/%d in %.0fs -> %s", done, len(tasks), time.time() - t0, out_path)
    return done
