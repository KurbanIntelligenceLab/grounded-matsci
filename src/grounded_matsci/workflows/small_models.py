"""Small-model capability probe: baseline versus Mode A on two small closed models.

Runs the same corpus and the same grounding loop as the main arms, but on two small
frontier-family models (verified live 2026-07-06), to test whether the repair benefit
depends on model scale.

Inputs, including the model list and the shared materials system prompt, come from
`configs/small_models.yaml`; the generate closure comes from
`io/openrouter.make_text_generate`. The output file is opened with truncation."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from grounded_matsci.io.openrouter import make_text_generate
from grounded_matsci.workflows import loop

logger = logging.getLogger(__name__)


def run(
    corpus_path, named_sg_path, ef_lookup_path, out_path, models, system, reps=(1, 2, 3), workers=12
):
    corpus = json.loads(Path(corpus_path).read_text())
    named_sg = json.loads(Path(named_sg_path).read_text())
    ef_lookup = json.loads(Path(ef_lookup_path).read_text())
    sg_lookup = named_sg.get("token_accepted_sg", named_sg)

    def gk_for(rec):
        return {
            "enable_physics": True,
            "ef_lookup": ef_lookup,
            "named_sg_lookup": sg_lookup,
            "extra_known_names": [rec["subject"].lower()],
        }

    def run_cell(args):
        i, rec, model, rep, arm = args
        ms = model.split("/")[-1]
        gen = make_text_generate(model)
        gk = gk_for(rec)
        out = {
            "prompt_idx": i,
            "subject": rec["subject"],
            "model": ms,
            "stratum": rec["stratum"],
            "claim_type": rec["claim_type"],
            "rep": rep,
            "arm": arm,
        }
        try:
            if arm == "baseline":
                txt = gen(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": rec["prompt"]},
                    ]
                )
                out["final_text"] = txt
            else:
                res = loop.run_mode_a(rec["prompt"], gen, max_rounds=3, gt_kwargs=gk, system=system)
                traj = res.get("trajectory", [])
                out.update(
                    final_text=res["final_text"],
                    rounds=res.get("rounds"),
                    inloop_fail_round0=traj[0]["n_fail"] if traj else None,
                    inloop_fail_final=res.get("final_n_fail"),
                    flagged=bool(traj and traj[0]["n_fail"] > 0),
                )
        except Exception as e:
            out["final_text"] = ""
            out["error"] = str(e)[:200]
        return out

    tasks = [
        (i, rec, m, rep, arm)
        for rep in reps
        for arm in ["baseline", "mode_a"]
        for m in models
        for i, rec in enumerate(corpus)
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
            if done % 200 == 0:
                logger.info("%d/%d %.0fs", done, len(tasks), time.time() - t0)
    logger.info("DONE %d/%d in %.0fs", done, len(tasks), time.time() - t0)
    return done
