"""Do-no-harm gated rerun on the physical-constants domain.

Mode A regenerates only when the round-0 value fails the policy check AND the
Platt-calibrated trust — fit on the development half and frozen before the holdout run —
falls below the gate threshold (`domain/calibration.py`). Gating this way is what keeps
the rerun from degrading answers that were already right.

Inputs come from `configs/gated_constants.yaml`. The output file is opened with
truncation."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from grounded_matsci.domain import codata
from grounded_matsci.domain.calibration import TRUST_THRESHOLD, platt
from grounded_matsci.io.openrouter import make_text_generate

logger = logging.getLogger(__name__)


def run(prompts_path, gt_path, out_path, models, system, reps=(1, 2, 3), workers=12):
    cp = json.loads(Path(prompts_path).read_text())
    gt2022 = json.loads(Path(gt_path).read_text())["constants"]

    def run_cell(args):
        p, model, rep, arm = args
        ms = model.split("/")[-1]
        gen = make_text_generate(model)
        subj = p["subject"]
        rec = gt2022.get(subj)
        out = {"subject": subj, "model": ms, "stratum": p["stratum"], "rep": rep, "arm": arm}
        # round 0
        txt = gen([{"role": "system", "content": system}, {"role": "user", "content": p["prompt"]}])
        correct0, _, _ = codata.grade_const_policy(txt, rec)
        # trust = 1 if the committed value passes the policy check, 0 if it fails; None if no commit
        trust = 1.0 if correct0 is True else (0.0 if correct0 is False else None)
        out["round0_correct"] = correct0
        out["trust"] = trust
        out["platt_trust"] = platt(trust) if trust is not None else None
        if arm == "baseline":
            out["final_text"] = txt
            out["regenerated"] = False
            out["gate_suppressed"] = False
            return out
        # gated Mode A: regenerate only if flagged (correct0 is False) AND platt(trust) < threshold
        flagged = correct0 is False
        gate_open = flagged and (
            out["platt_trust"] is not None and out["platt_trust"] < TRUST_THRESHOLD
        )
        out["flagged"] = flagged
        out["gate_suppressed"] = flagged and not gate_open
        if gate_open:
            fb = (
                "A reference tool (CODATA 2022) flagged your physical-constant value as "
                "incorrect. Provide a corrected answer fixing ONLY this value using the "
                "reference; introduce no new claims.\n\nFlagged:\n"
                f'  - "{subj}": CODATA reference value is {p["gt_value"]} {p.get("unit", "")}'
            )
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": p["prompt"]},
                {"role": "assistant", "content": txt},
                {"role": "user", "content": fb},
            ]
            txt2 = gen(msgs)
            out["final_text"] = txt2 if (txt2 or "").strip() else txt
            out["regenerated"] = True
        else:
            out["final_text"] = txt
            out["regenerated"] = False
        return out

    tasks = [
        (p, m, rep, arm)
        for rep in reps
        for arm in ["baseline", "mode_a"]
        for m in models
        for p in cp["prompts"]
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
            if done % 80 == 0:
                logger.info("%d/%d %.0fs", done, len(tasks), time.time() - t0)
    logger.info("DONE %d/%d in %.0fs", done, len(tasks), time.time() - t0)
    return done
