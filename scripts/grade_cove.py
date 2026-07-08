#!/usr/bin/env python3
"""Grade a CoVe arms JSONL with the FROZEN grader (grade_all.grade_record), matching the
arms-grading path (rule="any" for post-correction conditions). Validates on the existing
arms before use. Run from repo root:  python scripts/grade_cove.py

Emits results/cove_graded.json (per-cell correct/None) and prints per-surface err|commit.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

from grounded_matsci.domain import extract as extract_mod
from grounded_matsci.evaluation import grade_all
from grounded_matsci.verification import ground as ground_mod

REPO = Path(__file__).resolve().parent.parent
FROZEN = REPO / "data" / "frozen"
RESULTS = REPO / "results"
SURFACES = ("molecular", "property_ef", "crystalline")


def _load(name):
    return json.load(open(FROZEN / name))


def build_refs():
    gt = _load("holdout_ground_truth.json")
    named = _load("named_sg_rule_gt.json")
    named_sg_lookup = named.get("token_accepted_sg", named)
    ef_lookup = _load("ef_lookup.json")
    # exp_ef_ref: subject -> {'exp_ef_eV_atom': value}; keys in ef_lookup are lowercased names
    exp_ef_ref = {}
    for k, v in ef_lookup.items():
        if isinstance(v, dict) and "exp" in v:
            exp_ef_ref[k] = {"exp_ef_eV_atom": v["exp"]}
    exp_gap_ref = {k: {"exp_gap": val} for k, val in gt.get("experimental_gap", {}).items()}
    known_names = [s.lower() for s in gt.get("molecular", {})] + [s.lower() for s in gt.get("crystalline", {})]
    tol = gt["tolerances"]
    return gt, tol, named_sg_lookup, exp_ef_ref, exp_gap_ref, known_names


def grade_cells(cells, refs, rule="any"):
    gt, tol, named_sg_lookup, exp_ef_ref, exp_gap_ref, known_names = refs
    out = []
    for c in cells:
        if c["claim_type"] not in SURFACES:
            continue
        rec = {"claim_type": c["claim_type"], "subject": c["subject"], "text": c["text"]}
        g = grade_all.grade_record(
            rec, gt, tol, extract_mod, ground_mod, named_sg_lookup, known_names,
            exp_gap_ref=exp_gap_ref, rule=rule, exp_ef_ref=exp_ef_ref,
        )
        out.append({**{k: c[k] for k in ("subject", "claim_type", "stratum", "model", "rep", "condition")},
                    "correct": g.get("correct"), "dropped": g.get("dropped", False)})
    return out


def _out(r):
    if r.get("dropped"):
        return None
    c = r["correct"]
    return "abstain" if c is None else ("correct" if c else "wrong")


def err_commit(graded, ct, cond=None):
    sub = [r for r in graded if r["claim_type"] == ct and (cond is None or r.get("condition") == cond)]
    committed = [r for r in sub if _out(r) in ("correct", "wrong")]
    return (sum(_out(r) == "wrong" for r in committed) / len(committed)) if committed else None


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def main():
    refs = build_refs()
    cove = [json.loads(l) for l in open(FROZEN / "cove_arms.jsonl") if l.strip()]
    graded = grade_cells(cove, refs, rule="any")
    (RESULTS / "cove_graded.json").write_text(json.dumps(graded))
    print("CoVe graded cells:", len(graded))
    for ct in SURFACES:
        e = err_commit(graded, ct, "cove")
        print(f"  {ct}: CoVe err|commit = {e:.2%}" if e is not None else f"  {ct}: no commits")


if __name__ == "__main__":
    main()
