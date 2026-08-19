#!/usr/bin/env python3
"""Grade the chain-of-verification arm with the frozen grader.

Uses ``grade_all.grade_record`` with ``rule="final"``, the final-commitment rule. The
frozen grading policy reserves the lenient any-match rule for the verifier-corrected
Mode A and Mode B arms; CoVe revises its own draft without a verifier reference, so it is
graded on its final commitment like the other non-corrected conditions. Grading it under
any-match would credit CoVe for a value it mentioned and then abandoned, and would not be
comparable to the arms it is meant to be contrasted with.

Reads ``data/frozen/cove_arms.jsonl`` and writes ``results/cove_graded.json`` (per-cell
correct/abstain), then prints the per-surface error-given-commit rate. Run from the
repository root:

    uv run python scripts/grade_cove.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path
from typing import Any

from grounded_matsci.domain import extract as extract_mod
from grounded_matsci.evaluation import grade_all
from grounded_matsci.verification import ground as ground_mod

REPO = Path(__file__).resolve().parent.parent
FROZEN = REPO / "data" / "frozen"
RESULTS = REPO / "results"
SURFACES = ("molecular", "property_ef", "crystalline")

# (ground truth, tolerances, space-group lookup, formation-energy refs, gap refs, names)
Refs = tuple[dict, dict, dict, dict, dict, list[str]]


def _load(name: str) -> Any:  # noqa: ANN401 - raw JSON boundary
    return json.loads((FROZEN / name).read_text())


def build_refs() -> Refs:
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
    known_names = [s.lower() for s in gt.get("molecular", {})] + [
        s.lower() for s in gt.get("crystalline", {})
    ]
    tol = gt["tolerances"]
    return gt, tol, named_sg_lookup, exp_ef_ref, exp_gap_ref, known_names


def grade_cells(cells: list[dict], refs: Refs, rule: str = "any") -> list[dict[str, Any]]:
    gt, tol, named_sg_lookup, exp_ef_ref, exp_gap_ref, known_names = refs
    out = []
    for c in cells:
        if c["claim_type"] not in SURFACES:
            continue
        rec = {"claim_type": c["claim_type"], "subject": c["subject"], "text": c["text"]}
        g = grade_all.grade_record(
            rec,
            gt,
            tol,
            extract_mod,
            ground_mod,
            named_sg_lookup,
            known_names,
            exp_gap_ref=exp_gap_ref,
            rule=rule,
            exp_ef_ref=exp_ef_ref,
        )
        keys = ("subject", "claim_type", "stratum", "model", "rep", "condition")
        out.append(
            {
                **{k: c[k] for k in keys},
                "correct": g.get("correct"),
                "dropped": g.get("dropped", False),
            }
        )
    return out


def _out(r: dict) -> str | None:
    if r.get("dropped"):
        return None
    c = r["correct"]
    return "abstain" if c is None else ("correct" if c else "wrong")


def err_commit(graded: list[dict], ct: str, cond: str | None = None) -> float | None:
    sub = [
        r for r in graded if r["claim_type"] == ct and (cond is None or r.get("condition") == cond)
    ]
    committed = [r for r in sub if _out(r) in ("correct", "wrong")]
    return (sum(_out(r) == "wrong" for r in committed) / len(committed)) if committed else None


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2**n))


def main() -> None:
    refs = build_refs()
    lines = (FROZEN / "cove_arms.jsonl").read_text().splitlines()
    cove = [json.loads(line) for line in lines if line.strip()]
    graded = grade_cells(cove, refs, rule="final")
    (RESULTS / "cove_graded.json").write_text(json.dumps(graded))
    print("CoVe graded cells:", len(graded))
    for ct in SURFACES:
        e = err_commit(graded, ct, "cove")
        print(f"  {ct}: CoVe err|commit = {e:.2%}" if e is not None else f"  {ct}: no commits")


if __name__ == "__main__":
    main()
