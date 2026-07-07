"""
Unified grader: score a generated trace against the frozen holdout GT across all six
claim types, using the FROZEN adjudication procedure (domain/adjudicate.py). The molecular
and crystalline paths use the frozen extractor + named-phase rule; property/reasoning paths
use adjudicate.py with the reference-frame policy.

`rule`: "final" (final-commitment; baseline + non-correction conditions) or
        "any"   (any-match; post-correction Mode A/B only).
"""

from __future__ import annotations

import re
from collections import Counter

from grounded_matsci.domain import adjudicate


def _hill(f):
    tk = re.findall(r"([A-Z][a-z]?)(\d*)", f)
    c = Counter()
    for el, n in tk:
        if el:
            c[el] += int(n) if n else 1
    order = [e for e in ("C", "H") if e in c] + sorted(e for e in c if e not in ("C", "H"))
    return "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in order)


def _natoms(formula):
    return sum(int(n) if n else 1 for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if el)


def grade_record(
    rec,
    gt,
    tol,
    extract_mod,
    ground_mod,
    named_sg_lookup,
    known_names,
    exp_gap_ref=None,
    rule="final",
    exp_ef_ref=None,
):
    ct = rec["claim_type"]
    subj = rec["subject"]
    text = rec["text"]
    exp_gap_ref = exp_gap_ref or {}
    if ct == "molecular":
        g = gt["molecular"].get(subj, {})
        tf = g.get("formula")
        claims = [
            c
            for c in extract_mod.extract_all(text, known_names=[subj.lower(), *known_names])
            if c.kind == "formula"
        ]
        if not tf:
            return {"correct": None, "reason": "no GT formula"}
        if not claims:
            return {"correct": None, "reason": "no formula extracted"}
        formulas = [c.payload["formula"] for c in claims]
        th = _hill(tf)
        if rule == "any":
            if any(_hill(f) == th for f in formulas):
                return {"correct": True, "extracted": tf, "gt": tf}
        else:  # final-commitment: the subject-bound formula, else the last stated
            bound = [
                c.payload["formula"]
                for c in claims
                if c.payload.get("name", "").lower() == subj.lower()
            ]
            final = (bound or formulas)[-1]
            if _hill(final) == th:
                return {"correct": True, "extracted": tf, "gt": tf}
            return {"correct": False, "extracted": final, "gt": tf}
        subj_bound = [
            c.payload["formula"]
            for c in claims
            if c.payload.get("name", "").lower() == subj.lower()
        ]
        committed = max((subj_bound or formulas), key=lambda f: len(_hill(f)))
        return {"correct": False, "extracted": committed, "gt": tf}
    if ct == "crystalline":
        g = gt["crystalline"].get(subj, {})
        acc = set(g.get("accepted_sg") or [])
        got = None
        for c in ground_mod.ground_trace(
            text,
            enable_physics=False,
            extra_known_names=[subj.lower(), *known_names],
            named_sg_lookup=named_sg_lookup,
        ):
            if c.kind == "sg_number":
                got = c.payload.get("number")
                break
        if got is None:
            return {"correct": None, "reason": "no sg extracted"}
        return {"correct": got in acc, "extracted": got, "gt": sorted(acc)}
    if ct == "property_ef":
        g = gt["property"].get(f"{subj}::ef")
        if not g:
            return {"correct": None, "reason": "no GT"}
        na = _natoms(gt["_formula"].get(subj, "")) if "_formula" in gt else 1
        eef = (exp_ef_ref or {}).get(subj, {}).get("exp_ef_eV_atom")
        return adjudicate.grade_formation_energy(
            text, g, tol["formation_energy"], n_atoms=na, rule=rule, exp_ef=eef
        )
    if ct == "property_gap":
        g = gt["property"].get(f"{subj}::gap")
        if not g:
            return {"correct": None, "reason": "no GT"}
        eg = exp_gap_ref.get(subj, {}).get("exp_gap")
        if eg is None:  # no experimental frame available -> DROPPED
            return {
                "correct": None,
                "reason": "dropped: no experimental gap reference (PBE-only)",
                "dropped": True,
            }
        return adjudicate.grade_band_gap(text, g, tol["band_gap"], exp_gap=eg, rule=rule)
    if ct == "property_dipole":
        g = gt["property"].get(f"{subj}::dipole")
        if not g:
            return {"correct": None, "reason": "no GT"}
        return adjudicate.grade_dipole(text, g, tol["dipole"], rule=rule)
    if ct == "reasoning_system":
        g = gt["reasoning_system"].get(subj)
        if not g:
            return {"correct": None, "reason": "no GT"}
        return adjudicate.grade_crystal_system(text, g, rule=rule)
    return {"correct": None, "reason": f"unknown claim_type {ct}"}
