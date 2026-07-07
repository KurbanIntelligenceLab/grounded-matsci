"""
Supplementary grader for the round-7 expansion claim types (option b).

The frozen extractor (sha 42e03ae2, immutable for the holdout) covers formula,
space group, band gap, and dipole. It does NOT extract (a) formation-energy
numerics with eV/atom units or (b) crystal-system words. Per the frozen-extractor
rule, we add these at the VERIFICATION/grading layer, not by editing extract.py.

Each grader takes the raw model text + subject + frozen GT entry and returns:
    {"extracted": <value or None>, "gt": <value>, "correct": bool|None, "reason": str}
`correct=None` means the model did not commit a checkable value (no-answer), which is
scored as "no commitment", distinct from an error (governance: never count a missing
extraction as a wrong answer).
"""

from __future__ import annotations

import re

# --- crystal system reasoning ---
_SYSTEMS = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
    "rhombohedral",
]


def grade_crystal_system(text, subject, gt):
    t = text.lower()
    # rhombohedral is the trigonal setting; accept as trigonal
    found = [s for s in _SYSTEMS if re.search(r"\b" + s + r"\b", t)]
    norm = {"rhombohedral": "trigonal"}
    found_norm = [norm.get(s, s) for s in found]
    gtsys = gt["crystal_system"]
    if not found_norm:
        return {
            "extracted": None,
            "gt": gtsys,
            "correct": None,
            "reason": "no crystal system named",
        }
    # the model's COMMITTED system = the one it states for the subject; take the last distinct
    # (models often list systems in reasoning then commit at the end)
    committed = found_norm[-1]
    return {
        "extracted": committed,
        "gt": gtsys,
        "correct": committed == gtsys,
        "reason": f"committed {committed} vs GT {gtsys}",
    }


# --- formation energy (eV/atom) ---
_EF = re.compile(r"(-?\d+\.?\d*)\s*(?:eV\s*/\s*atom|eV\s*per\s*atom)", re.I)


def grade_formation_energy(text, subject, gt, tol):
    vals = [float(m.group(1)) for m in _EF.finditer(text)]
    if not vals:
        return {"extracted": None, "gt": gt["value"], "correct": None, "reason": "no eV/atom value"}
    g = gt["value"]
    band = max(tol["abs"], tol["rel"] * abs(g))
    if any(abs(v - g) <= band for v in vals):
        return {"extracted": g, "gt": g, "correct": True, "reason": "a stated value within tol"}
    v = vals[-1]
    return {
        "extracted": v,
        "gt": g,
        "correct": False,
        "reason": f"no stated value within tol; committed {v} vs {g} eV/atom",
    }


# --- band gap (eV) : accept metal/insulator qualitative + numeric ---
_GAP = re.compile(r"(-?\d+\.?\d*)\s*eV", re.I)


def grade_band_gap(text, subject, gt, tol):
    g = gt["value"]
    t = text.lower()
    # metallic claim
    if g < 0.1 and re.search(r"\bmetal(lic)?\b", t) and not re.search(r"non-?metal", t):
        return {"extracted": "metal", "gt": g, "correct": True, "reason": "metallic, GT gap<0.1"}
    vals = [float(m.group(1)) for m in _GAP.finditer(text)]
    if not vals:
        return {"extracted": None, "gt": g, "correct": None, "reason": "no eV value"}
    band = max(tol["abs"], tol["rel"] * abs(g))
    if any(abs(v - g) <= band for v in vals):
        return {"extracted": g, "gt": g, "correct": True, "reason": "a stated value within tol"}
    v = vals[-1]
    return {
        "extracted": v,
        "gt": g,
        "correct": False,
        "reason": f"no stated value within tol; committed {v} vs {g} eV",
    }


# --- dipole (debye) ---
_D = re.compile(r"(-?\d+\.?\d*)\s*(?:D\b|debye)", re.I)


def grade_dipole(text, subject, gt, tol):
    g = gt["value"]
    vals = [float(m.group(1)) for m in _D.finditer(text)]
    if not vals:
        return {"extracted": None, "gt": g, "correct": None, "reason": "no debye value"}
    er, ea = tol.get("exp_rel", tol["rel"]), tol.get("exp_abs", tol["abs"])
    band = max(ea, er * abs(g))
    if any(abs(v - g) <= band for v in vals):
        return {"extracted": g, "gt": g, "correct": True, "reason": "a stated value within exp tol"}
    v = vals[-1]
    return {
        "extracted": v,
        "gt": g,
        "correct": False,
        "reason": f"no stated value within tol; committed {v} vs {g} D",
    }


def grade(text, subject, claim_type, gt_entry, tolerances):
    if claim_type == "reasoning_system":
        return grade_crystal_system(text, subject, gt_entry)
    if claim_type == "property_ef":
        return grade_formation_energy(text, subject, gt_entry, tolerances["formation_energy"])
    if claim_type == "property_gap":
        return grade_band_gap(text, subject, gt_entry, tolerances["band_gap"])
    if claim_type == "property_dipole":
        return grade_dipole(text, subject, gt_entry, tolerances["dipole"])
    raise ValueError(claim_type)
