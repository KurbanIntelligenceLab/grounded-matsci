"""
Frozen adjudication procedure (round-7, post-audit). ONE canonical grader for the
holdout. Freeze this file's sha before grading any intervention trace.

Two grading rules are supported and BOTH reported (sensitivity analysis):
  - FINAL-COMMITMENT ("final"): grade the LAST stated value/answer. Used for baseline
    and all non-correction conditions (unguarded, self-critique, RAG-in-prompt).
  - ANY-MATCH ("any"): correct if ANY stated value is within tolerance. Used ONLY for
    post-correction outputs (Mode A/B), where adopting the corrected value is the
    measured behavior.

Reference-frame policy (band gap): a claim is correct if the final value is within the
registered tolerance of the PBE reference (MP) OR the experimental reference (where a
well-established value exists). MP-PBE systematically underestimates experiment, so
grading against PBE alone spuriously fails correct experimental answers. Subjects with no
clean experimental gap and only a PBE value are DROPPED from the band-gap cell (reported
as dropped), never graded against a frame the model was not asked for.

Formation energy: GT is per-atom (eV/atom). A value stated as per-formula-unit is a unit
convention, not a confabulation; the grader normalizes by atoms-per-formula-unit when the
per-atom reading fails but per-fu matches, and records the normalization.
"""

from __future__ import annotations

import re
from typing import Any

_NUM = r"(-?\d+\.?\d*)"
_EF = re.compile(_NUM + r"\s*(?:eV\s*/\s*atom|eV\s*per\s*atom)", re.I)
_EFU = re.compile(_NUM + r"\s*(?:eV\s*/\s*(?:f\.?u\.?|formula)|eV\s*per\s*formula)", re.I)
_GAP = re.compile(_NUM + r"\s*eV", re.I)
_D = re.compile(_NUM + r"\s*(?:D\b|debye)", re.I)
_SYS = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
    "rhombohedral",
]


def _pick(vals: list[float], rule: str) -> list[float]:
    """rule 'final' -> last value; 'any' -> list for any-match."""
    return vals[-1:] if rule == "final" else vals


def grade_formation_energy(
    text: str,
    gt: dict[str, Any],
    tol: dict[str, float],
    n_atoms: int = 1,
    rule: str = "final",
    exp_ef: float | None = None,
) -> dict[str, Any]:
    """Formation energy per atom (eV/atom). Reference-frame policy (symmetric with band gap):
    a value is correct if within tolerance of the MP DFT reference OR an experimental ΔHf
    reference (298 K, converted to eV/atom), when a clean experimental value exists. Per the
    team frame-policy instruction, subjects with no clean value in the claimed frame are
    dropped; the claimed frame here is MP DFT, which provides a valid formation energy for
    every crystalline subject (unlike PBE band gaps, which are systematically invalid and
    forced drops). The drop set for formation energy is therefore empty in practice — but the
    drop path is applied, not assumed away. The experimental frame only rescues borderline
    cases where DFT deviates from experiment (mean |exp-MP| ~ 0.11 eV/atom)."""
    g = gt.get("value")
    if g is None:  # no clean MP value in the claimed frame -> drop (per frame policy)
        return {
            "extracted": None,
            "gt": None,
            "correct": None,
            "dropped": True,
            "reason": "no clean value in claimed (MP) frame",
        }
    band = max(tol["abs"], tol["rel"] * abs(g))
    vals = [float(m.group(1)) for m in _EF.finditer(text)]
    if not vals:
        return {"extracted": None, "gt": g, "correct": None, "reason": "no eV/atom value"}
    cand = _pick(vals, rule)
    if any(abs(v - g) <= band for v in cand):
        return {
            "extracted": g,
            "gt": g,
            "correct": True,
            "reason": "per-atom value within MP tol",
            "frame": "MP",
        }
    # experimental frame (if a clean exp ΔHf exists for this subject)
    if exp_ef is not None:
        eb = max(tol["abs"], tol["rel"] * abs(exp_ef))
        if any(abs(v - exp_ef) <= eb for v in cand):
            return {
                "extracted": cand[-1],
                "gt": exp_ef,
                "correct": True,
                "reason": "per-atom value within experimental tol",
                "frame": "experimental",
            }
    # per-formula-unit fallback: normalize by n_atoms
    if n_atoms > 1 and any(abs(v / n_atoms - g) <= band for v in cand):
        return {
            "extracted": round(cand[-1] / n_atoms, 3),
            "gt": g,
            "correct": True,
            "reason": f"per-formula-unit value /{n_atoms} within tol",
            "frame": "per_fu_normalized",
        }
    return {
        "extracted": cand[-1],
        "gt": g,
        "correct": False,
        "reason": f"{cand[-1]} vs {g} eV/atom, no frame matches",
    }


def grade_band_gap(
    text: str,
    gt: dict[str, Any],
    tol: dict[str, float],
    exp_gap: float | None = None,
    rule: str = "final",
) -> dict[str, Any]:
    g = gt["value"]
    t = text.lower()
    if g < 0.1 and re.search(r"\bmetal(lic)?\b", t) and not re.search(r"non-?metal", t):
        return {
            "extracted": "metal",
            "gt": g,
            "correct": True,
            "reason": "metallic, GT<0.1",
            "frame": "qualitative",
        }
    vals = [float(m.group(1)) for m in _GAP.finditer(text)]
    if not vals:
        return {"extracted": None, "gt": g, "correct": None, "reason": "no eV value"}
    cand = _pick(vals, rule)
    band_pbe = max(tol["abs"], tol["rel"] * abs(g))
    if any(abs(v - g) <= band_pbe for v in cand):
        return {
            "extracted": cand[-1],
            "gt": g,
            "correct": True,
            "reason": "within PBE tol",
            "frame": "PBE",
        }
    if exp_gap is not None:
        band_exp = max(
            tol.get("exp_abs", tol["abs"]), tol.get("exp_rel", tol["rel"]) * abs(exp_gap)
        )
        if any(abs(v - exp_gap) <= band_exp for v in cand):
            return {
                "extracted": cand[-1],
                "gt": exp_gap,
                "correct": True,
                "reason": "within experimental tol",
                "frame": "experimental",
            }
    return {
        "extracted": cand[-1],
        "gt": g,
        "correct": False,
        "reason": f"{cand[-1]} matches neither PBE {g} nor exp {exp_gap}",
        "frame": "none",
    }


def grade_dipole(
    text: str, gt: dict[str, Any], tol: dict[str, float], rule: str = "final"
) -> dict[str, Any]:
    g = gt["value"]
    vals = [float(m.group(1)) for m in _D.finditer(text)]
    if not vals:
        return {"extracted": None, "gt": g, "correct": None, "reason": "no debye value"}
    cand = _pick(vals, rule)
    er, ea = tol.get("exp_rel", tol["rel"]), tol.get("exp_abs", tol["abs"])
    band = max(ea, er * abs(g))
    if any(abs(v - g) <= band for v in cand):
        return {"extracted": g, "gt": g, "correct": True, "reason": "within exp tol"}
    return {"extracted": cand[-1], "gt": g, "correct": False, "reason": f"{cand[-1]} vs {g} D"}


def grade_crystal_system(text: str, gt: dict[str, Any], rule: str = "final") -> dict[str, Any]:
    t = text.lower()
    # collect (text_offset, system) so final-commitment uses TEXT order, not list order
    hits: list[tuple[int, str]] = []
    for s in _SYS:
        for m in re.finditer(r"\b" + s + r"\b", t):
            hits.append((m.start(), s))
    hits.sort()
    norm = {"rhombohedral": "trigonal"}
    found_norm = [norm.get(s, s) for _, s in hits]
    gtsys = gt["crystal_system"]
    if not found_norm:
        return {
            "extracted": None,
            "gt": gtsys,
            "correct": None,
            "reason": "no crystal system named",
        }
    if rule == "any":
        return {
            "extracted": found_norm,
            "gt": gtsys,
            "correct": (gtsys in found_norm),
            "reason": "any-match",
        }
    committed = found_norm[-1]  # last-mentioned in text = the committed answer
    return {
        "extracted": committed,
        "gt": gtsys,
        "correct": committed == gtsys,
        "reason": f"final {committed} vs {gtsys}",
    }
