"""
Central tolerance policy for quantitative verification (spec Section 1.2).

DESIGN RULE: every threshold here is justified from *published method-vs-
experiment error statistics*, NOT tuned on any evaluation set. A referee will
ask "how did you choose these?" and the answer must be a citation, not a fit.

A claim is flagged FAIL only when it lies outside the reference value by more
than (systematic method error + a margin), so that we do not penalise the model
for an error that is actually the fault of our own cheap reference computation.
Where an *experimental* reference is available (Tier 1.5), the tolerance is
tighter because there is no method error to absorb.

All thresholds are RELATIVE (fraction) unless suffixed _abs (absolute, in the
property's unit).
"""

from __future__ import annotations

# --- Dipole moment (Debye) -------------------------------------------------
# B3LYP/def2-TZVP dipoles vs. experiment: mean abs. error ~0.1-0.3 D, and for
# most small organics the relative error is <~15-20% (Hickey & Rowley,
# J. Phys. Chem. A 2014, 118, 3678; Verma & Truhlar benchmarking). We set a
# generous band so a *plausible* claim is not falsely flagged; the tier is
# meant to catch order-of-magnitude confabulations (e.g. "12.5 D" for acetone),
# not to referee the second decimal.
DIPOLE: dict[str, float] = dict(
    rel=0.30,  # 30% relative window for DFT-referenced checks
    abs=0.30,  # OR within 0.30 D absolute (whichever is looser)
    exp_rel=0.15,  # tighter when an EXPERIMENTAL value is the reference
    exp_abs=0.20,
)

# --- HOMO-LUMO / band gap (eV) --------------------------------------------
# B3LYP HOMO-LUMO gaps are not rigorous band gaps and run several tenths of an
# eV to >1 eV from experiment; hybrid-functional band gaps for solids scatter
# by ~0.5-1 eV (e.g. Garza & Scuseria, J. Phys. Chem. Lett. 2016). We therefore
# use a wide band and treat this tier as a coarse plausibility gate.
GAP: dict[str, float] = dict(
    rel=0.50,
    abs=1.0,  # within 1.0 eV absolute
    exp_rel=0.30,
    exp_abs=0.5,
)

# --- Formation energy per atom (eV/atom) ----------------------------------
# MP formation energies carry their own DFT error (~0.1-0.2 eV/atom vs.
# experiment for many classes; Kirklin et al., npj Comput. Mater. 2015).
FORMATION_ENERGY: dict[str, float] = dict(rel=0.25, abs=0.15)

# --- Lattice parameters (Angstrom) ----------------------------------------
# GGA/PBE lattice constants vs. experiment agree to ~1-2%; we allow 3% so that
# a claim quoting an experimental value is not flagged against a DFT-relaxed MP
# cell (Lejaeghere et al., Science 2016, delta-codes benchmarking).
LATTICE: dict[str, float] = dict(rel=0.03, abs=0.15)

# --- Metric self-consistency of a crystal cell ----------------------------
# Used by crystal.py for a=b=c / angle==90 style checks. This is a *symmetry*
# tolerance, not a physical-accuracy one: it only asks whether the numbers the
# model itself wrote obey the crystal system it itself named.
CELL_METRIC_REL = 1e-2


PROPERTY_TOL: dict[str, dict[str, float]] = {
    "dipole_moment": DIPOLE,
    "homo_lumo_gap": GAP,
    "band_gap": GAP,
    "formation_energy_per_atom": FORMATION_ENERGY,
}


def within_tolerance(
    prop: str, claimed: float, reference: float, experimental: bool = False
) -> tuple[bool | None, float | None, str]:
    """Return (ok: bool, rel_err: float, rule: str) for a quantitative claim."""
    tol = PROPERTY_TOL.get(prop)
    if tol is None:
        return None, None, f"no tolerance policy for '{prop}'"
    rel = abs(claimed - reference) / max(abs(reference), 1e-9)
    abs_err = abs(claimed - reference)
    if experimental and "exp_rel" in tol:
        rel_thr, abs_thr = tol["exp_rel"], tol.get("exp_abs", tol.get("abs"))
        src = "experimental"
    else:
        rel_thr, abs_thr = tol["rel"], tol.get("abs")
        src = "computed"
    ok = (rel <= rel_thr) or (abs_thr is not None and abs_err <= abs_thr)
    rule = f"{src} tol: rel<={rel_thr:.0%} or abs<={abs_thr}"
    return ok, rel, rule
