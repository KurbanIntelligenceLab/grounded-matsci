"""Regression tests for the round-7 expansion grader. Assertions moved unchanged from
the bundle's `tests/test_regression.py`."""

from grounded_matsci.domain import tolerances
from grounded_matsci.evaluation import grade_expansion


def test_expansion_grader_claim_types():
    """Round-7 expansion: grade formation energy, band gap, dipole, crystal system."""
    tol = {
        "formation_energy": tolerances.FORMATION_ENERGY,
        "band_gap": tolerances.GAP,
        "dipole": tolerances.DIPOLE,
    }
    # crystal system: correct commitment
    r = grade_expansion.grade(
        "NaCl is cubic.", "NaCl", "reasoning_system", {"crystal_system": "cubic"}, tol
    )
    assert r["correct"] is True
    # crystal system: wrong commitment
    r = grade_expansion.grade(
        "NaCl is hexagonal.", "NaCl", "reasoning_system", {"crystal_system": "cubic"}, tol
    )
    assert r["correct"] is False
    # crystal system: no commitment -> None, not an error
    r = grade_expansion.grade(
        "NaCl is a salt.", "NaCl", "reasoning_system", {"crystal_system": "cubic"}, tol
    )
    assert r["correct"] is None
    # formation energy within tol
    r = grade_expansion.grade("-2.05 eV/atom", "NaCl", "property_ef", {"value": -2.1}, tol)
    assert r["correct"] is True
    # dipole uses experimental tolerance
    r = grade_expansion.grade("1.80 D", "water", "property_dipole", {"value": 1.85}, tol)
    assert r["correct"] is True
    # metallic band gap
    r = grade_expansion.grade("It is metallic.", "Cu", "property_gap", {"value": 0.0}, tol)
    assert r["correct"] is True
