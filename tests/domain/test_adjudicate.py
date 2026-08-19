"""Regression tests for the frozen adjudicator (frame policy). Assertions moved unchanged
from the pre-refactor regression suite."""

from grounded_matsci.domain import adjudicate, tolerances


def test_frozen_adjudicator_frame_policy():
    """Post-audit: band-gap accepts PBE or experimental frame; ef normalizes per-fu; rules."""
    G, EF, D = tolerances.GAP, tolerances.FORMATION_ENERGY, tolerances.DIPOLE
    # band gap: model gives experimental value, PBE GT lower -> correct via experimental frame
    r = adjudicate.grade_band_gap("The band gap is 3.37 eV.", {"value": 0.73}, G, exp_gap=3.37)
    assert r["correct"] is True and r["frame"] == "experimental"
    # band gap: matches PBE directly
    r = adjudicate.grade_band_gap("about 5.0 eV", {"value": 5.0}, G, exp_gap=8.5)
    assert r["correct"] is True and r["frame"] == "PBE"
    # band gap: matches neither -> wrong
    r = adjudicate.grade_band_gap("50 eV", {"value": 5.0}, G, exp_gap=8.5)
    assert r["correct"] is False
    # formation energy per-formula-unit normalization
    r = adjudicate.grade_formation_energy("-6.08 eV/atom", {"value": -3.04}, EF, n_atoms=2)
    assert r["correct"] is True and r["frame"] == "per_fu_normalized"
    # final-commitment: last value graded
    r = adjudicate.grade_formation_energy(
        "first -9.0 eV/atom then -2.10 eV/atom", {"value": -2.1}, EF, rule="final"
    )
    assert r["correct"] is True
    # crystal system final vs any
    r = adjudicate.grade_crystal_system(
        "cubic then hexagonal", {"crystal_system": "cubic"}, rule="final"
    )
    assert r["correct"] is False
    r = adjudicate.grade_crystal_system(
        "cubic then hexagonal", {"crystal_system": "cubic"}, rule="any"
    )
    assert r["correct"] is True


def test_ef_either_frame():
    """Formation energy accepts MP or experimental frame; drop path applied for
    subjects with no clean value in the claimed (MP) frame, empty in practice since
    MP covers every crystalline subject."""
    EF = tolerances.FORMATION_ENERGY
    # model gives experimental value; MP GT differs but within exp tol
    r = adjudicate.grade_formation_energy("-0.52 eV/atom", {"value": -0.784}, EF, exp_ef=-0.520)
    assert r["correct"] is True and r["frame"] == "experimental"
    # matches MP directly
    r = adjudicate.grade_formation_energy("-3.43 eV/atom", {"value": -3.427}, EF, exp_ef=-3.473)
    assert r["correct"] is True and r["frame"] == "MP"
    # genuinely wrong: misses both frames
    r = adjudicate.grade_formation_energy("-9.45 eV/atom", {"value": -3.427}, EF, exp_ef=-3.473)
    assert r["correct"] is False
