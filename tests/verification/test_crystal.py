"""Regression tests for the crystalline symmetry tier. Assertions moved unchanged from
the bundle's `tests/test_regression.py`."""

from grounded_matsci.verification import crystal


# ---------- space-group symbol/number and system ----------
def test_spacegroup_symbol_number_mismatch():
    st, detail = crystal.verify_spacegroup_symbol_number("Fm-3m", 186)
    assert st == "fail" and "P6_3mc" in detail


def test_spacegroup_symbol_number_match():
    st, _ = crystal.verify_spacegroup_symbol_number("Fm-3m", 225)
    assert st == "ok"


def test_spacegroup_system_contradiction():
    st, _ = crystal.verify_spacegroup_system("P6_3mc", "cubic")
    assert st == "fail"


# ---------- lattice / crystal-system metric consistency ----------
def test_cubic_lattice_violation():
    st, _ = crystal.verify_lattice_system(4.59, 4.59, 2.96, 90, 90, 90, "cubic")
    assert st == "fail"


def test_cubic_lattice_ok():
    st, _ = crystal.verify_lattice_system(5.64, 5.64, 5.64, 90, 90, 90, "cubic")
    assert st == "ok"
