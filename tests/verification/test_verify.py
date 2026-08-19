"""Regression tests for the SMILES/formula verifier tiers. Assertions moved unchanged
from the pre-refactor regression suite."""

from grounded_matsci.verification import verify


# ---------- Tier 0: SMILES syntactic / valence ----------
def test_hypervalent_nitrogen_fails():
    st, detail, *_ = verify.verify_smiles("c1ccccc1N(=O)(=O)(=O)")
    assert st == "fail" and "valence" in detail.lower()


def test_five_bond_carbon_fails():
    st, *_ = verify.verify_smiles("C(C)(C)(C)(C)C")
    assert st == "fail"


def test_valid_smiles_pass():
    for smi in ["c1ccccc1", "Oc1ccccc1", "CCO", "c1ccncc1", "c1ccccc1[N+](=O)[O-]"]:
        st, *_ = verify.verify_smiles(smi)
        assert st == "ok", smi


# ---------- Tier 1: formula identity ----------
def test_wrong_formula_fails():
    st, detail = verify.verify_formula("caffeine", "C8H10N4O3")
    assert st == "fail" and "C8H10N4O2" in detail


def test_correct_formula_pass():
    st, _ = verify.verify_formula("caffeine", "C8H10N4O2")
    assert st == "ok"
