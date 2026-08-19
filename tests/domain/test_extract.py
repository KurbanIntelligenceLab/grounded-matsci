"""Regression tests for the frozen extractor (planted-error suite).

Assertions are moved unchanged from the pre-refactor regression suite, including
the historical benzene/nitrobenzene name-binding bug (Section 4)."""

from grounded_matsci.domain import extract


# ---------- name binding: the benzene/nitrobenzene bug ----------
def test_nitrobenzene_not_bound_to_benzene():
    """A nitrobenzene dipole claim must NOT be checked against benzene (mu=0)."""
    text = "nitrobenzene has a dipole moment of 4.2 D"
    claims = extract.extract_properties(text, known_names=["nitrobenzene", "benzene"])
    assert len(claims) == 1
    assert claims[0].payload["name"] == "nitrobenzene", (
        f"bound to {claims[0].payload['name']} instead of nitrobenzene"
    )


def test_benzene_still_binds_when_alone():
    text = "benzene has a dipole moment of 0.0 D"
    claims = extract.extract_properties(text, known_names=["nitrobenzene", "benzene"])
    assert claims[0].payload["name"] == "benzene"


# ---------- extractor does not hallucinate space groups from formulas ----------
def test_formula_not_read_as_spacegroup():
    claims = extract.extract_crystal("caffeine has formula C8H10N4O2")
    assert not any(c.kind == "sg_number" for c in claims)


def test_spacegroup_symbols_not_read_as_smiles():
    for sym in ["Fm-3m", "Fd-3m"]:
        assert extract.extract_smiles(sym) == []


def test_smiles_string_not_parsed_as_formula():
    """A SMILES string labeled as SMILES must not be extracted as a formula
    (exposed on the stratified pilot; sha 42e03ae2 fix)."""
    for t in [
        "The SMILES of naproxen is CC(C)OC1=CC=C(C=C1)",
        "SMILES of ibuprofen: CC(C)Cc1ccc(cc1)",
    ]:
        fs = [
            c
            for c in extract.extract_all(t, known_names=["naproxen", "ibuprofen"])
            if c.kind == "formula"
        ]
        assert not fs, f"SMILES parsed as formula: {[c.payload for c in fs]}"
    # but a real formula still extracts
    assert any(
        c.kind == "formula" and c.payload["formula"] == "C9H8O4"
        for c in extract.extract_all("aspirin is C9H8O4", known_names=["aspirin"])
    )
