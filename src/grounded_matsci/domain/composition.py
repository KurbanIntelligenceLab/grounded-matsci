"""Molar mass from a verified molecular formula (the derived-quantity tier).

Molar mass is computed from the verified molecular formula by summing standard IUPAC
atomic weights (the _ATOMIC table below); no SMILES parsing is needed since the verified
object is already the formula. Extracted from the derived-quantity end-task runner
(now `workflows/endtask_derived.py`) unchanged."""

from __future__ import annotations

import re

_ATOMIC: dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.90,
    "Na": 22.990,
    "K": 39.098,
    "Ca": 40.078,
    "Mg": 24.305,
    "Fe": 55.845,
    "Zn": 65.38,
    "Al": 26.982,
    "Si": 28.085,
    "Ti": 47.867,
    "Sr": 87.62,
    "Pb": 207.2,
    "B": 10.81,
    "Se": 78.97,
}


def molar_mass_from_formula(formula: str) -> float | None:
    """Derived quantity from the VERIFIED formula: sum of standard atomic weights (_ATOMIC)."""
    toks = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    m = 0.0
    for el, n in toks:
        if not el:
            continue
        if el not in _ATOMIC:
            return None
        m += _ATOMIC[el] * (int(n) if n else 1)
    return round(m, 2)
