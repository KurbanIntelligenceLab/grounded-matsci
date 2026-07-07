"""
Tier 1.5: experimental / tabulated reference data (spec Section 1.2).

Escalation puts a *database lookup before DFT*: if we already know the
experimental dipole moment or the Materials Project band gap, that is a better
reference than a cheap DFT number and it is essentially free. Only claims with
no tabulated reference fall through to Tier 2/3 physics.

Two sources:
  - MOLECULAR experimental dipoles: a curated table sourced from NIST CCCBDB
    (https://cccbdb.nist.gov/, experimental dipole compilation) and standard
    physical-chemistry references. Each entry carries a provenance string.
    CCCBDB has no bulk REST API, so values are curated rather than fetched;
    every value is a published experimental gas-phase dipole.
  - CRYSTALLINE properties: delegated to the live Materials Project tier
    (io/matproj.py) at call time.

This table is deliberately small and auditable. For the paper it would be
expanded and released with per-value citations (spec Section 5).
"""

from __future__ import annotations

# name -> (experimental dipole moment in Debye, provenance)
# Gas-phase experimental dipole moments. Sources: NIST CCCBDB experimental
# dipole compilation; CRC Handbook of Chemistry and Physics.
EXPERIMENTAL_DIPOLE: dict[str, tuple[float, str]] = {
    "water": (1.85, "NIST CCCBDB; CRC Handbook"),
    "ammonia": (1.47, "NIST CCCBDB"),
    "methane": (0.00, "symmetry (Td); NIST CCCBDB"),
    "carbon dioxide": (0.00, "symmetry (linear D∞h)"),
    "acetone": (2.88, "NIST CCCBDB"),
    "acetonitrile": (3.92, "NIST CCCBDB"),
    "methanol": (1.70, "NIST CCCBDB"),
    "ethanol": (1.69, "NIST CCCBDB"),
    "formaldehyde": (2.33, "NIST CCCBDB"),
    "benzene": (0.00, "symmetry (D6h); NIST CCCBDB"),
    "toluene": (0.36, "NIST CCCBDB"),
    "phenol": (1.22, "NIST CCCBDB"),
    "aniline": (1.53, "NIST CCCBDB"),
    "pyridine": (2.22, "NIST CCCBDB"),
    "nitrobenzene": (4.22, "NIST CCCBDB; CRC Handbook"),
    "chlorobenzene": (1.69, "NIST CCCBDB"),
    "furan": (0.66, "NIST CCCBDB"),
    "chloroform": (1.04, "NIST CCCBDB"),
    "dimethyl ether": (1.30, "NIST CCCBDB"),
    "hydrogen fluoride": (1.82, "NIST CCCBDB"),
}


def experimental_dipole(name: str) -> tuple[float, str] | None:
    """Return (value_debye, provenance) or None."""
    rec = EXPERIMENTAL_DIPOLE.get(name.strip().lower())
    return rec if rec else None


def reference_property(name: str, prop: str) -> tuple[float, str, str] | None:
    """Tier 1.5 molecular lookup. Returns (value, unit, provenance) or None.

    Crystalline properties are handled by io/matproj.py, not here.
    """
    if prop == "dipole_moment":
        rec = experimental_dipole(name)
        if rec:
            return rec[0], "debye", rec[1]
    return None
