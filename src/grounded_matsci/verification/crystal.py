"""
Crystalline-materials verification tier (pymatgen / spglib).

Molecular verifiers (RDKit/PySCF) don't cover the objects that dominate a
*materials-science* reasoning trace: space groups, lattice parameters, Wyckoff
sites, crystal systems. This module adds checks for the confabulations that
show up there:

  - space-group SYMBOL <-> NUMBER mismatch          ("Fm-3m (#186)")
  - space group <-> CRYSTAL SYSTEM contradiction     ("cubic P6_3mc")
  - lattice PARAMETERS that violate the crystal system
        (e.g. a cubic cell with a != b, or alpha != 90)
  - space group recovered from an actual structure != the claimed one

These are pure-symmetry checks: cheap, deterministic, and independent of any
DFT. They are the crystalline analog of RDKit's valence check -- Tier 0/1 for
solids.
"""

from __future__ import annotations

import re
from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number
from pymatgen.core import Lattice


# --------------------------------------------------------------------------
# Space-group symbol <-> number <-> crystal system
# --------------------------------------------------------------------------
def _normalize_sg_symbol(sym):
    """Normalize a Hermann-Mauguin symbol: strip spaces, unify minus signs."""
    s = sym.strip().replace(" ", "").replace("\u2013", "-").replace("\u2212", "-")
    return s


def crystal_system_of(number):
    """International crystal system for a space-group NUMBER (1-230)."""
    n = int(number)
    ranges = [
        (1, 2, "triclinic"), (3, 15, "monoclinic"), (16, 74, "orthorhombic"),
        (75, 142, "tetragonal"), (143, 167, "trigonal"),
        (168, 194, "hexagonal"), (195, 230, "cubic"),
    ]
    for lo, hi, name in ranges:
        if lo <= n <= hi:
            return name
    return None


def verify_spacegroup_symbol_number(symbol, number):
    """Check that a claimed (symbol, number) pair is internally consistent."""
    try:
        canon = sg_symbol_from_int_number(int(number))
    except Exception as e:
        return "fail", f"invalid space-group number {number}: {e}"
    a = _normalize_sg_symbol(symbol)
    b = _normalize_sg_symbol(canon)
    # pymatgen SpaceGroup understands many symbol variants; compare via it
    try:
        same = SpaceGroup(symbol).int_number == int(number)
    except Exception:
        same = (a == b)
    if same:
        return "ok", f"symbol {symbol} matches space group #{number} ({canon})"
    return ("fail",
            f"MISMATCH: symbol {symbol} is not space group #{number}; "
            f"#{number} is {canon}")


def verify_spacegroup_system(symbol_or_number, claimed_system):
    """Check a claimed crystal system against the space group's true system."""
    try:
        if re.fullmatch(r"\d+", str(symbol_or_number).strip()):
            num = int(symbol_or_number)
        else:
            num = SpaceGroup(_normalize_sg_symbol(symbol_or_number)).int_number
    except Exception as e:
        return "unchecked", f"could not resolve space group '{symbol_or_number}': {e}"
    true_sys = crystal_system_of(num)
    cs = claimed_system.strip().lower()
    # pymatgen uses 'trigonal'; many texts say 'rhombohedral' for the lattice
    aliases = {"rhombohedral": "trigonal"}
    cs_norm = aliases.get(cs, cs)
    if cs_norm == true_sys:
        return "ok", f"crystal system '{claimed_system}' matches space group #{num}"
    return ("fail",
            f"MISMATCH: space group #{num} is {true_sys}, not '{claimed_system}'")


# --------------------------------------------------------------------------
# Lattice-parameter <-> crystal-system consistency
# --------------------------------------------------------------------------
def verify_lattice_system(a, b, c, alpha, beta, gamma, system, tol=1e-2):
    """Check that (a,b,c,alpha,beta,gamma) obey the metric constraints of the
    claimed crystal system. Returns (status, detail).
    """
    sys = system.strip().lower()
    sys = {"rhombohedral": "trigonal"}.get(sys, sys)

    def eq(x, y):
        return abs(x - y) <= tol * max(abs(x), abs(y), 1.0)

    ang90 = all(eq(x, 90.0) for x in (alpha, beta, gamma))
    checks = {
        "cubic":       (eq(a, b) and eq(b, c) and ang90,
                        "a=b=c and all angles 90"),
        "tetragonal":  (eq(a, b) and not eq(a, c) and ang90,
                        "a=b!=c and all angles 90"),
        "orthorhombic":(not eq(a, b) and not eq(b, c) and ang90,
                        "a!=b!=c and all angles 90"),
        "hexagonal":   (eq(a, b) and eq(alpha, 90) and eq(beta, 90) and eq(gamma, 120),
                        "a=b, alpha=beta=90, gamma=120"),
        "trigonal":    (eq(a, b) and eq(alpha, 90) and eq(beta, 90) and eq(gamma, 120)
                        or (eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma)),
                        "hexagonal setting (gamma=120) or rhombohedral (a=b=c, angles equal)"),
        "monoclinic":  (eq(alpha, 90) and eq(gamma, 90) and not eq(beta, 90),
                        "alpha=gamma=90, beta!=90"),
        "triclinic":   (True, "no metric constraint"),
    }
    if sys not in checks:
        return "unchecked", f"unknown crystal system '{system}'"
    ok, rule = checks[sys]
    params = f"a={a},b={b},c={c},alpha={alpha},beta={beta},gamma={gamma}"
    if ok:
        return "ok", f"lattice obeys {sys} constraints ({rule})"
    return "fail", f"lattice VIOLATES {sys} constraints ({rule}); given {params}"


# --------------------------------------------------------------------------
# Structure-level check: recover the space group from an actual structure
# --------------------------------------------------------------------------
def verify_structure_spacegroup(structure, claimed_number=None, claimed_symbol=None,
                                symprec=1e-2):
    """Given a pymatgen Structure, recover its space group and compare to a
    claim. Returns (status, detail, recovered_number).
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    num = sga.get_space_group_number()
    sym = sga.get_space_group_symbol()
    if claimed_number is not None and int(claimed_number) != num:
        return ("fail",
                f"structure has space group #{num} ({sym}), not claimed #{claimed_number}",
                num)
    if claimed_symbol is not None and _normalize_sg_symbol(claimed_symbol) != _normalize_sg_symbol(sym):
        return ("fail",
                f"structure has space group {sym} (#{num}), not claimed {claimed_symbol}",
                num)
    return "ok", f"structure space group confirmed: {sym} (#{num})", num
