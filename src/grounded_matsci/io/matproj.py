"""
Materials Project reference-data tier for crystalline claims.

This is the crystalline analog of the PubChem tier: instead of only checking a
claim for internal symmetry-consistency (verification/crystal.py), it compares
the claim to the *experimentally/DFT-refined reference* for that material in the
Materials Project database -- true space group, lattice constants, and band gap.

Requires an MP API key. The key is read from the environment variable
MP_API_KEY; it is never printed or written to any artifact. Without a key this
module degrades gracefully: `make_mp_lookup` returns None and the pipeline falls
back to the symmetry-only crystal tier.

Usage (analysis kernel):
    from grounded_matsci.io import matproj
    mp = matproj.make_mp_lookup()          # None if no key / mp-api missing
    ref = mp("TiO2")                       # -> {'spacegroup_number':136, ...}
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def _get_key() -> str | None:
    return os.environ.get("MP_API_KEY") or os.environ.get("MATERIALS_PROJECT_API_KEY")


def make_mp_lookup(prefer: str = "stable") -> Callable[[str], dict[str, Any] | None] | None:
    """Return lookup(formula) -> reference dict | None, or None if unavailable.

    The returned dict carries: formula_pretty, material_id, spacegroup_symbol,
    spacegroup_number, crystal_system, a/b/c/alpha/beta/gamma, band_gap,
    formation_energy_per_atom, is_stable.
    `prefer='stable'` picks the lowest energy-above-hull entry for the formula.
    """
    key = _get_key()
    if not key:
        return None
    try:
        from mp_api.client import MPRester
    except Exception:
        return None

    def lookup(formula: str) -> dict[str, Any] | None:
        try:
            with MPRester(key) as mpr:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "symmetry",
                        "structure",
                        "band_gap",
                        "formation_energy_per_atom",
                        "energy_above_hull",
                        "is_stable",
                    ],
                )
            if not docs:
                return None
            # choose the most stable polymorph
            docs = sorted(docs, key=lambda d: getattr(d, "energy_above_hull", 0) or 0)
            d = docs[0]
            sym = d.symmetry
            st = d.structure
            latt = st.lattice
            return {
                "formula_pretty": d.formula_pretty,
                "material_id": str(d.material_id),
                "spacegroup_symbol": sym.symbol if sym else None,
                "spacegroup_number": sym.number if sym else None,
                "crystal_system": str(sym.crystal_system) if sym else None,
                "a": latt.a,
                "b": latt.b,
                "c": latt.c,
                "alpha": latt.alpha,
                "beta": latt.beta,
                "gamma": latt.gamma,
                "band_gap": d.band_gap,
                "formation_energy_per_atom": d.formation_energy_per_atom,
                "is_stable": d.is_stable,
            }
        except Exception as e:
            return {"error": str(e)[:200]}

    return lookup


def verify_against_mp(
    mp_lookup: Callable[[str], dict[str, Any] | None] | None,
    formula: str,
    claimed_sg_number: int | None = None,
    claimed_band_gap: float | None = None,
    gap_tol: float = 0.5,
) -> tuple[str, str]:
    """Compare a claimed space group / band gap to the MP reference."""
    if mp_lookup is None:
        return "unchecked", "no Materials Project key configured"
    ref = mp_lookup(formula)
    if ref is None:
        return "unchecked", f"'{formula}' not found in Materials Project"
    if "error" in ref:
        return "unchecked", f"MP query failed: {ref['error']}"
    msgs = []
    verdict = "ok"
    if claimed_sg_number is not None and ref["spacegroup_number"] is not None:
        if int(claimed_sg_number) != int(ref["spacegroup_number"]):
            verdict = "fail"
            msgs.append(
                f"space group #{claimed_sg_number} != MP #{ref['spacegroup_number']} "
                f"({ref['spacegroup_symbol']}, {ref['material_id']})"
            )
        else:
            msgs.append(f"space group #{claimed_sg_number} matches MP ({ref['material_id']})")
    if claimed_band_gap is not None and ref["band_gap"] is not None:
        rel = abs(claimed_band_gap - ref["band_gap"]) / max(ref["band_gap"], 1e-6)
        if rel > gap_tol:
            verdict = "fail"
        msgs.append(
            f"band gap {claimed_band_gap} eV vs MP {ref['band_gap']:.2f} eV (rel.err {rel:.0%})"
        )
    return verdict, "; ".join(msgs) if msgs else "no comparable fields"
