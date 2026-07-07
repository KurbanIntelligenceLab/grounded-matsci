"""
Grounded-reasoning orchestrator.

Takes a raw LLM reasoning trace, extracts every chemical claim, runs the
tiered verifiers, and produces:

  1. an annotated trace (each claim tagged OK / FAIL / WARN / UNCHECKED)
  2. a structured audit table
  3. a trust score

This is the "verifier-in-the-loop" that turns a hallucination-prone trace
into a checked one. In a live system the same functions would run *during*
generation (tool-call interleaving); here we run them as a post-hoc pass so
the mechanism is easy to inspect.
"""

from __future__ import annotations

import re as _re

from grounded_matsci.domain import extract

from . import verify

try:
    from . import crystal
except Exception:            # pymatgen optional
    crystal = None


SYMBOL = {"ok": "OK  ", "fail": "FAIL", "warn": "WARN", "unchecked": "?   "}


# cost class per tier, for the cost-accounting figure (spec Section 3.3)
TIER_COST = {"0": "free", "1": "api", "1.5": "table", "2": "cheap", "3": "dft",
             None: "none"}


# Orchestration-layer formation-energy detection tier (spec Tier 1.5).
# The FROZEN extractor does not emit formation-energy claims (only dipole/gap), so wrong
# formation energies were invisible to the in-loop verifier (6% detection recall). This tier
# extracts formation-energy values at the ORCHESTRATION layer, binds the nearest named
# material, and checks against MP-tabulated values under the either-frame policy (MP DFT OR
# experimental ΔHf within tolerance). Frozen extractor is untouched, exactly as the named-SG
# tier. ef_lookup: {subject_lower -> {"mp": eV/atom, "exp": eV/atom or None, "n_atoms": int}}.
_EF_VALUE = _re.compile(r"(-?\d+\.?\d*)\s*(?:eV\s*/\s*atom|eV\s*per\s*atom)", _re.IGNORECASE)


def detect_formation_energy(text, ef_lookup, tol):
    """Return synthetic formation-energy Claims (orchestration tier). Uses the SAME strict
    'eV/atom'-qualified extraction as the adjudicator (adjudicate._EF) so detection and
    grading agree by construction: only explicitly per-atom values are candidates (bare 'eV'
    and kJ/mol intermediates are ignored). The FINAL such value in text (final-commitment) is
    bound to the nearest known material name and checked either-frame (MP DFT OR experimental
    ΔHf within tolerance). Frozen extractor is untouched."""
    from grounded_matsci.domain import extract as _ex
    out = []
    names = list(ef_lookup)
    # collect (offset, value) for every explicit eV/atom value, bound to nearest material
    per_mat = {}
    for vm in _EF_VALUE.finditer(text):
        mat = _ex._nearest_known_name(text, vm.start(), names)
        if mat is None:
            continue
        per_mat.setdefault(mat, []).append((vm.start(), float(vm.group(1))))
    for mat, vals in per_mat.items():
        ref = ef_lookup.get(mat) or ef_lookup.get(mat.lower())
        if not ref:
            continue
        vals.sort()
        off, val = vals[-1]  # final committed eV/atom value
        g = ref["mp"]
        na = ref.get("n_atoms", 1)
        exp = ref.get("exp")
        band = max(tol["abs"], tol["rel"] * abs(g))
        ok = abs(val - g) <= band
        if not ok and exp is not None:
            ok = abs(val - exp) <= max(tol["abs"], tol["rel"] * abs(exp))
        if not ok and na > 1:
            ok = abs(val / na - g) <= band  # per-formula-unit normalization
        c = extract.Claim("property_ef", f"{val} eV/atom", (off, off + 1),
                          payload={"name": mat, "value": val, "unit": "ev/atom",
                                   "log": {"tier": "1.5", "cost": "table", "wall_clock_s": 0.0}})
        c.status = "ok" if ok else "fail"
        c.detail = (f"formation energy {val} eV/atom for {mat}; MP {g} eV/atom"
                    + (f" / exp {exp}" if exp is not None else "")
                    + f"; {'within' if ok else 'outside'} tol")
        c.payload["bound_name"] = mat
        out.append(c)
    return out


def ground_trace(text: str, enable_physics=True, external_lookup=None,
                 extra_known_names=None, mp_lookup=None, named_sg_lookup=None,
                 ef_lookup=None):
    """Extract and verify every chemical claim in a trace.

    Escalation order (spec Section 1.3): each claim stops at the first tier that
    falsifies it or confirms it against a reference. Every claim carries a
    `log` dict {tier, cost, wall_clock_s} for the cost-accounting analysis.

    `named_sg_lookup`: optional dict {material_name(lower) -> set/list of accepted
    space-group NUMBERS}, e.g. {"pyrite": {205}, "quartz": {152,154,180,181}}.
    When supplied, an sg_number claim is additionally checked against the space
    group that is CORRECT FOR THE NAMED MATERIAL (a reference-identity tier),
    not just internal symbol<->number consistency. This is the crystalline
    analog of the formula PubChem check and closes the space-group detection gap
    (self-consistent-but-wrong pairs like brookite->#136). Name binding is done
    at the verification layer via `extract._nearest_known_name`, so the frozen
    extractor is not modified.
    """
    import time
    known = set(verify.REFERENCE.keys())
    if extra_known_names:
        known |= {n.strip().lower() for n in extra_known_names}
    if named_sg_lookup:
        known |= {n.strip().lower() for n in named_sg_lookup}
    if ef_lookup:
        known |= {n.strip().lower() for n in ef_lookup}
    known = sorted(known, key=len, reverse=True)
    claims = extract.extract_all(text, known_names=known)
    # Orchestration-layer formation-energy detection tier (frozen extractor untouched).
    if ef_lookup:
        from grounded_matsci.domain import tolerances as _t
        claims = list(claims) + detect_formation_energy(text, ef_lookup, _t.FORMATION_ENERGY)
    for c in claims:
        c.payload.setdefault("log", {"tier": None, "cost": None, "wall_clock_s": 0.0})
        t0 = time.time()
        if c.kind == "smiles":
            status, detail, canon, _ = verify.verify_smiles(c.raw)
            c.status, c.detail = status, detail
            if canon:
                c.payload["canonical"] = canon
            c.payload["log"]["tier"] = "0"
        elif c.kind == "formula":
            c.status, c.detail = verify.verify_formula(
                c.payload["name"], c.payload["formula"], external_lookup)
            c.payload["log"]["tier"] = "1"
        elif c.kind == "property":
            c.status, c.detail, meta = verify.verify_property(
                c.payload["name"], c.payload["prop"],
                c.payload["value"], c.payload["unit"],
                external_lookup=external_lookup,
                enable_physics=enable_physics, return_meta=True)
            c.payload["log"]["tier"] = meta["tier"]
            c.payload["log"]["wall_clock_s"] = meta["wall_clock_s"]
            c.payload["meta"] = meta
        elif c.kind == "sg_number" and crystal:
            c.status, c.detail = crystal.verify_spacegroup_symbol_number(
                c.payload["symbol"], c.payload["number"])
            c.payload["log"]["tier"] = "0"
            # Reference-identity tier: is this space group correct for the NAMED
            # material? Bind the nearest preceding known material name to the claim
            # span and check the claimed number against the named-phase reference.
            if named_sg_lookup and c.status != "fail":
                # Context-aware binding: bind the material whose name is nearest
                # BEFORE the claim, but only if no OTHER material name (e.g. a
                # "for comparison, Rutile is #136" aside) sits closer. This avoids
                # attributing a comparison-context space group to the wrong phase.
                cand = list(named_sg_lookup)
                # include common comparison names so an intervening "Rutile"/"Anatase"
                # blocks a false bind even if not itself in the reference set
                block_names = cand + ["rutile", "anatase", "marcasite", "metacinnabar",
                                      "calcite", "aragonite", "cristobalite"]
                mat = extract._nearest_known_name(text, c.span[0], cand)
                nearest_any = extract._nearest_known_name(text, c.span[0], block_names)
                if mat is not None and nearest_any is not None and nearest_any != mat:
                    mat = None  # a different (comparison) material is closer -> do not bind
                if mat is not None:
                    accepted = named_sg_lookup.get(mat) or named_sg_lookup.get(mat.lower())
                    if accepted:
                        accepted = set(accepted)
                        num = c.payload.get("number")
                        if num not in accepted:
                            c.status = "fail"
                            c.detail = (f"space group #{num} ({c.payload.get('symbol')}) is wrong "
                                        f"for {mat}; reference phase is #{sorted(accepted)}")
                            c.payload["bound_name"] = mat
                            c.payload["log"]["tier"] = "1"
        elif c.kind == "sg_system" and crystal:
            c.status, c.detail = crystal.verify_spacegroup_system(
                c.payload["sg"], c.payload["system"])
            c.payload["log"]["tier"] = "0"
        elif c.kind == "lattice" and crystal:
            p = c.payload
            c.status, c.detail = crystal.verify_lattice_system(
                p["a"], p["b"], p["c"], p["alpha"], p["beta"], p["gamma"],
                p["system"])
            c.payload["log"]["tier"] = "0"
        elif crystal is None and c.kind in ("sg_number", "sg_system", "lattice"):
            c.status, c.detail = "unchecked", "pymatgen not available"
        c.payload["log"]["wall_clock_s"] = max(
            c.payload["log"]["wall_clock_s"], time.time() - t0)
        c.payload["log"]["cost"] = TIER_COST.get(c.payload["log"]["tier"], "none")
    return claims


def verify_formula_mp(formula, mp_lookup, claimed_sg_number=None):
    """Cross-check a crystalline formula's space group against Materials Project
    (Tier 1 identity for solids). Returns (status, detail)."""
    if mp_lookup is None:
        return "unchecked", "no Materials Project lookup configured"
    from grounded_matsci.io import matproj
    return matproj.verify_against_mp(mp_lookup, formula,
                                     claimed_sg_number=claimed_sg_number)


def annotate(text: str, claims):
    """Insert inline [FAIL: ...] markers after each verified claim."""
    out = []
    last = 0
    for c in sorted(claims, key=lambda x: x.span[0]):
        s, e = c.span
        out.append(text[last:e])
        tag = SYMBOL.get(c.status, "?").strip()
        out.append(f"  «{tag}: {c.detail}»")
        last = e
    out.append(text[last:])
    return "".join(out)


def audit_rows(claims):
    rows = []
    for c in claims:
        rows.append({
            "kind": c.kind,
            "claim": c.raw.strip()[:60],
            "status": c.status,
            "detail": c.detail,
        })
    return rows


def trust_score(claims):
    checked = [c for c in claims if c.status in ("ok", "fail")]
    if not checked:
        return None
    ok = sum(1 for c in checked if c.status == "ok")
    return ok / len(checked)
