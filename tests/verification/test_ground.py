"""Regression tests for the grounding orchestrator's reference-identity tiers.
Assertions moved unchanged from the pre-refactor regression suite."""

from grounded_matsci.domain import tolerances
from grounded_matsci.verification import ground


def test_named_phase_sg_tier():
    """Named-phase reference tier: flags a space group that is self-consistent
    but WRONG for the named material, and does NOT flag a comparison-context
    aside about a different phase (sha 42e03ae2+ verifier tier)."""
    ref = {"pyrite": {205}, "brookite": {61}, "rutile": {136}, "anatase": {141}}
    # wrong-for-material: pyrite is Pa-3 #205, not Fm-3m #225
    claims = ground.ground_trace(
        "Pyrite (FeS2) has space group Fm-3m (#225).",
        enable_physics=False,
        extra_known_names=list(ref),
        named_sg_lookup=ref,
    )
    sg = [c for c in claims if c.kind == "sg_number"]
    assert sg and sg[0].status == "fail", "named-phase tier should flag pyrite #225"
    # comparison context: brookite is #61, and mentioning rutile #136 as an aside
    # must NOT be flagged as a brookite error
    txt = "Brookite is Pbca (#61). For comparison, rutile is P4_2/mnm (#136)."
    claims = ground.ground_trace(
        txt, enable_physics=False, extra_known_names=list(ref), named_sg_lookup=ref
    )
    fails = [c for c in claims if c.kind == "sg_number" and c.status == "fail"]
    assert not fails, f"comparison-context sg wrongly flagged: {[c.payload for c in fails]}"


def test_ef_detection_tier():
    """Orchestration-layer formation-energy detection: flags wrong, passes right, either-frame."""
    tol = tolerances.FORMATION_ENERGY
    ef = {
        "halite": {"mp": -2.1, "exp": -2.131, "n_atoms": 2},
        "corundum": {"mp": -3.427, "exp": -3.473, "n_atoms": 5},
    }
    # wrong: halite -3.15 misses both frames -> fail
    cs = ground.detect_formation_energy("The formation energy of halite is -3.15 eV/atom.", ef, tol)
    assert any(c.status == "fail" for c in cs), "should flag wrong EF"
    # right (MP frame): corundum -3.43 eV/atom -> ok
    cs = ground.detect_formation_energy("corundum formation energy is -3.43 eV/atom", ef, tol)
    assert cs and all(c.status == "ok" for c in cs), "should pass correct EF"
    # final-commitment: intermediate wrong value then correct final -> ok
    cs = ground.detect_formation_energy(
        "halite: first I estimate -3.15 eV/atom, but correctly it is -2.1 eV/atom.", ef, tol
    )
    assert cs and cs[-1].status == "ok", "should judge final committed value"
