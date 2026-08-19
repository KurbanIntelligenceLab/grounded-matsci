"""Two-stage triage detector, at the orchestration layer.

The deterministic verifier tier fires first (ground.ground_trace); the
sampling-consistency signal (threshold 0.25, frozen on the development half before the
holdout run) is invoked ONLY on units the deterministic tier does not flag. This is
compute-aware triage, not a naive OR: the expensive signal is spent only where the cheap
one is silent.

Prior art: detector aggregation; two-stage / compute-aware triage; stepwise-consistency filtering.
"""

from __future__ import annotations

CONSISTENCY_THRESHOLD = 0.25  # frozen on the development half, before the holdout run


class _SoftFail:
    def __init__(self, raw, cons):
        self.raw = raw
        self.detail = (
            f"cross-sample sampling-consistency below threshold ({cons:.2f} <= "
            f"{CONSISTENCY_THRESHOLD}); the value is unstable across independent resamples "
            "and may be unreliable. Re-derive it carefully and state the corrected value."
        )


def make_cell_verify(base_verify, cell_consistency):
    """base_verify: the ORIGINAL loop._verify (captured before patching, avoids recursion).
    cell_consistency: frozen consistency signal for THIS cell's committed object (float or None)."""
    import re as _re

    _EFV = _re.compile(r"-?\d+\.?\d*\s*(?:eV\s*/\s*atom|eV\s*per\s*atom)", _re.I)  # noqa: N806

    def _verify2(text, **gt):
        claims, fails = base_verify(text, **gt)
        if fails:
            return claims, fails  # stage 1: deterministic tier flagged
        # stage 2: consistency-triggered, only on deterministic-passed units.
        if cell_consistency is not None and cell_consistency <= CONSISTENCY_THRESHOLD:
            ef = [c for c in claims if getattr(c, "kind", "") == "property_ef"]
            if ef:
                raw = getattr(ef[0], "raw", "the stated value")
                return claims, [_SoftFail(raw, cell_consistency)]
            # value present but unbound (terse answer, no adjacent material name): still fire
            # on the consistency signal against the raw eV/atom value so the triage is not
            # silently skipped.
            m = _EFV.search(text or "")
            if m:
                return claims, [_SoftFail(m.group(0).strip(), cell_consistency)]
        return claims, []

    return _verify2
