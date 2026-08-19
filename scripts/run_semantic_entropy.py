#!/usr/bin/env python3
"""Regenerate the semantic-entropy detector results from the committed frozen inputs.

Reanalysis only, no generation. Reads
``data/frozen/{selfcheck_samples.jsonl, selfcheck_detection.json}`` and writes
``results/semantic_entropy_detection.json`` plus the per-cell detail. All scoring lives
in ``grounded_matsci.verification.semantic_entropy``; this file only selects paths and
serializes. Run from the repository root:

    uv run python scripts/run_semantic_entropy.py

The reported summary reproduces the committed
``results/semantic_entropy_detection.json`` exactly. The per-cell entropies can differ
from the committed file in the last floating-point digit on some platforms, because
``math.log`` is supplied by the host C library; the threshold, the cluster counts, and
every reported precision/recall figure are unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grounded_matsci.verification import semantic_entropy as se

REPO = Path(__file__).resolve().parent.parent
FROZEN = REPO / "data" / "frozen"
RESULTS = REPO / "results"
SURFACES = ("molecular", "property_ef", "crystalline")
PERCELL_FIELDS = (
    "subject",
    "claim_type",
    "model",
    "is_wrong",
    "consistency",
    "sem_entropy",
    "n_clusters",
    "split",
)


def _round(value: float) -> float | None:
    """Round for reporting; NaN (an empty precision/recall cell) serializes as null."""
    return round(value, 2) if value == value else None


def summary(thr: float, joined: list[dict[str, Any]], full: dict, hold: dict) -> dict[str, Any]:
    return {
        "method": "semantic_entropy_detector",
        "reference": "Farquhar, Kossen, Kuhn, Gal (2024) Nature 630:625-630",
        "n_units": len(joined),
        "n_samples_per_unit": 8,
        "clustering": {
            "molecular": "Hill-canonical formula string, exact-match clusters; no-commit=NC",
            "crystalline": "normalized space-group token, exact-match clusters; no-commit=NC",
            "property_ef": (
                f"order-invariant single-linkage on eV/atom at tol {se.FE_ABS_TOL}; no-commit=NC"
            ),
        },
        "entropy": "normalized Shannon entropy over clusters, H/log(n_samples) in [0,1]",
        "threshold": {
            "value": thr,
            "frozen_on": (f"dev half (subject-hash split, seed '{se.SPLIT_SEED}')"),
            "rule": "flag if entropy >= threshold; dev-optimal pooled F1",
        },
        "pr_full_set": {
            ct: {"P": _round(full[ct][0]), "R": _round(full[ct][1]), **full[ct][2]}
            for ct in SURFACES
        },
        "pr_holdout_half": {
            ct: {"P": _round(hold[ct][0]), "R": _round(hold[ct][1])} for ct in SURFACES
        },
        "note": (
            "Detection-only: returns no reference value to inject, so it can detect but not repair."
        ),
    }


def main() -> None:
    joined, thr, full, hold = se.run(FROZEN)
    (RESULTS / "semantic_entropy_detection.json").write_text(
        json.dumps(summary(thr, joined, full, hold), indent=2)
    )
    (RESULTS / "semantic_entropy_percell.json").write_text(
        json.dumps([{k: x[k] for k in PERCELL_FIELDS} for x in joined])
    )
    print(f"frozen threshold {thr}")
    for ct in SURFACES:
        precision, recall, counts = full[ct]
        print(f"{ct:14s} P={precision:.2f} R={recall:.2f}  {counts}")


if __name__ == "__main__":
    main()
