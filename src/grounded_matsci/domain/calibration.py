"""Platt calibration gate for the gated constants rerun.

PLATT_A/PLATT_B and TRUST_THRESHOLD were fit on the development half and frozen before
the gated rerun; they are scientific constants of the registered method, not tunables.
Used by `workflows/gated_constants.py`."""

from __future__ import annotations

import math

PLATT_A, PLATT_B = 3.728, -0.970  # fit on the development half, frozen pre-holdout
TRUST_THRESHOLD = 0.5  # Platt-probability gate, frozen pre-holdout


def platt(t: float) -> float:
    return 1.0 / (1.0 + math.exp(-(PLATT_A * t + PLATT_B)))
