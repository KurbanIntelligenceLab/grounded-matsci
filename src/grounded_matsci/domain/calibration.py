"""Dev-frozen Platt calibration gate for the gated CODATA rerun (EXP2, §2.8).

PLATT_A/PLATT_B and TRUST_THRESHOLD were frozen on the development half before the
gated rerun; they are scientific constants of the registered method, not tunables.
Extracted from the EXP2 runner (now `workflows/exp2_gated_codata.py`) unchanged."""

from __future__ import annotations

import math

PLATT_A, PLATT_B = 3.728, -0.970  # dev-frozen (§2.8)
TRUST_THRESHOLD = 0.5  # dev-frozen Platt-prob gate


def platt(t: float) -> float:
    return 1.0 / (1.0 + math.exp(-(PLATT_A * t + PLATT_B)))
