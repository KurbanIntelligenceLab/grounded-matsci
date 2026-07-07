"""EXP4 isotope-half-life domain: orchestration-layer verifier tier + grader.
Frozen cores untouched. Value parser hardened for units + scientific notation + unicode powers
(constants lesson applied pre-emptively). Grades against IAEA half_life_sec with relative tolerance.
"""

from __future__ import annotations

import re
from typing import Any

_UNIT_SEC: dict[str, float] = {
    "s": 1, "sec": 1, "second": 1, "seconds": 1, "ms": 1e-3, "us": 1e-6, "µs": 1e-6,
    "ns": 1e-9, "ps": 1e-12,
    "min": 60, "minute": 60, "minutes": 60, "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400, "wk": 604800, "week": 604800, "weeks": 604800,
    "y": 31557600, "yr": 31557600, "year": 31557600, "years": 31557600, "a": 31557600,
    "ka": 31557600e3, "kyr": 31557600e3, "my": 31557600e6, "myr": 31557600e6, "ma": 31557600e6,
    "gy": 31557600e9, "gyr": 31557600e9, "ga": 31557600e9, "by": 31557600e9,
}
_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
        "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+"}


def _norm(t: str) -> str:
    for k, v in _SUP.items():
        t = t.replace(k, v)
    # strip thousands separators between digit groups: 76,000 -> 76000
    t = re.sub(r"(?<=\d),(?=\d\d\d(?:\D|$))", "", t)
    # unify multiplication signs, then collapse "N x 10 ^ M" / "N * 10^M" -> "Ne M"
    t = t.replace("×", "x").replace("·", "x")
    t = re.sub(r"(\d(?:\.\d+)?)\s*x\s*10\s*\^?\s*([+-]?\d+)", r"\1e\2", t)
    # collapse a stray "e ^ M" spacing from unicode-power normalization
    t = re.sub(r"(\d)\s*e\s*\^?\s*([+-]?\d+)", r"\1e\2", t)
    return t


_HL = re.compile(
    r"(-?\d+\.?\d*(?:e[+-]?\d+)?)\s*(ms|us|µs|ns|ps|sec|seconds|second|s|minutes|minute|min|"
    r"hours|hour|hr|h|days|day|d|weeks|week|wk|years|year|yr|y|kyr|ka|myr|my|ma|gyr|ga|by|a)\b",
    re.I,
)


def parse_halflife_sec(text: str) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for m in _HL.finditer(_norm(text)):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        u = m.group(2).lower()
        if u in _UNIT_SEC:
            out.append((val * _UNIT_SEC[u], m.group(0)))
    return out


REL_TOL = 0.10


def grade_halflife(text: str, gt_sec: float, rel: float = REL_TOL) -> dict[str, Any]:
    vals = parse_halflife_sec(text)
    if not vals:
        return {"correct": None, "reason": "no half-life extracted"}
    final = vals[-1][0]  # final-commitment
    ok = abs(final - gt_sec) <= rel * gt_sec
    # also accept if ANY stated value matches (models sometimes list then restate)
    any_ok = any(abs(v - gt_sec) <= rel * gt_sec for v, _ in vals)
    return {"correct": bool(ok), "any_correct": bool(any_ok), "extracted_sec": final,
            "gt_sec": gt_sec, "n_values": len(vals)}
