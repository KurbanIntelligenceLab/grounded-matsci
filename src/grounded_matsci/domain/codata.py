"""Cross-domain transfer to the physical-constants domain: the grading half.

Demonstrates the method template — extract checkable object -> tiered grounding -> gated
repair — on a second domain with CODATA as the Tier-1 identity reference. Lives entirely
at the orchestration layer; the core extractor sha (42e03ae2) and verifier config
(6e45797a) are untouched. The Mode-A runner half lives in `workflows/codata.py`.

Detection: parse the final committed numeric value for the named constant from the trace,
compare against the frozen CODATA reference within relative tolerance. A mismatch is a
flag (Mode A then injects the reference value and asks for correction, exactly as the
materials verifier does)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_SUP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁻": "-",
    "⁺": "+",
}


def _clean(s: str) -> str:
    """Normalize scientific-notation forms so the exponent is never dropped: unicode superscripts
    (10⁻¹⁹), LaTeX braces (10^{-19}), unicode minus. Fixes the orchestration-layer extraction
    artifact where a mantissa was captured but its ×10^n exponent lost (2026-07-06 CODATA audit)."""
    s = "".join(_SUP.get(c, c) for c in s).replace("−", "-").replace("·", "*")
    s = re.sub(r"10\s*\^\s*\{\s*([-+]?\d+)\s*\}", r"10^\1", s)  # 10^{-19} -> 10^-19
    return s.replace("{", "").replace("}", "")


def _parse_number(expr: str) -> float | None:
    """Parse ONE scientific number from the start of expr (mantissa [× 10^exp | e exp | 10^exp]).
    Collapses CODATA digit-group spaces inside the mantissa only; stops at the trailing unit.
    Handles the common closed-form '4π × 10^-7' the exact vacuum-permeability answer takes."""
    import math

    expr = _clean(expr).strip().replace("π", "*pi").replace("\\pi", "*pi")
    pim = re.match(
        r"\s*([-+]?\d*\.?\d*)\s*\*?\s*pi\s*(?:[×x*·]\s*10\s*\^?\s*([-+]?\d+)|[eE]\s*([-+]?\d+))?",
        expr,
    )
    if pim and "pi" in expr[: pim.end()]:
        coef = (
            float(pim.group(1))
            if pim.group(1) not in (None, "", "+", "-")
            else (1.0 if pim.group(1) != "-" else -1.0)
        )
        val = coef * math.pi
        e = pim.group(2) or pim.group(3)
        if e:
            val *= 10 ** int(e)
        return val
    mm = re.match(r"\s*([-+]?\d[\d\s]*\.?\d*)", expr)
    if not mm:
        return None
    try:
        val = float(mm.group(1).replace(" ", ""))
    except ValueError:
        return None
    rest = expr[mm.end() :]
    em = re.match(
        r"\s*(?:[eE]\s*([-+]?\d+)|[×x*·]\s*10\s*\^?\s*([-+]?\d+)|10\s*\^\s*([-+]?\d+))", rest
    )
    if em:
        val *= 10 ** int(next(g for g in em.groups() if g is not None))
    return val


_SCI = r"[-+]?\d[\d\s]*\.?\d*\s*(?:[eE]\s*[-+]?\d+|[×x*·]\s*10\s*\^?\s*[-+]?\d+|10\^[-+]?\d+)"


def extract_const_value(text: str | None, unit: str | None = None) -> float | None:
    """Take the FINAL committed value: prefer a 'Value: X' line, else the last scientific token."""
    t = text or ""
    m = re.findall(r"[Vv]alue\s*[:=]\s*(.+)", t)
    if m:
        v = _parse_number(m[-1])
        if v is not None:
            return v
    cands = re.findall(_SCI, _clean(t))
    return _parse_number(cands[-1]) if cands else None


def grade_const(
    text: str, gt_value: float, rel_tol: float = 0.01
) -> tuple[bool | None, float | None]:
    """Legacy signature (rel_tol). Returns (correct, extracted); None if no value committed."""
    v = extract_const_value(text)
    if v is None or v == 0:
        return None, None
    if gt_value == 0:
        return abs(v) < rel_tol, v
    return abs(v - gt_value) / abs(gt_value) <= rel_tol, v


def _round_sig(x: float, k: int) -> float:
    """Round x to k significant figures."""
    import math

    if x == 0:
        return 0.0
    d = k - 1 - math.floor(math.log10(abs(x)))
    return round(x, d)


def _trunc_sig(x: float, k: int) -> float:
    """Truncate x toward zero to k significant figures."""
    import math

    if x == 0:
        return 0.0
    d = k - 1 - math.floor(math.log10(abs(x)))
    f: float = 10**d
    return math.trunc(x * f) / f


def _quoted_sigfigs(m: float) -> int:
    """Recover the number of significant figures the model actually quoted for mantissa m∈[1,10),
    robust to binary float representation noise: the smallest k for which rounding m to k sig figs
    reproduces m within relative 1e-9 (so 6.62 with float tail 6.6200000001 counts as 3, not 12)."""
    for k in range(1, 15):
        if abs(_round_sig(m, k) - m) <= 1e-9 * abs(m):
            return k
    return 15


def _sigfig_ok(v: float, gt: float) -> bool:
    """Significant-figures policy for EXACT constants: the stated value is correct iff its quoted
    mantissa is a correct rounding OR truncation of the reference to the number of
    significant digits the model provided, at the same order of magnitude. A
    truncated-but-digit-correct value (e.g. c = 3×10^8 or 2.998×10^8, Planck = 6.62×10⁻³⁴)
    passes; any altered digit at the quoted precision fails. The quoted precision is
    recovered from the value itself, not from a noisy float repr."""
    import math

    if v == 0 or gt == 0:
        return abs(v - gt) < 1e-30
    if (v < 0) != (gt < 0):
        return False
    v, gt = abs(v), abs(gt)
    ev = math.floor(math.log10(v))
    eg = math.floor(math.log10(gt))
    if ev != eg:
        return False
    mv = v / 10**ev  # model mantissa in [1,10)
    mg = gt / 10**eg  # reference mantissa in [1,10)
    n = _quoted_sigfigs(mv)
    # tolerance = 10% of one unit in the last quoted place: rejects a last-digit alteration
    # (a full ULP away) while absorbing binary-float representation noise (~1e-16 relative).
    eps: float = 0.1 * 10 ** (1 - n)
    return (
        abs(_round_sig(mv, n) - _round_sig(mg, n)) <= eps
        or abs(_round_sig(mv, n) - _trunc_sig(mg, n)) <= eps
    )


def grade_const_policy(
    text: str, rec: dict[str, Any], round_floor: float = 1e-4
) -> tuple[bool | None, float | None, float | str | None]:
    """CODATA-2022 tolerance policy (registered 2026-07-06):
      - EXACT SI-defining/derived constants: significant-figures policy — the quoted mantissa
        must be a correct rounding or truncation of the reference at the model's own precision
        (truncation OK, digit alteration = error). Implemented by _sigfig_ok (not a flat
        relative tolerance).
      - MEASURED constants: correct within max(CODATA-2022 rel std uncertainty, 2018->2022 drift,
        round_floor), so neither the adjustment version nor quoted-digit truncation can be an error.
    Returns (correct|None, extracted, tol_used). tol_used='sigfig' for exact constants."""
    v = extract_const_value(text)
    if v is None or v == 0:
        return None, None, None
    gt = rec["value"]
    if rec.get("exact"):
        return _sigfig_ok(v, gt), v, "sigfig"
    tol = max(rec.get("rel_unc_2022", 0.0), rec.get("drift_2018_2022", 0.0), round_floor)
    if gt == 0:
        return abs(v) <= tol, v, tol
    return abs(v - gt) / abs(gt) <= tol, v, tol


def make_const_lookup(
    codata_gt: dict[str, dict[str, Any]],
) -> Callable[[str], tuple[None, str] | None]:
    """gt_kwargs external_lookup analogue: name -> reference value string for Mode A feedback."""
    ref = {name.lower(): rec["value"] for name, rec in codata_gt.items()}

    def lookup(name: str) -> tuple[None, str] | None:
        v = ref.get(name.lower())
        return (None, f"{v} {codata_gt.get(name, {}).get('unit', '')}") if v is not None else None

    return lookup
