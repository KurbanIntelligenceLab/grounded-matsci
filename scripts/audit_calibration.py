#!/usr/bin/env python3
"""Audit Platt calibration constants against paper ECE claims (0.111 -> 0.060).

Expects a JSONL with one row per baseline item:
  {"trust": float, "correct": bool}

Split is by sorted subject hash (dev vs holdout half), matching the paper protocol
described in results.tex subsec:calibration.

Usage:
  uv run python scripts/audit_calibration.py --items handoff/calibration_items.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from grounded_matsci.domain.calibration import PLATT_A, PLATT_B, platt


def _ece(scores: list[float], labels: list[int], n_bins: int = 10) -> float:
    if not scores:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for s, y in zip(scores, labels, strict=True):
        b = min(int(s * n_bins), n_bins - 1)
        bins[b].append((s, y))
    ece = 0.0
    n = len(scores)
    for bucket in bins:
        if not bucket:
            continue
        conf = sum(s for s, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        ece += abs(acc - conf) * len(bucket) / n
    return ece


def _fit_platt(trust: list[float], labels: list[int], max_iter: int = 200) -> tuple[float, float]:
    """Simple gradient descent on Platt log-loss (for audit replay only)."""
    a, b = PLATT_A, PLATT_B
    lr = 0.05
    for _ in range(max_iter):
        grad_a = grad_b = 0.0
        for t, y in zip(trust, labels, strict=True):
            p = platt(t) if (a, b) == (PLATT_A, PLATT_B) else 1 / (1 + math.exp(-(a * t + b)))
            err = p - y
            grad_a += err * t
            grad_b += err
        a -= lr * grad_a / len(trust)
        b -= lr * grad_b / len(trust)
    return a, b


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=Path, required=True, help="JSONL trust/correct rows")
    p.add_argument("--subject-key", default="subject", help="Field for dev/holdout split")
    args = p.parse_args()

    rows: list[dict] = []
    with args.items.open() as f:
        for line in f:
            rows.append(json.loads(line))

    def half(row: dict) -> str:
        key = str(row.get(args.subject_key, row.get("cell", len(rows))))
        h = hashlib.sha256(key.encode()).hexdigest()
        return "holdout" if int(h[:8], 16) % 2 else "dev"

    dev = [r for r in rows if half(r) == "dev"]
    hold = [r for r in rows if half(r) == "holdout"]

    def vecs(batch: list[dict]) -> tuple[list[float], list[int]]:
        trust = [float(r["trust"]) for r in batch if r.get("trust") is not None]
        labels = [int(r["correct"]) for r in batch if r.get("trust") is not None]
        return trust, labels

    hold_trust, hold_y = vecs(hold)
    raw_ece = _ece(hold_trust, hold_y)
    scaled = [platt(t) for t in hold_trust]
    platt_ece = _ece(scaled, hold_y)

    dev_trust, dev_y = vecs(dev)
    a_fit, b_fit = _fit_platt(dev_trust, dev_y)

    print(f"Items: n={len(rows)} (dev={len(dev)}, holdout={len(hold)})")
    print(f"Frozen Platt: a={PLATT_A:.3f}, b={PLATT_B:.3f}")
    print(f"Holdout raw ECE:   {raw_ece:.3f}  (paper 0.111)")
    print(f"Holdout Platt ECE: {platt_ece:.3f}  (paper 0.060)")
    print(f"Dev refit (audit): a={a_fit:.3f}, b={b_fit:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
