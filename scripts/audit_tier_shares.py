#!/usr/bin/env python3
"""Recompute fig_method tier-share percentages from claim-level verifier logs.

Denominator: claims that resolve at a tier (status ok or fail), excluding unchecked.
Numerator per tier: payload log tier field at first resolution (0, 1, 1.5, 3).

Usage:
  uv run python scripts/audit_tier_shares.py \\
    --traces handoff/holdout_arms_rep0.jsonl \\
    --named-sg data/frozen/named_sg_rule_gt.json \\
    --ef-lookup data/frozen/ef_lookup.json \\
    --pubchem data/frozen/pubchem_cache.json \\
    --holdout-gt data/frozen/holdout_ground_truth.json

Each trace JSONL row must include a ``text`` field (final or round-0 answer).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from grounded_matsci.io.pubchem import make_combined_lookup
from grounded_matsci.verification import ground


def _load_json(path: Path):
    return json.loads(path.read_text())


def tier_shares(traces: list[dict], gt_kwargs: dict) -> dict[str, float]:
    tier_counts: Counter[str] = Counter()
    for row in traces:
        text = row.get("text") or row.get("final_text") or ""
        if not text.strip():
            continue
        claims = ground.ground_trace(text, **gt_kwargs)
        for c in claims:
            if c.status not in ("ok", "fail"):
                continue
            tier = (c.payload.get("log") or {}).get("tier")
            if tier is None:
                continue
            key = str(tier).replace("(skipped)", "")
            tier_counts[key] += 1
    total = sum(tier_counts.values())
    if total == 0:
        return {}
    return {k: 100.0 * tier_counts[k] / total for k in sorted(tier_counts)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, required=True, help="JSONL with text/final_text")
    p.add_argument("--named-sg", type=Path, default=Path("data/frozen/named_sg_rule_gt.json"))
    p.add_argument("--ef-lookup", type=Path, default=Path("data/frozen/ef_lookup.json"))
    p.add_argument("--pubchem", type=Path, default=Path("data/frozen/pubchem_cache.json"))
    p.add_argument("--holdout-gt", type=Path, default=Path("data/frozen/holdout_ground_truth.json"))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    named_sg = _load_json(args.named_sg)
    sg_lookup = named_sg.get("token_accepted_sg", named_sg)
    gt_kwargs = {
        "enable_physics": True,
        "ef_lookup": _load_json(args.ef_lookup),
        "named_sg_lookup": sg_lookup,
        "external_lookup": make_combined_lookup(args.pubchem, args.holdout_gt),
    }

    traces: list[dict] = []
    with args.traces.open() as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            traces.append(json.loads(line))

    shares = tier_shares(traces, gt_kwargs)
    paper = {"0": 40.9, "1": 48.7, "1.5": 10.4, "3": 0.0}
    print(f"Resolved claims: {sum(1 for _ in traces)} traces processed")
    print(f"{'tier':>6}  {'computed':>10}  {'paper':>10}")
    for tier in ("0", "1", "1.5", "3"):
        comp = shares.get(tier, 0.0)
        ref = paper.get(tier, 0.0)
        flag = "  OK" if abs(comp - ref) < 0.15 else "  DIFF"
        print(f"{tier:>6}  {comp:10.1f}  {ref:10.1f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
