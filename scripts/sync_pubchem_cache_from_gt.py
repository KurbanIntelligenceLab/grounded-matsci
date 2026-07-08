#!/usr/bin/env python3
"""Merge holdout GT molecular formulas into pubchem_cache.json (formula-only stubs).

Existing cache entries are preserved. New keys get formula + null smiles so
make_combined_lookup can fall back to GT formulas for tier-1 audits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounded_matsci.io.pubchem import _gt_formulas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path, default=Path("data/frozen/holdout_ground_truth.json"))
    p.add_argument("--cache", type=Path, default=Path("data/frozen/pubchem_cache.json"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cache: dict = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text())
    gt = _gt_formulas(args.gt)
    added = 0
    for name, formula in gt.items():
        if name in cache and cache[name].get("formula"):
            continue
        cache[name] = {
            "formula": formula,
            "smiles": cache.get(name, {}).get("smiles"),
            "cid": cache.get(name, {}).get("cid"),
            "inchikey": cache.get(name, {}).get("inchikey"),
            "error": cache.get(name, {}).get("error"),
            "source": "holdout_gt_sync",
        }
        added += 1
    print(f"GT molecular subjects: {len(gt)}")
    print(f"Cache keys after sync: {len(cache)} (+{added} new/updated stubs)")
    if not args.dry_run:
        args.cache.write_text(json.dumps(cache, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
