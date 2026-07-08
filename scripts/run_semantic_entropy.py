#!/usr/bin/env python3
"""Compute the semantic-entropy detector (Table S6, 4th detector) from frozen inputs.

Reanalysis only, no generation. Reads data/frozen/{selfcheck_detection.json, selfcheck_samples.jsonl}
and writes results/semantic_entropy_detection.json (+ per-cell detail). Run from the repo root:

    python scripts/run_semantic_entropy.py
"""
from __future__ import annotations

import json
from pathlib import Path

from grounded_matsci.verification.semantic_entropy import (
    load_units,
    semantic_entropy,
    _in_dev,
    _pr,
)

REPO = Path(__file__).resolve().parent.parent
FROZEN = REPO / "data" / "frozen"
RESULTS = REPO / "results"
FE_ABS_TOL = 0.15
SURFACES = ("molecular", "property_ef", "crystalline")


def main():
    # load_units expects selfcheck_rep1.jsonl; the repo stores it as selfcheck_samples.jsonl,
    # so join here directly using the shared naming.
    samples = [json.loads(l) for l in open(FROZEN / "selfcheck_samples.jsonl") if l.strip()]
    det = json.load(open(FROZEN / "selfcheck_detection.json"))
    smap = {(r["subject"], r["claim_type"], r["model"].split("/")[-1]): r["sample_objects"]
            for r in samples}
    joined = []
    for d in det:
        so = smap.get((d["subject"], d["claim_type"], d["model"].split("/")[-1]))
        if so is not None:
            joined.append({**d, "sample_objects": so})

    for x in joined:
        se, k = semantic_entropy(x["claim_type"], x["sample_objects"])
        x["sem_entropy"], x["n_clusters"] = se, k
        x["split"] = "dev" if _in_dev(x["subject"]) else "holdout"

    dev = [x for x in joined if x["split"] == "dev"]
    grid = [round(0.05 + 0.01 * i, 2) for i in range(91)]

    def f1(thr):
        P, R, _ = _pr(dev, thr)
        P = 0.0 if P != P else P
        R = 0.0 if R != R else R
        return 2 * P * R / (P + R) if (P + R) else 0.0

    thr = max(grid, key=f1)
    full = {ct: _pr([x for x in joined if x["claim_type"] == ct], thr) for ct in SURFACES}
    hold = {ct: _pr([x for x in joined if x["claim_type"] == ct and x["split"] == "holdout"], thr)
            for ct in SURFACES}

    three = json.load(open(RESULTS / "detection_table_3way.json")) if (RESULTS / "detection_table_3way.json").exists() else {}

    def rnd(v):
        return round(v, 2) if v == v else None

    out = {
        "method": "semantic_entropy_detector",
        "reference": "Farquhar, Kossen, Kuhn, Gal (2024) Nature 630:625-630",
        "n_units": len(joined),
        "n_samples_per_unit": 8,
        "clustering": {
            "molecular": "Hill-canonical formula string, exact-match clusters; no-commit=NC",
            "crystalline": "normalized space-group token, exact-match clusters; no-commit=NC",
            "property_ef": f"order-invariant single-linkage on eV/atom at tol {FE_ABS_TOL}; no-commit=NC",
        },
        "entropy": "normalized Shannon entropy over clusters, H/log(n_samples) in [0,1]",
        "threshold": {"value": thr, "frozen_on": "dev half (subject-hash split, seed 'sement')",
                      "rule": "flag if entropy >= threshold; dev-optimal pooled F1"},
        "pr_full_set": {ct: {"P": rnd(full[ct][0]), "R": rnd(full[ct][1]), **full[ct][2]}
                        for ct in SURFACES},
        "pr_holdout_half": {ct: {"P": rnd(hold[ct][0]), "R": rnd(hold[ct][1])} for ct in SURFACES},
        "note": "Detection-only: returns no reference value to inject, so it can detect but not repair.",
    }
    (RESULTS / "semantic_entropy_detection.json").write_text(json.dumps(out, indent=2))
    print(f"frozen threshold {thr}")
    for ct in SURFACES:
        P, R, c = full[ct]
        print(f"{ct:14s} P={P:.2f} R={R:.2f}  {c}")


if __name__ == "__main__":
    main()
