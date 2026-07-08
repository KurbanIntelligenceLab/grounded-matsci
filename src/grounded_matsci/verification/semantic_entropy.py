"""Semantic-entropy detector (Farquhar et al., Nature 2024) as a 4th detector for Table S6.

Reanalysis of the stored 8-sample sets from the sampling-consistency run (SelfCheckGPT-style);
NO new generation. For each unit (subject, claim_type, model) we cluster its 8 sampled objects by
MEANING (not surface string), compute normalized Shannon entropy over the clusters, and flag a unit
when entropy exceeds a threshold frozen on a dev half. Precision/recall are scored against the same
is_wrong labels and the same units as the deterministic / LLM-judge / sampling-consistency detectors.

Inputs (handoff/):
  selfcheck_rep1.jsonl        -- per-unit list of 8 sample_objects (the frozen extractor's committed
                                 object from each sample)
  selfcheck_detection.json    -- aligned per-unit is_wrong / modal / consistency labels

Clustering per surface:
  molecular    Hill-canonical formula string, exact-match clusters
  crystalline  normalized space-group token, exact-match clusters
  property_ef  order-invariant single-linkage on committed eV/atom values at the frozen 0.15 tol
  no-commit samples form their own 'NC' cluster on every surface

Threshold is a single global scalar chosen to maximize pooled F1 on the dev half (subject-hash
split), mirroring the single frozen consistency threshold (0.25). Reported P/R are on the full unit
set (as the other three detectors are reported); a holdout-half score is also emitted as a
leakage-free check.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

FE_ABS_TOL = 0.15  # eV/atom, frozen formation-energy tolerance
SPLIT_SEED = "sement"


def _first(obj):
    if not obj:
        return None
    v = obj[0]
    return v.strip() if isinstance(v, str) else v


def _hill(formula):
    toks = re.findall(r"([A-Z][a-z]?)(\d*)", formula or "")
    counts, ok = {}, False
    for el, n in toks:
        if not el:
            continue
        ok = True
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    if not ok:
        return (formula or "").upper()
    parts = []
    if "C" in counts:
        for el in ("C", "H"):
            if el in counts:
                parts.append(f"{el}{counts.pop(el)}")
    for el in sorted(counts):
        parts.append(f"{el}{counts[el]}")
    return "".join(parts)


def _parse_fe(tok):
    if tok is None:
        return None
    m = re.search(r"-?\d+\.?\d*", str(tok))
    return float(m.group(0)) if m else None


def cluster_labels(surface, samples8):
    """Meaning-based cluster label per sample; None (no-commit) -> 'NC'."""
    toks = [_first(s) for s in samples8]
    if surface == "molecular":
        return ["NC" if _first([t]) is None or not t else _hill(t) for t in toks]
    if surface == "crystalline":
        return ["NC" if not t else re.sub(r"\s+", "", str(t)).lower() for t in toks]
    if surface == "property_ef":
        vals = [_parse_fe(t) for t in toks]
        idx_vals = sorted((v, i) for i, v in enumerate(vals) if v is not None)
        labels = ["NC"] * len(vals)
        cid, prev = -1, None
        for v, i in idx_vals:
            if prev is None or (v - prev) > FE_ABS_TOL:
                cid += 1
            labels[i] = f"c{cid}"
            prev = v
        return labels
    raise ValueError(surface)


def semantic_entropy(surface, samples8):
    labels = cluster_labels(surface, samples8)
    n = len(labels)
    counts = Counter(labels)
    H = -sum((c / n) * math.log(c / n) for c in counts.values())
    Hmax = math.log(n)
    return (H / Hmax if Hmax > 0 else 0.0), len(counts)


def _in_dev(subject):
    h = int(hashlib.sha256(f"{SPLIT_SEED}|{subject}".encode()).hexdigest(), 16)
    return (h % 2) == 0


def _pr(cells, thr):
    tp = fp = fn = 0
    for x in cells:
        flag = x["sem_entropy"] >= thr
        wrong = bool(x["is_wrong"])
        if flag and wrong:
            tp += 1
        elif flag and not wrong:
            fp += 1
        elif not flag and wrong:
            fn += 1
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return P, R, {"tp": tp, "fp": fp, "fn": fn, "n": len(cells)}


def load_units(handoff: Path):
    samples = [json.loads(l) for l in open(handoff / "selfcheck_rep1.jsonl") if l.strip()]
    det = json.load(open(handoff / "selfcheck_detection.json"))
    smap = {(r["subject"], r["claim_type"], r["model"].split("/")[-1]): r["sample_objects"]
            for r in samples}
    joined = []
    for d in det:
        so = smap.get((d["subject"], d["claim_type"], d["model"].split("/")[-1]))
        if so is None:
            continue
        joined.append({**d, "sample_objects": so})
    return joined


def run(handoff: Path):
    joined = load_units(handoff)
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
    surfaces = ["molecular", "property_ef", "crystalline"]
    full = {ct: _pr([x for x in joined if x["claim_type"] == ct], thr) for ct in surfaces}
    hold = {ct: _pr([x for x in joined if x["claim_type"] == ct and x["split"] == "holdout"], thr)
            for ct in surfaces}
    return joined, thr, full, hold


if __name__ == "__main__":
    import sys
    hp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("handoff")
    joined, thr, full, hold = run(hp)
    print(f"frozen threshold: {thr}")
    for ct in ("molecular", "property_ef", "crystalline"):
        P, R, c = full[ct]
        print(f"{ct:14s} P={P:.2f} R={R:.2f}  {c}")
