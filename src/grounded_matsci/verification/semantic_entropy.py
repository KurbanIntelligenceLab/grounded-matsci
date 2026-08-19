"""Semantic-entropy detector (Farquhar et al., Nature 2024), the fourth detector baseline.

Reanalysis of the stored 8-sample sets from the sampling-consistency run
(SelfCheckGPT-style); no new generation. For each unit (subject, claim_type, model) the
eight sampled objects are clustered by MEANING rather than surface string, normalized
Shannon entropy is computed over the clusters, and a unit is flagged when its entropy
exceeds a threshold frozen on the development half. Precision and recall are scored
against the same is_wrong labels and the same units as the deterministic, LLM-judge, and
sampling-consistency detectors, so the four are directly comparable.

Inputs (committed under `data/frozen/`):
  selfcheck_samples.jsonl   -- per-unit list of eight sample objects, each the frozen
                               extractor's committed object from one sample
  selfcheck_detection.json  -- aligned per-unit is_wrong / modal / consistency labels

Clustering per surface:
  molecular    Hill-canonical formula string, exact-match clusters
  crystalline  normalized space-group token, exact-match clusters
  property_ef  order-invariant single-linkage on committed eV/atom values at the frozen
               0.15 tolerance
  samples that commit to no object form their own 'NC' cluster on every surface

The threshold is a single global scalar chosen to maximize pooled F1 on the development
half (subject-hash split), mirroring the single frozen consistency threshold of 0.25.
Reported precision/recall are on the full unit set, as the other three detectors are
reported; a holdout-half score is also emitted as a leakage-free check.
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

FROZEN_DIR = Path("data/frozen")
SAMPLES_FILE = "selfcheck_samples.jsonl"
DETECTION_FILE = "selfcheck_detection.json"


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
    """Meaning-based cluster label per sample; a no-commit sample maps to 'NC'."""
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
    """Normalized cluster entropy for one unit, and the number of clusters found."""
    labels = cluster_labels(surface, samples8)
    n = len(labels)
    counts = Counter(labels)
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    max_entropy = math.log(n)
    return (entropy / max_entropy if max_entropy > 0 else 0.0), len(counts)


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
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return precision, recall, {"tp": tp, "fp": fp, "fn": fn, "n": len(cells)}


def load_units(frozen_dir: Path = FROZEN_DIR):
    """Join the stored sample sets to their detection labels on (subject, type, model)."""
    samples = [
        json.loads(line)
        for line in (frozen_dir / SAMPLES_FILE).read_text().splitlines()
        if line.strip()
    ]
    det = json.loads((frozen_dir / DETECTION_FILE).read_text())
    smap = {
        (r["subject"], r["claim_type"], r["model"].split("/")[-1]): r["sample_objects"]
        for r in samples
    }
    joined = []
    for d in det:
        so = smap.get((d["subject"], d["claim_type"], d["model"].split("/")[-1]))
        if so is None:
            continue
        joined.append({**d, "sample_objects": so})
    return joined


def run(frozen_dir: Path = FROZEN_DIR):
    """Score the detector; returns the per-unit rows, the threshold, and full/holdout PR."""
    joined = load_units(frozen_dir)
    for x in joined:
        se, k = semantic_entropy(x["claim_type"], x["sample_objects"])
        x["sem_entropy"], x["n_clusters"] = se, k
        x["split"] = "dev" if _in_dev(x["subject"]) else "holdout"
    dev = [x for x in joined if x["split"] == "dev"]
    grid = [round(0.05 + 0.01 * i, 2) for i in range(91)]

    def f1(thr):
        precision, recall, _ = _pr(dev, thr)
        precision = 0.0 if precision != precision else precision
        recall = 0.0 if recall != recall else recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    thr = max(grid, key=f1)
    surfaces = ["molecular", "property_ef", "crystalline"]
    full = {ct: _pr([x for x in joined if x["claim_type"] == ct], thr) for ct in surfaces}
    hold = {
        ct: _pr([x for x in joined if x["claim_type"] == ct and x["split"] == "holdout"], thr)
        for ct in surfaces
    }
    return joined, thr, full, hold
