"""
Labeled-trace corpus harness (spec Section 2).

This is the infrastructure for the dataset that is the paper's centerpiece. It
does NOT label anything with an LLM — labeling is done by human domain experts
(spec Section 6 prohibits LLM-as-judge ground truth). What this module does:

  1. `generate_traces`  -- elicit CoT traces from the model set on a prompt list,
     recording exact model string, sampling params, and timestamp (spec Section 5).
  2. `build_claim_records` -- run the extractor over each trace and emit
     structured claim objects {trace_id, span, type, referent, value, unit,
     source_sentence_id} (spec Section 1.1 format), each with the verifier's
     automatic verdict attached as a *suggestion* (never a label).
  3. `write_labeling_sheet` -- emit a JSONL + CSV labeling sheet where each row
     is one claim for a human to mark correct / incorrect / unverifiable.
  4. `cohens_kappa` and `agreement_report` -- inter-annotator agreement on the
     double-labeled subset (spec Section 2: >=20% double-labeled, kappa reported).

The verifier's own verdict is stored in a separate column so that, once experts
label, we can compute detection precision/recall/F1 of the verifier against the
human ground truth (spec Section 3.3) -- but the human label is authoritative.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

from grounded_matsci.verification import ground


# ---------------------------------------------------------------------------
# 1. trace generation
# ---------------------------------------------------------------------------
def generate_traces(
    prompts,
    generate,
    model_string,
    domain,
    system=None,
    n_samples=1,
    temperature=0.7,
    out_path=None,
):
    """Elicit CoT traces. `generate(messages, sample=bool)` is a driver callable.

    Records provenance for reproducibility. Returns a list of trace dicts.
    """
    traces = []
    for pid, prompt in enumerate(prompts):
        for s in range(n_samples):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            text = generate(msgs, sample=(n_samples > 1))
            tid = hashlib.sha1(f"{model_string}|{domain}|{pid}|{s}".encode()).hexdigest()[:12]
            traces.append(
                {
                    "trace_id": tid,
                    "domain": domain,  # 'molecular' | 'crystalline'
                    "prompt_id": pid,
                    "sample_index": s,
                    "prompt": prompt,
                    "model": model_string,
                    "temperature": temperature if n_samples > 1 else 0.0,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "text": text,
                }
            )
    if out_path:
        with Path(out_path).open("w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
    return traces


# ---------------------------------------------------------------------------
# 2. structured claim records (spec Section 1.1 schema)
# ---------------------------------------------------------------------------
def _sentence_index(text, span_start):
    """0-based index of the sentence containing character offset span_start."""
    import re

    ends = [m.end() for m in re.finditer(r"[.!?]\s|\n", text)]
    idx = 0
    for e in ends:
        if span_start < e:
            return idx
        idx += 1
    return idx


def build_claim_records(
    traces, external_lookup=None, extra_known_names=None, enable_physics=False, mp_lookup=None
):
    """Extract + auto-verify every claim in every trace. Emits records in the
    spec Section 1.1 schema plus the verifier's suggested verdict."""
    records = []
    for tr in traces:
        claims = ground.ground_trace(
            tr["text"],
            enable_physics=enable_physics,
            external_lookup=external_lookup,
            extra_known_names=extra_known_names,
            mp_lookup=mp_lookup,
        )
        for j, c in enumerate(claims):
            raw = c.raw.strip()
            span_start = tr["text"].find(raw)
            referent = (
                c.payload.get("name")
                or c.payload.get("symbol")
                or c.payload.get("sg")
                or c.payload.get("system")
            )
            records.append(
                {
                    "claim_id": f"{tr['trace_id']}-{j:03d}",
                    "trace_id": tr["trace_id"],
                    "domain": tr["domain"],
                    "model": tr["model"],
                    "prompt_id": tr["prompt_id"],
                    "type": c.kind,
                    "span_text": raw,
                    "span_start": span_start,
                    "referent": referent,
                    "value": c.payload.get("value"),
                    "unit": c.payload.get("unit"),
                    "source_sentence_id": _sentence_index(tr["text"], max(span_start, 0)),
                    # verifier SUGGESTION -- never a ground-truth label
                    "verifier_verdict": c.status,
                    "verifier_tier": c.payload.get("log", {}).get("tier"),
                    "verifier_detail": c.detail,
                    # to be filled by human annotators:
                    "human_label": "",  # correct | incorrect | unverifiable
                    "annotator_id": "",
                }
            )
    return records


# ---------------------------------------------------------------------------
# 3. labeling sheet
# ---------------------------------------------------------------------------
LABEL_COLUMNS = [
    "claim_id",
    "trace_id",
    "domain",
    "model",
    "type",
    "referent",
    "span_text",
    "value",
    "unit",
    "source_sentence_id",
    "verifier_verdict",
    "verifier_detail",
    "human_label",
    "annotator_id",
]


def write_labeling_sheet(records, csv_path, jsonl_path=None):
    """Write a human-labeling sheet. `human_label` starts blank; annotators fill
    it with correct / incorrect / unverifiable. verifier_verdict is shown for
    context but experts are instructed NOT to defer to it."""
    with Path(csv_path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    if jsonl_path:
        with Path(jsonl_path).open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return csv_path


# ---------------------------------------------------------------------------
# 4. inter-annotator agreement (spec Section 2)
# ---------------------------------------------------------------------------
def cohens_kappa(labels_a, labels_b):
    """Cohen's kappa for two annotators over aligned categorical labels."""
    assert len(labels_a) == len(labels_b) and labels_a, "need aligned non-empty labels"
    cats = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    po = sum(1 for a, b in zip(labels_a, labels_b, strict=False) if a == b) / n
    pa = {c: labels_a.count(c) / n for c in cats}
    pb = {c: labels_b.count(c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {
        "kappa": kappa,
        "observed_agreement": po,
        "expected_agreement": pe,
        "n": n,
        "categories": cats,
    }


def agreement_report(records_a, records_b):
    """Align two annotators' labeling sheets by claim_id and report kappa on the
    overlapping (double-labeled) subset."""
    la = {r["claim_id"]: r["human_label"] for r in records_a if r.get("human_label")}
    lb = {r["claim_id"]: r["human_label"] for r in records_b if r.get("human_label")}
    shared = sorted(set(la) & set(lb))
    if not shared:
        return {"error": "no overlapping labeled claims"}
    a = [la[k] for k in shared]
    b = [lb[k] for k in shared]
    rep = cohens_kappa(a, b)
    rep["n_double_labeled"] = len(shared)
    return rep


def detection_metrics(labeled_records, positive="incorrect"):
    """Once humans have labeled, compare verifier verdict to ground truth.
    Verifier 'fail' == predicted-incorrect; human 'incorrect' == actually-incorrect.
    Returns per-type and overall precision/recall/F1 (spec Section 3.3)."""
    from collections import defaultdict

    buckets = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for r in labeled_records:
        hl = r.get("human_label", "")
        if hl not in ("correct", "incorrect"):
            continue  # unverifiable / unlabeled excluded from P/R
        pred_pos = r["verifier_verdict"] == "fail"
        true_pos = hl == positive
        b = buckets[r["type"]]
        key = (
            "tp"
            if (pred_pos and true_pos)
            else "fp"
            if (pred_pos and not true_pos)
            else "fn"
            if (not pred_pos and true_pos)
            else "tn"
        )
        b[key] += 1

    def prf(d):
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        p = tp / (tp + fp) if tp + fp else float("nan")
        r_ = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * r_ / (p + r_) if (p == p and r_ == r_ and p + r_) else float("nan")
        return {"precision": p, "recall": r_, "f1": f1, **d}

    out = {t: prf(d) for t, d in buckets.items()}
    tot = {
        "tp": sum(d["tp"] for d in buckets.values()),
        "fp": sum(d["fp"] for d in buckets.values()),
        "fn": sum(d["fn"] for d in buckets.values()),
        "tn": sum(d["tn"] for d in buckets.values()),
    }
    out["_overall"] = prf(tot)
    return out
