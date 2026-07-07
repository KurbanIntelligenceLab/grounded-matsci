"""
Evidence-grounded PRE-ANNOTATION for the labeling corpus (assists spec Section 2).

WHAT THIS IS AND IS NOT.
  This module does NOT produce ground-truth labels. The spec (Section 2 / Section
  6) prohibits LLM-as-judge labeling: the authoritative label for every claim is
  a human domain expert's. What this module does is *pre-annotation* -- it gathers
  external EVIDENCE for each claim and attaches a PROVISIONAL suggestion, so the
  human annotator adjudicates in seconds instead of minutes.

  The distinction that keeps this within the spec:
    - A database hit (PubChem formula, Materials Project space group, CCCBDB
      dipole, RDKit valence) is retrieved FACT, not an LLM judgment. These are
      marked authority='database' and are effectively ground-truth-grade.
    - A literature read (an LLM deciding whether a retrieved abstract supports a
      claim) IS an LLM judgment. It is marked authority='literature_llm',
      confidence is capped, and it NEVER stands as a label -- it only routes the
      human's attention.

  Guardrails (all reported in the paper):
    1. human_label is the only field used for detection metrics; provisional_label
       is stored separately and never overwrites it.
    2. agent-vs-human agreement (kappa) is measured on the pilot and reported, so
       reviewers can see the human was not rubber-stamping.
    3. every provisional label carries its evidence + authority + confidence.

MCP calls (OpenAlex/arXiv/PubMed) run in the repl kernel; this module's
`literature_query_plan` builds the queries and `attach_literature_evidence`
consumes results passed back via a handoff file. The deterministic DB evidence
needs no MCP.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# 1. deterministic database / physics evidence (authority = 'database')
# ---------------------------------------------------------------------------
def db_evidence(record):
    """Turn a claim record's existing verifier verdict into a structured evidence
    item. This is retrieved fact, not an LLM opinion."""
    verdict = record.get("verifier_verdict")
    tier = record.get("verifier_tier")
    detail = record.get("verifier_detail", "")
    # map verifier verdict -> provisional label
    if verdict == "fail":
        prov = "incorrect"
    elif verdict == "ok":
        prov = "correct"
    else:
        prov = "unverifiable"
    # authority: tiers 0/1/1.5 are deterministic DB/physics; unchecked is weak
    authority = "database" if verdict in ("ok", "fail") else "none"
    # confidence: DB identity/symmetry checks are near-certain; property checks
    # (DFT-referenced) are softer.
    if authority == "database":  # noqa: SIM108 - keep the two confidence regimes visually separate
        conf = 0.97 if tier in ("0", "1", "1.5") else 0.80
    else:
        conf = 0.0
    return {
        "source": f"verifier tier {tier}",
        "authority": authority,
        "supports_label": prov,
        "confidence": conf,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# 2. literature evidence (authority = 'literature_llm', capped confidence)
# ---------------------------------------------------------------------------
def literature_query_plan(records, only_uncovered=True):
    """Build literature search queries only for claims a database could not
    settle (verifier verdict 'unchecked'), since DB evidence already dominates
    the rest. Returns [{claim_id, query, kind}]."""
    plan = []
    for r in records:
        if only_uncovered and r.get("verifier_verdict") in ("ok", "fail"):
            continue
        ref = r.get("referent") or r.get("span_text")
        typ = r.get("type")
        if typ == "property":
            q = f"{ref} {r.get('value', '')} {r.get('unit', '')} experimental"
        elif typ in ("formula", "smiles"):
            q = f"{ref} molecular structure formula"
        elif typ in ("sg_number", "sg_system", "lattice"):
            q = f"{ref} crystal structure space group"
        else:
            q = str(ref)
        plan.append({"claim_id": r["claim_id"], "query": q.strip(), "kind": typ})
    return plan


def synthesize_literature_label(host, claim_record, papers, max_papers=5):
    """Ask an LLM to judge whether retrieved papers SUPPORT/CONTRADICT/are
    SILENT on a claim. This IS an LLM judgment -> provisional only, low authority.
    `papers` is a list of {title, abstract, doi/url, year}."""
    if not papers:
        return {
            "source": "literature",
            "authority": "literature_llm",
            "supports_label": "unverifiable",
            "confidence": 0.0,
            "detail": "no relevant papers retrieved",
            "citations": [],
        }
    ev = "\n".join(
        f"[{i}] {p.get('title', '')} ({p.get('year', '')}). {(p.get('abstract') or '')[:400]}"
        for i, p in enumerate(papers[:max_papers])
    )
    claim_txt = (
        f'Claim: about "{claim_record.get("referent")}", '
        f"type={claim_record.get('type')}, "
        f"value={claim_record.get('value')} {claim_record.get('unit') or ''}, "
        f'text="{claim_record.get("span_text")}"'
    )
    prompt = (
        "You are helping PRE-SCREEN a chemistry claim for a human annotator. "
        "Based ONLY on the retrieved literature below, does the evidence support "
        "the claim, contradict it, or say nothing conclusive? "
        "Answer with a compact JSON object using these keys: verdict "
        "(one of support, contradict, silent), rationale (one sentence), "
        "cited (list of the [i] indices you used). Do not use angle brackets.\n\n"
        f"{claim_txt}\n\nLiterature:\n{ev}"
    )
    try:
        resp = host.llm(prompt, max_tokens=250)
        txt = resp["text"] if isinstance(resp, dict) else str(resp)
        start, end = txt.find("{"), txt.rfind("}")
        obj = json.loads(txt[start : end + 1]) if start >= 0 else {}
    except Exception as e:
        obj = {"verdict": "silent", "rationale": f"synthesis failed: {e}", "cited": []}
    vmap = {"support": "correct", "contradict": "incorrect", "silent": "unverifiable"}
    prov = vmap.get(str(obj.get("verdict", "silent")).lower(), "unverifiable")
    cited_idx = obj.get("cited", []) or []
    cites = []
    for i in cited_idx:
        if isinstance(i, int) and 0 <= i < len(papers):
            p = papers[i]
            cites.append(
                {
                    "title": p.get("title"),
                    "doi": p.get("doi") or p.get("url"),
                    "year": p.get("year"),
                }
            )
    # confidence capped: this is an LLM read of abstracts, not a database fact
    conf = 0.5 if prov != "unverifiable" else 0.1
    return {
        "source": "literature",
        "authority": "literature_llm",
        "supports_label": prov,
        "confidence": conf,
        "detail": obj.get("rationale", ""),
        "citations": cites,
    }


# ---------------------------------------------------------------------------
# 3. combine evidence -> provisional label (human still decides)
# ---------------------------------------------------------------------------
def combine(record, evidence_items):
    """Pick the highest-authority, highest-confidence evidence as the provisional
    suggestion. Database evidence always outranks literature evidence."""
    rank = {"database": 2, "literature_llm": 1, "none": 0}
    usable = [
        e
        for e in evidence_items
        if e.get("supports_label") != "unverifiable" or e.get("authority") == "database"
    ]
    if not usable:
        usable = evidence_items
    best = max(usable, key=lambda e: (rank.get(e["authority"], 0), e["confidence"]), default=None)
    out = dict(record)
    out["provisional_label"] = best["supports_label"] if best else "unverifiable"
    out["provisional_authority"] = best["authority"] if best else "none"
    out["provisional_confidence"] = round(best["confidence"], 2) if best else 0.0
    out["evidence"] = json.dumps(evidence_items)[:2000]
    # human fields remain blank for the expert to fill / override
    out.setdefault("human_label", "")
    out.setdefault("annotator_id", "")
    return out


def preannotate_db_only(records):
    """Fast path: pre-annotate using only deterministic DB/physics evidence."""
    return [combine(r, [db_evidence(r)]) for r in records]
