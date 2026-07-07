"""
Closed-loop grounding (spec Section 1.4).

Mode A - tool-interleaved: the model generates a reasoning step; we verify every
    chemical claim in it; any FAIL is injected back as a tool/correction message
    and the model revises. Iterated until no failures remain or a step budget is
    hit.
Mode B - generate-and-rerank: sample N independent reasoning paths, score each
    with the trust metric, select the argmax (ties broken by fewer uncheckable
    claims).

Both take a `generate(messages) -> text` callable so the module is model-
agnostic; the OpenRouter driver lives in `io/openrouter.py`. Verification reuses
`ground.ground_trace` with whatever tiers are enabled (PubChem/MP/DFT).
"""

from __future__ import annotations

from grounded_matsci.verification import ground


def _verify(text, **gt_kwargs):
    claims = ground.ground_trace(text, **gt_kwargs)
    fails = [c for c in claims if c.status == "fail"]
    return claims, fails


def trust_and_uncheckable(claims):
    ts = ground.trust_score(claims)
    unchecked = sum(1 for c in claims if c.status not in ("ok", "fail"))
    return (ts if ts is not None else 0.0), unchecked


def _fail_feedback(fails):
    """Correction feedback.

    IMPORTANT (refinement after the first closed-loop run): the instruction is
    constrained to *repair only the flagged claims*. The earlier "correct them
    and continue" wording invited the model to introduce NEW molecules/materials,
    which enlarged the error surface and produced net-new failures. We now tell
    the model explicitly not to add new chemical entities.
    """
    lines = [
        "A verification tool (RDKit / PubChem / Materials Project / DFT) "
        "flagged the following chemical claims in your previous answer as "
        "incorrect. Provide a corrected answer that:",
        "  1. fixes ONLY these specific claims using the tool feedback,",
        "  2. does NOT introduce any new molecules, materials, or examples "
        "not already in your previous answer,",
        "  3. keeps every claim that was not flagged exactly as it was.",
        "",
        "Flagged claims:",
    ]
    for c in fails:
        lines.append(f'  - "{c.raw.strip()}": {c.detail}')
    return "\n".join(lines)


def run_mode_a(prompt, generate, max_rounds=3, gt_kwargs=None, system=None):
    """Tool-interleaved correction. Returns a result dict with the trajectory."""
    gt_kwargs = gt_kwargs or {}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    trajectory = []
    text = generate(messages)
    for rnd in range(max_rounds):
        claims, fails = _verify(text, **gt_kwargs)
        ts, _unchk = trust_and_uncheckable(claims)
        trajectory.append(
            {"round": rnd, "text": text, "n_claims": len(claims), "n_fail": len(fails), "trust": ts}
        )
        if not fails:
            break
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": _fail_feedback(fails)})
        text = generate(messages)
    if not (text or "").strip():
        # empty final decode (reasoning model can exhaust the token budget on a
        # follow-up turn): fall back to the last non-empty answer in the trajectory.
        prior = [t["text"] for t in trajectory if (t["text"] or "").strip()]
        if prior:
            text = prior[-1]
    final_claims, final_fails = _verify(text, **gt_kwargs)
    ts, _unchk = trust_and_uncheckable(final_claims)
    return {
        "mode": "A",
        "final_text": text,
        "final_trust": ts,
        "final_n_fail": len(final_fails),
        "rounds": len(trajectory),
        "trajectory": trajectory,
        "final_claims": final_claims,
    }


def run_self_critique(prompt, generate, gt_kwargs=None, system=None, max_rounds=2):
    """Condition 2 (spec 3.2): self-critique WITHOUT tools (Self-Refine style).

    The model is asked to critique and revise its own answer with NO external
    verifier feedback. Expected null/negative per CRITIC / Huang et al.; reported
    either way. Scored with the SAME verifier post-hoc (for the accuracy metric),
    but the verifier output is NOT shown to the model — that is the whole point of
    the contrast with Mode A.
    """
    gt_kwargs = gt_kwargs or {}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text = generate(messages)
    trajectory = []
    for rnd in range(max_rounds):
        _claims, fails = _verify(text, **gt_kwargs)  # scoring only, hidden from model
        trajectory.append({"round": rnd, "text": text, "n_fail": len(fails)})
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": "Review your previous answer for any chemical or crystallographic "
                "errors (molecular formulas, SMILES, space groups). If you find mistakes, "
                "provide a corrected answer; otherwise restate it. Do not invent new compounds.",
            }
        )
        text = generate(messages)
    if not (text or "").strip():
        prior = [t["text"] for t in trajectory if (t["text"] or "").strip()]
        if prior:
            text = prior[-1]
    final_claims, final_fails = _verify(text, **gt_kwargs)
    return {
        "mode": "self_critique",
        "final_text": text,
        "final_n_fail": len(final_fails),
        "rounds": len(trajectory),
        "trajectory": trajectory,
        "final_claims": final_claims,
    }


def run_rag_in_prompt(prompt, generate, fact_lookup, gt_kwargs=None, system=None):
    """Condition 3 (spec 3.2): RAG-in-prompt — the decisive Mode-A contrast.

    Resolve the entities named in the prompt, retrieve their reference facts
    (PubChem formula / MP or named-phase space group), PREPEND them to the prompt,
    and generate ONCE with no loop. This is the best non-verifier alternative: it
    tests whether plain retrieval achieves what Mode A's verifier-injected reference
    achieves. `fact_lookup(prompt) -> str` returns the retrieved-facts block (built
    by the caller from the same PubChem/MP references used for grading, so the
    head-to-head is fair). Retrieve-everything cost = one retrieval per prompt,
    regardless of whether the model would have erred (contrast: Mode A retrieves
    only on a detected failure).
    """
    gt_kwargs = gt_kwargs or {}
    facts = fact_lookup(prompt) or ""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    aug = (
        f"Reference facts (from PubChem / Materials Project):\n{facts}\n\n{prompt}"
        if facts
        else prompt
    )
    messages.append({"role": "user", "content": aug})
    text = generate(messages)
    claims, fails = _verify(text, **gt_kwargs)
    return {
        "mode": "rag_in_prompt",
        "final_text": text,
        "retrieved_facts": facts,
        "final_n_fail": len(fails),
        "final_claims": claims,
    }


def run_mode_b(prompt, generate, n=8, gt_kwargs=None, system=None, matched_baseline=True):
    """Generate-and-rerank. Returns the best-of-N path by trust score.

    ROOT-CAUSE FIX (the earlier "2->3 regression" investigation):
    The apparent regression was NOT a scoring/selection bug. It was a
    temperature-inconsistency artifact in the *comparison*, not in this function:
    the reranked best-of-N is sampled at the driver's temperature (`sample=True`,
    e.g. 0.7), but it was being compared against a *greedy* single decode
    (`sample=False`, temperature 0.0). Greedy decoding is already near-optimal for
    factual recall, so best-of-N at a higher temperature can legitimately score
    worse than one greedy sample — the difference is the temperature, not the
    reranking.

    Correct experimental design: Mode B's lift must be measured against a
    single sample drawn AT THE SAME temperature (`sample=True`). When
    `matched_baseline=True` we draw that matched single sample here and return it
    as `matched_baseline`, so the caller compares like with like (best-of-N vs.
    one-of-N, both at the sampling temperature). The reported Mode B lift is
    then attributable to the reranking alone.
    """
    gt_kwargs = gt_kwargs or {}
    cands = []
    for i in range(n):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = generate(messages, sample=True)  # driver applies temperature
        claims, fails = _verify(text, **gt_kwargs)
        ts, unchk = trust_and_uncheckable(claims)
        cands.append(
            {
                "i": i,
                "text": text,
                "trust": ts,
                "n_fail": len(fails),
                "n_unchecked": unchk,
                "n_claims": len(claims),
                "claims": claims,
            }
        )
    # argmax trust; tie-break fewer uncheckable, then fewer fails
    best = max(cands, key=lambda d: (d["trust"], -d["n_unchecked"], -d["n_fail"]))
    # Matched-temperature baseline = the FIRST sample (also sample=True). This is
    # one draw from the same distribution the rerank selects over, so
    # (best.n_fail - matched.n_fail) isolates the reranking effect.
    matched = cands[0] if matched_baseline else None
    return {"mode": "B", "n": n, "best": best, "candidates": cands, "matched_baseline": matched}
