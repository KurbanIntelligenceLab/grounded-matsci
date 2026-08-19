"""Regression tests for the closed-loop conditions. Assertions moved unchanged from the
pre-refactor regression suite; the fake generators keep the suite offline."""

from grounded_matsci.workflows import loop


# ---------- Mode B temperature-consistency (root-cause of the 2->3 "regression") ----------
def test_mode_b_returns_matched_temperature_baseline():
    """The Mode B regression was a temperature-inconsistent COMPARISON: best-of-N
    (sampled) vs a greedy single decode. run_mode_b must now return a matched
    baseline drawn with sample=True (same temperature as the reranked candidates),
    and it must be one of the candidates so the comparison isolates reranking."""
    # fake generator: records the `sample` flag it was called with; returns a
    # valid molecular trace so verification runs deterministically (no API).
    calls = []

    def fake_generate(messages, sample=False):
        calls.append(sample)
        return "Benzene has SMILES c1ccccc1 and molecular formula C6H6."

    res = loop.run_mode_b("q", fake_generate, n=4, gt_kwargs={}, system=None)
    # every candidate draw must be a sampled draw (sample=True), never greedy
    assert all(calls), "Mode B must sample all candidates at temperature (sample=True)"
    # a matched baseline must be returned and must be one of the candidates
    assert res["matched_baseline"] is not None
    assert res["matched_baseline"] in res["candidates"]
    # best is selected by trust; with identical texts it equals the matched draw's fails
    assert res["best"]["n_fail"] == res["matched_baseline"]["n_fail"]


def test_rag_and_self_critique_conditions_exist():
    """Conditions 2 (self-critique) and 3 (RAG-in-prompt) are implemented."""
    assert hasattr(loop, "run_self_critique"), "condition 2 missing"
    assert hasattr(loop, "run_rag_in_prompt"), "condition 3 missing"
    # RAG prepends facts and does not loop
    calls = []

    def fake_gen(messages, sample=False):
        calls.append(messages)
        return "aspirin has formula C9H8O4"

    r = loop.run_rag_in_prompt(
        "What is aspirin?",
        fake_gen,
        fact_lookup=lambda p: "- aspirin: C9H8O4",
        gt_kwargs={"enable_physics": False},
    )
    assert r["mode"] == "rag_in_prompt"
    assert "C9H8O4" in r["retrieved_facts"]
    assert len(calls) == 1, "RAG-in-prompt must generate once (no loop)"
    assert "Reference facts" in calls[0][-1]["content"], "facts not prepended"
