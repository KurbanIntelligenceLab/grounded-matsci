"""
Token and API-cost accounting for the closed loop.

The scientific question this answers, and the one to settle BEFORE scaling:
the closed loop spends *extra* tokens -- Mode A adds correction rounds, Mode B
samples N paths -- so is the factual-accuracy gain worth the token cost? This
module attributes every LLM call to a (condition, prompt) and produces the
per-trace token + USD cost breakdown, plus the physics-tier compute cost, that
feed the "tiered vs flat-DFT vs flat-retrieval" cost figure.

It also compares against the two flat baselines the spec names:
  - flat-DFT: what it would cost to send EVERY checkable quantitative claim to
    DFT (Tier 3) instead of resolving most at Tier 0/1/1.5.
  - flat-retrieval: what it would cost to run a database/literature retrieval for
    every claim regardless of whether a cheaper tier already settled it.

All LLM costs are real USD from OpenRouter's usage accounting; physics costs are
wall-clock-based estimates with a stated $/CPU-hour assumption.
"""

from __future__ import annotations

# Converting local DFT wall-clock into a comparable $ figure.
# CPU_DOLLARS_PER_HOUR is an ASSUMPTION (typical cloud CPU-hour, $0.05-0.10).
# DFT_SECONDS_PER_CLAIM is a single-molecule MEASUREMENT, not a benchmarked
# average: nitrobenzene (15 atoms) b3lyp/def2-TZVP+df dipole = 7.5-8.0 s on this
# 12-core box (re-measured; also produces 4.866 D vs. exp 4.22). It is a rough
# proxy -- real per-claim DFT time scales steeply with atom count, so the figure
# is a floor for small molecules and an underestimate for larger ones. For the
# paper this should be replaced by a size-binned timing benchmark.
CPU_DOLLARS_PER_HOUR = 0.08  # assumption (cloud CPU-hour)
DFT_SECONDS_PER_CLAIM = 8.0  # measured on nitrobenzene (15 atoms); see note


def summarize_calls(call_log):
    """Aggregate a driver call-log into totals."""
    if not call_log:
        return {
            "n_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "wall_s": 0.0,
        }
    return {
        "n_calls": len(call_log),
        "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in call_log),
        "completion_tokens": sum(c.get("completion_tokens", 0) for c in call_log),
        "total_tokens": sum(c.get("total_tokens", 0) for c in call_log),
        "cost_usd": sum(c.get("cost_usd", 0.0) for c in call_log),
        "wall_s": sum(c.get("wall_s", 0.0) for c in call_log),
    }


def dft_dollar(n_claims, seconds_per_claim=DFT_SECONDS_PER_CLAIM):
    return n_claims * seconds_per_claim / 3600.0 * CPU_DOLLARS_PER_HOUR


def tier_cost_breakdown(claim_logs):
    """From per-claim logs (each {tier, cost, wall_clock_s}), tally how many
    claims resolved at each tier and the wall-clock spent. `claim_logs` is a
    flat list of the per-claim 'log' dicts emitted by ground.ground_trace."""
    from collections import defaultdict

    tally = defaultdict(lambda: {"n": 0, "wall_s": 0.0})
    for lg in claim_logs:
        t = lg.get("tier")
        tally[t]["n"] += 1
        tally[t]["wall_s"] += lg.get("wall_clock_s", 0.0)
    return {k: dict(v) for k, v in tally.items()}


def counterfactual_costs(claim_logs):
    """Compare the ACTUAL tiered cost to the two flat baselines.

    Returns dict with dollar estimates for: our tiered physics cost, flat-DFT
    (all quantitative claims to Tier 3), and flat-retrieval (a DB/API call for
    every claim). LLM generation cost is handled separately in per-condition
    accounting; this function is about the *verification* cost only.
    """
    n_total = len(claim_logs)
    # quantitative claims are the ones that could go to DFT (property-type reached
    # a tier of '1.5' or '3'); everything else is Tier 0/1 identity/symmetry.
    n_quant = sum(1 for lg in claim_logs if lg.get("tier") in ("1.5", "3", "3(skipped)"))
    # actual tiered physics cost: only claims that actually hit Tier 3 cost DFT
    n_dft_actual = sum(1 for lg in claim_logs if lg.get("tier") == "3")
    actual_dft_cost = dft_dollar(n_dft_actual)
    # flat-DFT: every quantitative claim pays DFT
    flat_dft_cost = dft_dollar(n_quant)
    # flat-retrieval: assume a database/API round trip per claim; PubChem/MP calls
    # are free of $ but cost latency -- approximate at a nominal API cost.
    NOMINAL_API_DOLLARS = 0.0  # noqa: N806 - PubChem/MP are free; kept explicit for the figure
    flat_retrieval_cost = n_total * NOMINAL_API_DOLLARS
    return {
        "n_claims": n_total,
        "n_quantitative": n_quant,
        "n_dft_actual": n_dft_actual,
        "tiered_dft_cost_usd": actual_dft_cost,
        "flat_dft_cost_usd": flat_dft_cost,
        "dft_savings_usd": flat_dft_cost - actual_dft_cost,
        "dft_calls_avoided": n_quant - n_dft_actual,
        "flat_retrieval_cost_usd": flat_retrieval_cost,
    }


def cost_per_correct(condition_summary, n_correct):
    """Efficiency metric: USD (and tokens) per additional correct answer.
    condition_summary is a summarize_calls() output; n_correct is the count of
    correct end-task answers under that condition."""
    if n_correct <= 0:
        return {"cost_per_correct_usd": float("inf"), "tokens_per_correct": float("inf")}
    return {
        "cost_per_correct_usd": condition_summary["cost_usd"] / n_correct,
        "tokens_per_correct": condition_summary["total_tokens"] / n_correct,
    }
