"""Triplicate reporting: per arm per surface, ITT / err-given-commit / no-commit / detection
recall / flag rate, with McNemar + bootstrap CIs across tier-active reps {1,2,3}. Item-3
discipline: never print ITT and err-given-commit in one column without labels; no-commit rate
sits beside every error rate; identical no-commit scoring across arms."""

from __future__ import annotations

import random
from math import comb

SURFACES = ["molecular", "crystalline", "property_ef", "property_dipole", "reasoning_system"]
CONDS = ["self_critique", "rag_in_prompt", "mode_a", "mode_b"]


def summarize_rep(graded, base):
    bmap = {(r["model"], r["subject"], r["claim_type"]): r["correct"] for r in base}
    out = {}
    for ct in SURFACES:
        out[ct] = {}
        for cond in CONDS:
            cells = [r for r in graded if r["condition"] == cond and r["claim_type"] == ct]
            n = len(cells)
            commit = [r for r in cells if r["correct"] is not None and not r.get("dropped")]
            wrong = sum(1 for r in commit if r["correct"] is False)
            nocommit = sum(1 for r in cells if r["correct"] is None and not r.get("dropped"))
            ndrop = [r for r in cells if not r.get("dropped")]
            rec = {
                "n": n,
                "err_commit": (wrong / len(commit) if commit else None),
                "err_commit_wc": (wrong, len(commit)),
                "nocommit_rate": (nocommit / len(ndrop) if ndrop else None),
                "itt": ((wrong + nocommit) / len(ndrop) if ndrop else None),
            }
            if cond == "mode_a":
                flagged = sum(1 for r in cells if r.get("flagged"))
                bw = [r for r in cells if bmap.get((r["model"], r["subject"], ct)) is False]
                rec["flag_rate"] = flagged / n if n else None
                rec["det_recall"] = (sum(1 for r in bw if r.get("flagged")), len(bw))
            out[ct][cond] = rec
    return out


def _mean(xs):
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None


def _boot_ci(per_rep_rates, iters=2000, seed=42):
    """Bootstrap a 95% CI over replicate-level rates (one rate per rep)."""
    vals = [r for r in per_rep_rates if r is not None]
    if len(vals) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(sum(s := [rng.choice(vals) for _ in vals]) / len(s) for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def mcnemar_p(b, c):
    """Two-sided exact McNemar p for discordant pair counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2**n))


def mcnemar_pairs(base, arm_graded, ct):
    """Pairwise baseline-vs-arm discordant counts on committed cells, keyed (model,subject).
    b = baseline-correct & arm-wrong; c = baseline-wrong & arm-correct."""
    bmap = {(r["model"], r["subject"]): r["correct"] for r in base if r["claim_type"] == ct}
    b = c = 0
    for r in arm_graded:
        if r["claim_type"] != ct or r["correct"] is None or r.get("dropped"):
            continue
        bc = bmap.get((r["model"], r["subject"]))
        if bc is True and r["correct"] is False:
            b += 1
        elif bc is False and r["correct"] is True:
            c += 1
    return b, c


def _outcome(r):
    if r.get("dropped"):
        return None
    return "correct" if r["correct"] is True else ("wrong" if r["correct"] is False else "abstain")


def aggregated_mcnemar(base, graded_reps, cond, ct):
    """Corrected inference: aggregate each unit (subject,claim_type,model)'s replicates to ONE
    majority outcome, then McNemar on aggregated units — avoids pseudoreplication (repeated
    measures on the same prompt are not independent matched pairs). Returns (b, c, p)."""
    from collections import Counter, defaultdict

    bmap = {(r["subject"], r["claim_type"], r["model"]): _outcome(r) for r in base}
    per = defaultdict(list)
    for rep in graded_reps:
        for r in rep:
            if r["condition"] == cond and r["claim_type"] == ct:
                per[(r["subject"], r["claim_type"], r["model"])].append(_outcome(r))
    b = c = 0
    for u, outs in per.items():
        cc = Counter(o for o in outs if o is not None)
        if not cc:
            continue
        ao = cc.most_common(1)[0][0]
        bo = bmap.get(u)
        if bo == "correct" and ao == "wrong":
            b += 1
        elif bo == "wrong" and ao == "correct":
            c += 1
    return b, c, mcnemar_p(b, c)


def clustered_bootstrap(graded_reps, cond, ct, metric="itt", iters=2000, seed=1):
    """Prompt-clustered bootstrap 95% CI: resample SUBJECTS (a subject's replicates travel
    together), so the CI respects the repeated-measures structure. metric 'itt' or 'err_commit'."""
    from collections import defaultdict

    subj = defaultdict(list)
    for rep in graded_reps:
        for r in rep:
            if r["condition"] == cond and r["claim_type"] == ct:
                subj[r["subject"]].append(_outcome(r))
    subs = list(subj)
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        flat = [x for _ in subs for x in subj[rng.choice(subs)]]
        o = [x for x in flat if x is not None]
        if not o:
            continue
        wrong = sum(1 for x in o if x == "wrong")
        cor = sum(1 for x in o if x == "correct")
        ab = sum(1 for x in o if x == "abstain")
        v = (
            (wrong + ab) / len(o)
            if metric == "itt"
            else (wrong / (cor + wrong) if cor + wrong else None)
        )
        if v is not None:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]) if vals else (None, None)


def triplicate(graded_reps, base):
    """graded_reps: list of graded-cell lists (tier-active reps 1,2,3). Returns pooled table."""
    per = [summarize_rep(g, base) for g in graded_reps]
    table = {}
    for ct in SURFACES:
        table[ct] = {}
        for cond in CONDS:
            itt = [p[ct][cond]["itt"] for p in per]
            ec = [p[ct][cond]["err_commit"] for p in per]
            nc = [p[ct][cond]["nocommit_rate"] for p in per]
            row = {
                "itt_mean": _mean(itt),
                "itt_ci": _boot_ci(itt),
                "err_commit_mean": _mean(ec),
                "err_commit_ci": _boot_ci(ec),
                "nocommit_mean": _mean(nc),
            }
            if cond == "mode_a":
                fr = [p[ct][cond].get("flag_rate") for p in per]
                dr = [p[ct][cond]["det_recall"] for p in per]
                row["flag_rate_mean"] = _mean(fr)
                row["det_recall_pooled"] = (sum(x[0] for x in dr), sum(x[1] for x in dr))
            table[ct][cond] = row
    return table
