# grounded-matsci

**Grounded verification of chemical and materials reasoning: detection is the bottleneck**

Language models are moving into chemistry and materials discovery workflows, where a wrong
molecular formula, space group, or formation energy can silently propagate into downstream
decisions. These confabulations hide inside fluent reasoning traces and concentrate on
rare, long-tail entities, where model confidence is least trustworthy. Retrieving reference
data for every prompt would catch them, but at a heavy coverage and abstention cost.

This repository implements the tiered verifier and the closed-loop experiments behind that
result. The verifier extracts each checkable claim from a reasoning trace — molecular
formulas, SMILES, space groups, formation energies, physical constants, isotope half-lives
— tests it against authoritative databases and physical law (RDKit, PubChem, Materials
Project, NIST CCCBDB, CODATA, IAEA), and retrieves a reference value **only when a check
fails**. Across four models and over five hundred condition-pinned prompts, gated
correction cuts the error rate of committed formulas from 22% to 4% with 3.2× fewer
retrievals than blanket augmentation, and outperforms a conversational retrieval oracle
when every answer, corrected or not, is scored. When a flag fires, repair almost always
succeeds: the binding constraint is detection, not repair.

## Installation

Requires Python 3.12 (exact pin — this is a paper-reproduction repository) and
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Note that rdkit, pyscf, and pymatgen are heavy compiled dependencies; the first sync
downloads several hundred MB. Exact versions are locked in `uv.lock`.

Secrets are environment variables only, never config values:

```bash
export OPENROUTER_API_KEY=...   # LLM generation (all runners)
export MP_API_KEY=...           # Materials Project tier (optional; degrades gracefully)
```

## Reproduce

The regression suite runs fully offline (no API keys, no DFT):

```bash
uv run pytest
```

Experiment runners are CLI subcommands, each driven by a versioned YAML config:

```bash
uv run grounded-matsci --help
uv run grounded-matsci two-stage-triage --config configs/two_stage_triage.yaml
uv run grounded-matsci isotopes --config configs/isotopes.yaml --debug   # --debug allows a dirty tree
```

Every run snapshots its config and writes a provenance `manifest.json` under
`outputs/<experiment_id>/`.

**[EXPERIMENTS.md](EXPERIMENTS.md) maps every subcommand to what it measures, the config
that drives it, and the committed result it reproduces** — start there. It also defines
the verifier tiers, the five experimental conditions, and the rarity strata that the
module docstrings assume.

## Data availability

- Frozen paper results are committed under `results/`, with SHA-256 hashes and
  provenance in `data/manifests/results_manifest.yaml`.
- The frozen experiment inputs — prompt corpus (registered short sha `66365980`),
  holdout ground truth (`5e80c788`), CODATA/isotope prompt sets and ground truth,
  lookup tables, RAG facts, and the PubChem cache — are committed under `data/frozen/`
  with per-file hashes in `data/manifests/frozen_inputs.yaml`. The isotope ground
  truth (IAEA) is at `data/reference/isotope_gt.json`.
- The labeling-pilot traces and annotation sheets are under `data/pilot/`.
- The raw model-generation traces (~49 MB of JSONL) are committed under `data/traces/`,
  hashed in `data/manifests/traces.yaml`: one row per (prompt, model, condition,
  replicate) cell with the model's final text and the in-loop verifier verdicts. The
  sampling-consistency samples are `data/frozen/selfcheck_samples.jsonl`.
- Two artifacts are still missing and are enumerated in
  `data/manifests/pending_traces.yaml`: the inline-confidence raw traces and the
  calibration audit's per-item rows.

Every experiment subcommand is runnable with the committed inputs (API keys required for
generation).

## Frozen scientific cores

The claim extractor, adjudicator, tolerance policy, and verifier tiers were frozen
before the holdout runs, with content hashes registered ex ante in a decisions log kept
in the project's private records — e.g. extractor `42e03ae2`, verifier config
`6e45797a`, corpus `66365980`. Those
hashes refer to the pre-refactor files; this repository restructured packaging,
imports, typing, and configuration **without changing behavior**, which is enforced by
the migrated regression suite (`tests/`). Their behavior may not change without explicit
approval. The frozen modules are:

- `domain/{extract,adjudicate,tolerances,refdata,codata,isotope,composition,calibration}.py`
- `verification/{verify,crystal,ground,two_stage}.py`
- `evaluation/{grade_all,grade_expansion}.py`
- `workflows/loop.py`

## Repository layout

```
src/grounded_matsci/
  domain/         pure scientific contract code (extractor, tolerances, graders)
  verification/   tiered verifier stack (RDKit / pymatgen / PySCF) + orchestrator
  io/             OpenRouter client, PubChem cache, Materials Project, checkpoints
  evaluation/     unified/expansion graders, cost accounting, triplicate statistics
  workflows/      generation loops (Mode A/B, self-critique, RAG) + experiment runners
  config.py       YAML -> frozen dataclass validation
  reproducibility.py  seeding + per-run provenance manifests
  cli.py          `grounded-matsci` subcommands
configs/          versioned experiment configs (models, prompts, paths)
scripts/          reanalysis and audit entry points over committed artifacts
tests/            mirrors src/; offline regression + property tests
data/frozen/      frozen experiment inputs (committed)
data/manifests/   dataset + results hashes; the pending-traces manifest
results/          frozen paper outputs (committed)
outputs/          live-run area (gitignored)
```

## Models

Four current models, verified at run time (two frontier closed, two open-weights):
`anthropic/claude-sonnet-5`, `openai/gpt-5.5`, `deepseek/deepseek-v4-pro`,
`qwen/qwen3-235b-a22b-2507`; plus `anthropic/claude-haiku-4.5` and
`openai/gpt-5.4-mini` in the small-model probe (`small-models`).

## Citation

The preprint is on arXiv:

```bibtex
@article{polat2026grounded,
  title   = {Grounded verification of chemical and materials reasoning:
             detection is the bottleneck},
  author  = {Polat, Can and Kurban, Mustafa and Serpedin, Erchin and Kurban, Hasan},
  journal = {arXiv preprint arXiv:2607.17417},
  year    = {2026}
}
```

See `CITATION.cff` for machine-readable citation metadata (authors, ORCIDs, abstract) and
the preferred citation. A journal DOI will be added on publication.

## License

- **Code** — MIT, see [`LICENSE`](LICENSE).
- **Data and results** — Creative Commons Attribution 4.0 International (CC BY 4.0),
  see [`LICENSE-DATA`](LICENSE-DATA). This covers the prompt corpus, the frozen ground
  truth, the lookup and prompt sets under `data/`, and the frozen results under `results/`.

Third-party reference data redistributed here (IAEA, PubChem, Materials Project, NIST
CCCBDB) remain subject to their own source terms, recorded per artifact in
`data/manifests/`.

## Known limitations

- The DFT tier's timing constant is a single-molecule measurement, not a size-binned
  benchmark (`evaluation/costing.py`).
- Hypothesis property tests and per-function coverage for `workflows/`/`evaluation/`
  are a target, not yet met (see CHANGELOG.md).
- `scripts/audit_calibration.py` needs per-item trust/correctness rows that are not in the
  archive (see `data/manifests/pending_traces.yaml`), so the calibration audit cannot be
  re-run from a clean clone.
- `scripts/audit_tier_shares.py` runs against the committed traces but does not reproduce
  the reported verifier tier shares: it re-grounds final answer text, while the reported
  shares describe the tiers exercised in loop during the run. The in-loop tier events are
  not among the released artifacts.
- The figures under `results/figures/` are committed as released artifacts; the plotting
  code that produced them is not yet in this repository.
- Regenerating `results/semantic_entropy_percell.json` can differ from the committed file
  in the last floating-point digit, because `math.log` comes from the host C library. The
  reported summary, thresholds, and precision/recall are unaffected.
