# grounded-matsci

Grounded verification for chemical and materials reasoning: a tiered, deterministic
verifier that checks the chemical claims in LLM reasoning traces (molecular formulas,
SMILES, space groups, formation energies, physical constants, isotope half-lives)
against authoritative references (RDKit, PubChem, Materials Project, NIST CCCBDB,
CODATA, IAEA), and closed-loop experiments measuring whether verifier feedback repairs
model errors.

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
uv run grounded-matsci exp1 --config configs/exp1.yaml
uv run grounded-matsci exp4 --config configs/exp4.yaml --debug   # --debug allows a dirty git tree
```

Every run snapshots its config and writes a provenance `manifest.json` under
`outputs/<experiment_id>/`.

## Data availability

- Frozen paper results are committed under `results/`, with SHA-256 hashes and
  provenance in `data/manifests/results_manifest.yaml`.
- The frozen experiment inputs — prompt corpus (registered short sha `66365980`),
  holdout ground truth (`5e80c788`), CODATA/isotope prompt sets and ground truth,
  lookup tables, RAG facts, and the PubChem cache — are committed under `data/frozen/`
  with per-file hashes in `data/manifests/frozen_inputs.yaml`. The isotope ground
  truth (IAEA) is at `data/reference/isotope_gt.json`.
- The §2 labeling-pilot traces and annotation sheets are under `data/pilot/`.
- **Only the raw model-generation traces (~50 MB of JSONL) are not yet committed**;
  they are enumerated in `data/manifests/handoff_missing.yaml` and will be deposited
  upon acceptance. Every experiment subcommand is runnable with the committed inputs
  (API keys required for generation).

## Frozen scientific cores

The claim extractor, adjudicator, tolerance policy, and verifier tiers were frozen
before the holdout runs, with content hashes registered ex ante in a decisions log kept
in the project's private records — e.g. extractor `42e03ae2`, verifier config
`6e45797a`, corpus `66365980`. Those
hashes refer to the pre-refactor bundle files; this repository restructured packaging,
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
tests/            mirrors src/; offline regression + property tests
data/manifests/   dataset + results hashes; the missing-handoff manifest
results/          frozen paper outputs (committed)
outputs/          live-run area (gitignored)
```

## Models

Four current models, verified at run time (two frontier closed, two open-weights):
`anthropic/claude-sonnet-5`, `openai/gpt-5.5`, `deepseek/deepseek-v4-pro`,
`qwen/qwen3-235b-a22b-2507`; plus `anthropic/claude-haiku-4.5` and
`openai/gpt-5.4-mini` in the small-model probe (EXP3).

## Citation

See `CITATION.cff` for software citation metadata (authors, ORCIDs, title, abstract).
A manuscript is in preparation; a journal DOI will be added on publication.

## License

MIT — see `LICENSE`.

## Known limitations

- The raw model-generation traces are not yet committed (see Data availability).
- The DFT tier's timing constant is a single-molecule measurement, not a size-binned
  benchmark (`evaluation/costing.py`).
- Hypothesis property tests and per-function coverage for `workflows/`/`evaluation/`
  are a target, not yet met (see CHANGELOG.md).
