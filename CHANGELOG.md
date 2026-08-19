# Changelog

## Unreleased

- Local-only agent config and manuscript sources are no longer part of the
  repository: gitignored, and all published docs now reference only committed paths.
- Populated `CITATION.cff` with authors, ORCIDs, title, and abstract.

## 0.1.0 — 2026-07-07

Initial migration of the paper bundle into this repository.

- Restructured the flat `grounded_matsci/` bundle package into a `src/` layout
  with `domain/`, `verification/`, `io/`, `evaluation/`, and `workflows/`
  sub-packages. Scientific logic, constants, regexes, prompts, and control flow
  are unchanged; only packaging, imports, typing, and configuration plumbing
  moved.
- Replaced `requirements.txt` with `pyproject.toml` + `uv.lock` (uv-managed,
  Python 3.12 pinned).
- Deduplicated the OpenRouter client code shared by the experiment runners into
  `io/openrouter.py` (the two distinct call families are kept separate) and the
  JSONL checkpoint-resume helpers into `io/checkpoint.py`.
- Moved runtime experiment parameters (model lists, system prompts, worker
  counts, paths) into versioned YAML under `configs/`, validated into frozen
  dataclasses. Frozen scientific constants (tolerances, thresholds, reference
  tables) stay in code.
- Added `cli.py` entry point (`uv run grounded-matsci <subcommand>`),
  `reproducibility.py` (seeding + output manifests), and `logging_config.py`.
- Migrated the regression suite to `tests/` mirroring `src/`, removing the
  `sys.path` hack. Assertions are unchanged. Added minimal config, seeding,
  and data-manifest hash tests.
- Added LICENSE (MIT), CITATION.cff, data manifests, and this changelog.

Post-migration: imported the frozen experiment inputs (corpus, holdout GT, CODATA and
isotope prompt sets and ground truth, lookups, RAG facts, PubChem cache) into
`data/frozen/` and the labeling-pilot artifacts into `data/pilot/`, verified against
the ex-ante registered hashes (corpus `66365980`, holdout GT `5e80c788`, CODATA-2022
constants payload `95e59c7a`); configs now point at the committed inputs and write to
`outputs/`.

Known deferred items:

- The raw model-generation trace JSONLs (~50 MB) are not in this repository; see
  `data/manifests/pending_traces.yaml`.
- Hypothesis property tests and per-public-function coverage for
  `workflows/` and `evaluation/` are a target, not yet met.
