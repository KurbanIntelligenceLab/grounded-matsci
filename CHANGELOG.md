# Changelog

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

Known deferred items:

- The frozen corpus, ground-truth files, raw trace JSONLs, and PubChem cache
  (`handoff/`) are not in this repository; see
  `data/manifests/handoff_missing.yaml`.
- Hypothesis property tests and per-public-function coverage for
  `workflows/` and `evaluation/` are a target, not yet met.
- `paper/figures/make_figs.py` is manuscript tooling and was not refactored.
