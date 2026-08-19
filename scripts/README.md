# Scripts

Experiment entry points are **not** here. Every experiment runs as a CLI subcommand,

```bash
uv run grounded-matsci <subcommand> --config configs/<name>.yaml
```

driven by a versioned config (see [`../EXPERIMENTS.md`](../EXPERIMENTS.md) and
`src/grounded_matsci/cli.py`). No new generation happens in this directory.

What lives here is reanalysis and audit over already-committed artifacts. Each script is a
thin entry point: the scientific logic it calls belongs to the package, so that the same
code paths are exercised as in the experiment runners.

| Script | Purpose | Inputs |
|---|---|---|
| `run_semantic_entropy.py` | regenerate the semantic-entropy detector results | `data/frozen/` (committed) |
| `grade_cove.py` | grade the chain-of-verification arm with the frozen grader | `data/frozen/` (committed) |
| `sync_pubchem_cache_from_gt.py` | build a PubChem cache offline from committed ground truth | `data/frozen/` (committed) |
| `audit_tier_shares.py` | recompute verifier tier shares from claim-level logs | `data/traces/`, `data/frozen/` (committed) |
| `audit_calibration.py` | re-derive the calibration ECE figures | per-item rows not in the archive |

`audit_calibration.py` needs the per-item trust/correctness rows enumerated in
`data/manifests/pending_traces.yaml`, which are not in the project archive; it is the one
entry point here that cannot be run from a clean clone.
