"""
PubChem-backed Tier-1 identity provider.

Resolving a compound name to its authoritative CID, molecular formula, and canonical
SMILES requires a network call, but verification must be offline and deterministic. The
two are bridged by a small on-disk cache:

    1. `resolve_names(names)` queries PubChem through an external resolver and writes
       {name -> {formula, smiles, cid, inchikey}} to a JSON cache. This is the only
       step that touches the network.
    2. `make_lookup(cache_path)` builds an `external_lookup(name)` callable that the
       verifiers consume -- returning (canonical_smiles, hill_formula) or None, exactly
       the contract `verify.verify_formula` expects.

Reference facts therefore come from a live authoritative database rather than a
hand-built table, while verification itself stays reproducible from the committed cache.
The paper runs used `data/frozen/pubchem_cache.json`, which is committed; the cache path
is a parameter, configured in `configs/pubchem.yaml`. Populating a cache from the
committed ground truth, without any network access, is what
`scripts/sync_pubchem_cache_from_gt.py` does.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


# ---- online: populates the cache -----------------------------------------
def resolve_names(
    host: Any,  # noqa: ANN401 - resolver client, no stable type
    names: Iterable[str],
    cache_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Resolve names via the chemistry MCP and persist to a JSON cache.

    Call from a `repl` cell where `host` is in scope. Merges into any existing
    cache so repeated traces accumulate resolved facts.
    """
    cache_file = Path(cache_path)
    cache: dict[str, dict[str, Any]] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
    for nm in names:
        key = nm.strip().lower()
        if key in cache:
            continue
        rec: dict[str, Any] = {
            "formula": None,
            "smiles": None,
            "cid": None,
            "inchikey": None,
            "error": None,
        }
        try:
            r = host.mcp(
                "chemistry",
                "pubchem_search_compounds",
                query=nm,
                namespace="name",
                max_cids=1,
                with_properties=True,
            )
            if isinstance(r, str):  # MCP surfaced an error string
                rec["error"] = r[:200]
            else:
                props = r.get("properties") or []
                p = props[0] if props else {}
                rec.update(
                    cid=(r.get("cids") or [None])[0],
                    formula=p.get("MolecularFormula"),
                    # ConnectivitySMILES is the stereo-stripped connectivity
                    smiles=p.get("ConnectivitySMILES") or p.get("SMILES"),
                    inchikey=p.get("InChIKey"),
                )
                if not rec["cid"]:
                    rec["error"] = "no CID match"
        except Exception as e:
            rec["error"] = str(e)[:200]
        cache[key] = rec
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=2))
    return cache


# ---- offline: reads the cache --------------------------------------------
def make_lookup(
    cache_path: str | Path, canonicalize: bool = True
) -> Callable[[str], tuple[str, str] | None]:
    """Return external_lookup(name) -> (canonical_smiles, formula) | None.

    PubChem returns Kekulized SMILES (C1=CC=CC=C1); we canonicalize through
    RDKit so aromatic forms compare equal and the formula is Hill-normalized.
    """
    cache_file = Path(cache_path)
    cache: dict[str, dict[str, Any]] = (
        json.loads(cache_file.read_text()) if cache_file.exists() else {}
    )
    _canon: Callable[[str], str] | None = None
    if canonicalize:
        try:
            from rdkit import Chem, RDLogger

            RDLogger.DisableLog("rdApp.*")

            def _canon_impl(smi: str) -> str:
                m = Chem.MolFromSmiles(smi)
                return str(Chem.MolToSmiles(m)) if m else smi

            _canon = _canon_impl
        except Exception:
            _canon = None

    def lookup(name: str) -> tuple[str, str] | None:
        rec = cache.get(name.strip().lower())
        if not rec or not rec.get("formula"):
            return None
        smi = rec.get("smiles") or ""
        if smi and _canon:
            smi = _canon(smi)
        return smi, rec["formula"]

    return lookup


def _gt_formulas(gt_path: str | Path) -> dict[str, str]:
    """Extract {name_lower -> formula} from holdout ground-truth JSON."""
    raw = json.loads(Path(gt_path).read_text())
    out: dict[str, str] = {}

    def ingest(block: object) -> None:
        if not isinstance(block, dict):
            return
        for key, val in block.items():
            if key.startswith("_"):
                continue
            if isinstance(val, dict) and val.get("formula") and "cid" in val:
                out[key.strip().lower()] = val["formula"]
            elif isinstance(val, dict) and not val.get("formula"):
                ingest(val)

    ingest(raw)
    if "molecular" in raw and isinstance(raw["molecular"], dict):
        ingest(raw["molecular"])
    if "crystalline" in raw and isinstance(raw["crystalline"], dict):
        ingest(raw["crystalline"])
    return out


def make_combined_lookup(
    cache_path: str | Path | None = None,
    gt_path: str | Path | None = None,
) -> Callable[[str], tuple[str, str] | None]:
    """PubChem cache first, then holdout GT formula fallback (formula-only OK)."""
    pubchem = make_lookup(cache_path) if cache_path and Path(cache_path).exists() else None
    gt = _gt_formulas(gt_path) if gt_path and Path(gt_path).exists() else {}

    def lookup(name: str) -> tuple[str, str] | None:
        if pubchem is not None:
            hit = pubchem(name)
            if hit is not None:
                return hit
        formula = gt.get(name.strip().lower())
        if formula:
            return ("", formula)
        return None

    return lookup
