"""
PubChem-backed Tier-1 identity provider.

The chemistry MCP resolves a compound name to its authoritative CID, molecular
formula, and canonical SMILES. MCP calls can only be issued from the `repl`
tool, whereas the verifiers run in the analysis kernel. We bridge the two with a
small on-disk cache:

    1. `resolve_names(names)` runs in a `repl` cell, hits PubChem via the MCP,
       and writes {name -> {formula, smiles, cid, inchikey}} to a JSON cache.
    2. `make_lookup(cache_path)` builds an `external_lookup(name)` callable that
       the verifiers consume -- returning (canonical_smiles, hill_formula) or
       None, exactly the contract `verify.verify_formula` expects.

This keeps the physics/identity kernel offline-deterministic while letting the
reference facts come from a live authoritative database instead of a hand-built
table.

The cache path is an explicit parameter (configured in `configs/pubchem.yaml`);
the original bundle hardcoded `handoff/pubchem_cache.json`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


# ---- runs in the repl tool (has host.mcp) --------------------------------
def resolve_names(
    host: Any,  # noqa: ANN401 - MCP host object, no stable type
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


# ---- runs in the analysis kernel (no MCP needed) -------------------------
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
        if not rec or not rec.get("smiles") or not rec.get("formula"):
            return None
        smi = rec["smiles"]
        if _canon:
            smi = _canon(smi)
        return smi, rec["formula"]

    return lookup
