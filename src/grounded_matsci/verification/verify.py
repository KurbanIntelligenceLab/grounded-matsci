"""
Tiered verification of extracted chemical claims.

Design principle: run the CHEAPEST check that can falsify a claim first, and
only escalate to expensive physics for claims that survive and that carry a
quantitative assertion worth confirming.

    Tier 0  syntactic   RDKit sanitize            microseconds
    Tier 1  identity     RDKit canonical / formula  milliseconds
    Tier 3  physics      PySCF DFT (dipole, gap)    seconds-minutes

Each verifier returns (status, detail):
    status in {"ok", "fail", "warn", "unchecked"}
"""

from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")  # silence RDKit's parse chatter


# --------------------------------------------------------------------------
# Tier 0 + 1 : SMILES syntactic + identity
# --------------------------------------------------------------------------
def verify_smiles(smiles: str):
    """Parse + sanitize. Returns (status, detail, canonical|None, mol|None)."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return "fail", "unparseable SMILES (RDKit could not read tokens)", None, None
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        msg = str(e).strip().splitlines()[-1] if str(e) else "sanitization failed"
        return "fail", f"illegal chemistry: {msg}", None, None
    canonical = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    detail = f"valid; canonical={canonical}; formula={formula}; MW={Descriptors.MolWt(mol):.2f}"
    return "ok", detail, canonical, mol


# --------------------------------------------------------------------------
# Tier 1 : name -> formula, checked against a local reference then (optionally)
# an external lookup. The local table keeps the demo self-contained and offline;
# swap `lookup_formula_external` for a PubChem MCP call in production.
# --------------------------------------------------------------------------
# name -> (canonical SMILES, expected Hill formula)
REFERENCE = {
    "water": ("O", "H2O"),
    "methane": ("C", "CH4"),
    "ammonia": ("N", "H3N"),
    "benzene": ("c1ccccc1", "C6H6"),
    "ethanol": ("CCO", "C2H6O"),
    "acetic acid": ("CC(=O)O", "C2H4O2"),
    "carbon dioxide": ("O=C=O", "CO2"),
    "aspirin": ("CC(=O)Oc1ccccc1C(=O)O", "C9H8O4"),
    "caffeine": ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C8H10N4O2"),
    "glucose": ("OCC1OC(O)C(O)C(O)C1O", "C6H12O6"),
    "toluene": ("Cc1ccccc1", "C7H8"),
    "phenol": ("Oc1ccccc1", "C6H6O"),
    "acetone": ("CC(C)=O", "C3H6O"),
    "formaldehyde": ("C=O", "CH2O"),
    "hydrogen peroxide": ("OO", "H2O2"),
}


def _hill_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return rdMolDescriptors.CalcMolFormula(mol)


def _normalize_formula(f):
    """Canonicalize a formula string to a comparable element->count dict."""
    import re

    counts = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        if not el:
            continue
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts


def verify_formula(name: str, formula: str, external_lookup=None):
    ref = REFERENCE.get(name)
    if ref is None and external_lookup is not None:
        ref = external_lookup(name)  # e.g. a PubChem MCP call
    if ref is None:
        return "unchecked", f"'{name}' not in reference set; cannot confirm formula"
    _ref_smiles, ref_formula = ref
    if _normalize_formula(formula) == _normalize_formula(ref_formula):
        return "ok", f"formula matches reference ({ref_formula})"
    return ("fail", f"formula MISMATCH: trace says {formula}, reference {name} is {ref_formula}")


def name_to_smiles(name, external_lookup=None):
    ref = REFERENCE.get(name)
    if ref is None and external_lookup is not None:
        ref = external_lookup(name)
    return ref[0] if ref else None


# --------------------------------------------------------------------------
# Tier 3 : physics. Compute a reference value with PySCF and compare to the
# claimed value. Guarded so the module imports even if PySCF is absent.
# --------------------------------------------------------------------------
def compute_property_pyscf(smiles: str, prop: str, xc="b3lyp", basis="def2-TZVP", density_fit=True):
    """Return (value, unit, info) computed from first principles, or raise.

    Default basis is def2-TZVP (the policy requires >= def2-TZVP; STO-3G
    is prohibited in reported results). Density fitting is on by default to keep
    the larger basis affordable. `info` carries method/basis and wall-clock.
    """
    import time

    import numpy as np
    from pyscf import dft, gto
    from rdkit.Chem import AllChem

    _t0 = time.time()

    mol_rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
    # Try several seeds; a single failed embedding can yield a degenerate
    # geometry (and a spurious ~0 dipole), so we don't trust the first attempt.
    params = AllChem.ETKDGv3()
    embedded = False
    for seed in (1, 7, 42, 2024):
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol_rd, params) == 0:
            embedded = True
            break
    if not embedded:
        raise RuntimeError("3D embedding failed for all seeds")
    if AllChem.MMFFOptimizeMolecule(mol_rd) not in (0, 1):
        # MMFF params unavailable -> fall back to UFF rather than an unrelaxed geom
        AllChem.UFFOptimizeMolecule(mol_rd)
    conf = mol_rd.GetConformer()
    atoms = []
    for a in mol_rd.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        atoms.append((a.GetSymbol(), (p.x, p.y, p.z)))

    mol = gto.M(atom=atoms, basis=basis, unit="Angstrom", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    if density_fit:
        mf = mf.density_fit()
    mf.kernel()
    dt = time.time() - _t0
    info = f"{xc}/{basis}{'+df' if density_fit else ''} ({dt:.1f}s)"

    if prop == "dipole_moment":
        d = mf.dip_moment(unit="Debye")  # returns vector in Debye
        return float(np.linalg.norm(d)), "debye", info
    if prop == "homo_lumo_gap":
        mo_e = mf.mo_energy
        occ = mf.mo_occ
        homo = mo_e[occ > 0].max()
        lumo = mo_e[occ == 0].min()
        return float((lumo - homo) * 27.2114), "ev", info
    raise ValueError(f"unknown property {prop}")


def verify_property(
    name, prop, value, unit, external_lookup=None, enable_physics=True, return_meta=False
):
    """Verify a quantitative property claim with tier escalation.

    Order: Tier 1.5 experimental/tabulated reference FIRST
    (free, and a better standard than cheap DFT); fall through to Tier 3 DFT
    only when no tabulated value exists. Tolerances come from the central
    policy (domain/tolerances.py), never tuned on eval data.
    """
    from grounded_matsci.domain import refdata, tolerances

    meta = {
        "tier": None,
        "reference": None,
        "ref_source": None,
        "wall_clock_s": 0.0,
        "rel_err": None,
    }

    # ---- Tier 1.5: experimental / tabulated reference ----
    ref = refdata.reference_property(name, prop)
    if ref is not None:
        ref_val, ref_unit, prov = ref
        ok, rel, rule = tolerances.within_tolerance(prop, value, ref_val, experimental=True)
        meta.update(tier="1.5", reference=ref_val, ref_source=f"exp: {prov}", rel_err=rel)
        verdict = "ok" if ok else "fail"
        detail = (
            f"claimed {value} {unit}; experimental {ref_val} {ref_unit} "
            f"({prov}); rel.err={rel:.0%} [{rule}]"
        )
        return (verdict, detail, meta) if return_meta else (verdict, detail)

    # ---- Tier 3: DFT ----
    smiles = name_to_smiles(name, external_lookup)
    if smiles is None:
        meta.update(tier="none")
        r = ("unchecked", f"no structure or reference for '{name}'; cannot check {prop}")
        return (*r, meta) if return_meta else r
    if not enable_physics:
        meta.update(tier="3(skipped)")
        r = ("unchecked", "physics tier disabled; no tabulated reference")
        return (*r, meta) if return_meta else r
    try:
        import time

        t0 = time.time()
        ref_val, ref_unit, info = compute_property_pyscf(smiles, prop)
        wall = time.time() - t0
    except Exception as e:
        meta.update(tier="3")
        r = ("unchecked", f"DFT computation failed: {e}")
        return (*r, meta) if return_meta else r
    ok, rel, rule = tolerances.within_tolerance(prop, value, ref_val, experimental=False)
    meta.update(tier="3", reference=ref_val, ref_source=info, wall_clock_s=wall, rel_err=rel)
    verdict = "ok" if ok else "fail"
    detail = (
        f"claimed {value} {unit}; DFT({info}) gives {ref_val:.2f} {ref_unit}; "
        f"rel.err={rel:.0%} [{rule}]"
    )
    return (verdict, detail, meta) if return_meta else (verdict, detail)
