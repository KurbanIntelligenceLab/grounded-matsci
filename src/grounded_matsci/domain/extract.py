"""
Claim extraction from an LLM reasoning trace.

We pull three kinds of chemical claims out of free text:

  - SMILES candidates          -> validated downstream by RDKit (Tier 0/1)
  - name = formula assertions  -> checked against structure (Tier 1)
  - name property = value      -> checked against physics (Tier 3)

Extraction is deliberately *permissive*: we over-generate candidates and let
the verifiers reject them. A false "candidate" that RDKit refuses to parse
simply gets dropped; the cost of a missed real claim is higher than the cost
of an extra check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# --- text normalization (applied before ALL extractors) --------------------
# Models emit chemical formulas with Unicode subscripts/superscripts
# ("C₈H₉NO₂", "Al₂O₃") and markdown bold ("**C8H9NO2**").
# The ASCII-only formula/crystal/SMILES regexes silently miss these, which was
# the cause of near-zero extraction recall on formula/space-group/property claims
# in the fresh-batch audit. We fold subscripts/superscripts to ASCII digits and
# strip markdown emphasis BEFORE extraction, preserving character count so span
# offsets stay meaningful (each Unicode digit maps to exactly one ASCII digit).
_SUB = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
}
_SUP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
}
_UNICODE_DIGITS = {**_SUB, **_SUP}


def normalize_text(text: str) -> str:
    """Fold Unicode sub/superscript digits to ASCII and neutralize markdown
    emphasis, length-preserving so char spans remain valid. Also maps the
    Angstrom sign and a couple of common unicode minus/dot variants.

    Hermann-Mauguin inversion axes: models write the bar as a COMBINING OVERLINE
    (U+0304) AFTER the digit -- "R3̄c" renders as R-3-bar-c but the canonical
    ASCII form is "R-3c" (bar BEFORE the digit). We detect "<digit>̄" and
    rewrite it to "-<digit>", which is NOT length-preserving, so this pass runs
    separately and callers of normalize_text that need spans use the digit/markdown
    pass only. extract_all uses the full normalization (spans are recomputed by
    the regexes against the normalized string, so length change is fine there).
    """
    out = []
    for ch in text:
        if ch in _UNICODE_DIGITS:
            out.append(_UNICODE_DIGITS[ch])
        elif ch == "−":  # unicode minus -> ASCII hyphen
            out.append("-")
        elif ch in "*`":  # markdown emphasis/code -> space (len-preserving)
            out.append(" ")
        else:
            out.append(ch)
    s = "".join(out)
    # combining overline on a digit -> ASCII inversion notation: "3̄" -> "-3".
    # Also handle LaTeX \bar{3} / \overline{3} and \(\bar{3}\) wrappers.
    s = re.sub(r"([0-9])̄", r"-\1", s)
    s = re.sub(r"\\+\(?\s*\\?(?:bar|overline)\s*\{?\s*([0-9])\s*\}?\s*\\?\)?", r"-\1", s)
    # LaTeX inline math delimiters "$...$" around a fragment: models write
    # "Fm$-3$m" for Fm-3m. Drop the $ so the HM symbol is contiguous.
    s = s.replace("$", "")
    # approximation sign before a number -> space (value regexes tolerate ws):
    # "≈ 1.04" -> "  1.04"
    s = s.replace("≈", " ")

    # Collapse spaces inside a Hermann-Mauguin symbol that FOLLOWS "space group":
    # models write "space group R 3 c" or "F m -3 m". Scope the collapse to the
    # ~12 chars after the cue so ordinary prose spacing is untouched.
    def _despace_sg(mm: re.Match[str]) -> str:
        head, sym = mm.group(1), mm.group(2)
        sym = re.sub(r"\s+", "", sym)
        sym = sym.rstrip("-")  # don't keep a trailing list-separator hyphen
        return head + sym

    # symbol glyphs are letters/digits/_/ and an internal hyphen; a hyphen must be
    # followed by a digit to stay (inversion axis), else it's a separator -> stop.
    s = re.sub(
        r"(space group\s+(?:symbol\s*[:=]?\s*)?)"
        r"([PABCIFR](?:\s*(?:[a-zA-Z0-9_/̄]|-(?=\s*[0-9̄]))){1,10})",
        _despace_sg,
        s,
        flags=re.IGNORECASE,
    )
    # re-apply overline fix in case the collapse brought "3 ̄" together
    return re.sub(r"([0-9])̄", r"-\1", s)


_NOT_FORMULA_WORDS = {
    "SMILES",
    "INCHI",
    "INCHIKEY",
    "IUPAC",
    "CAS",
    "NIST",
    "CID",
    "PDB",
    "DFT",
    "HOMO",
    "LUMO",
    "NMR",
    "IR",
    "UV",
}


def _preceded_by_smiles(text: str, start: int, window: int = 28) -> bool:
    """True if the ~14 chars before `start` mention SMILES/InChI — the token is a
    structure string being parsed as a formula (e.g. 'SMILES of naproxen is CC...').
    A leading 'COC1' of a SMILES otherwise passes the formula regex."""
    pre = text[max(0, start - window) : start].lower()
    return "smiles" in pre or "inchi" in pre


def _plausible_formula(f: str) -> bool:
    """Reject tokens that match the formula regex but are actually words/labels
    (SMILES, InChI, ...). A real formula either contains a digit, or is a short
    all-caps run that is NOT a known non-formula acronym and has <=3 distinct
    uppercase letters (element symbols), so 'SMILES' (6 caps) is rejected but
    'NaCl'/'CO' pass."""
    if f.upper() in _NOT_FORMULA_WORDS:
        return False
    if re.search(r"\d", f):
        return True
    caps = re.findall(r"[A-Z]", f)
    # element-only formula like "NaCl", "CO", "KBr": few capitals, valid element start
    return 2 <= len(caps) <= 3 and f[0].isupper()


def _nearest_known_name(text: str, upto: int, known_names: list[str] | None) -> str | None:
    """Return the KNOWN compound/material name whose whole-word occurrence ends
    nearest to (at or before) char offset `upto`, else None. Whole-word so
    'benzene' does not match inside 'nitrobenzene'."""
    bound, best = None, -1
    window = text[:upto].lower()
    for kn in known_names or []:
        for mm in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(kn) + r"(?![A-Za-z0-9])", window):
            if mm.end() > best:
                best, bound = mm.end(), kn
    return bound


@dataclass
class Claim:
    """One extracted chemical claim.

    Deliberately MUTABLE (rule C.3 documented reason): the verification layer
    (`verification.ground`) fills `status`/`detail` and augments `payload` after
    construction; freezing this class would change the frozen verifier's behavior.
    """

    kind: str  # "smiles" | "formula" | "property"
    raw: str  # the exact substring found
    span: tuple[int, int]  # (start, end) char offsets in the trace
    payload: dict[str, Any] = field(default_factory=dict)
    # verification results filled in later
    status: str | None = None  # "ok" | "fail" | "warn" | "unchecked"
    detail: str = ""


# ---- SMILES ---------------------------------------------------------------
# Characters that make up a SMILES. We require at least one ring-closure digit
# or a bond/branch token so we don't grab ordinary words like "CCC" acronyms.
_SMILES_CHARS = r"[A-Za-z0-9@+\-\[\]()=#/\\%\.]"
# Hermann-Mauguin space-group symbols look superficially SMILES-like (Fm-3m,
# P6_3mc, Fd-3m, P4_2/mnm). They carry tokens SMILES never uses: an underscore,
# a slash, or a lattice-letter-then-digit-then-lowercase pattern with a hyphen.
_SG_SHAPE = re.compile(r"^[PABCIFR][a-zA-Z0-9_/\-]*[_/\-][a-zA-Z0-9_/]*$")


def _looks_like_spacegroup(s: str) -> bool:
    s = s.strip(".,;)")
    if "_" in s or "/" in s:
        return bool(re.match(r"^[PABCIFR]", s))
    # Fm-3m / Fd-3m: lattice letter, lowercase, hyphen, digit
    return bool(re.match(r"^[PABCIFR][a-z]?-?\d", s)) and "-" in s


# explicit tagged form:  SMILES: <str>  or  SMILES=<str>
_TAGGED = re.compile(r"SMILES\s*[:=]\s*([^\s,;]+)", re.IGNORECASE)
# bare candidates: a run of SMILES chars with structure-y tokens
_BARE = re.compile(rf"(?<![A-Za-z0-9])((?:{_SMILES_CHARS}){{3,}})(?![A-Za-z0-9])")
_STRUCTUREY = re.compile(r"[=#\[\]()@/\\]")  # tokens a formula never has
# A Hill-style molecular formula: capital-led element blocks each optionally
# followed by a count, e.g. H2O, C8H10N4O2, NH3. We do NOT want to treat these
# as SMILES candidates -- they belong to the formula channel.
_FORMULA_LIKE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def _looks_like_formula(s: str) -> bool:
    s = s.rstrip(".")
    if not _FORMULA_LIKE.match(s):
        return False
    # must contain at least one digit or >=2 element blocks to be a formula
    blocks = re.findall(r"[A-Z][a-z]?\d*", s)
    return bool(re.search(r"\d", s)) or len(blocks) >= 2


# Organic-subset atoms that can begin a SMILES (plus '[' for bracket atoms).
_SMILES_START = re.compile(r"^[\[BCNOPSFIbcnops]")
_STRUCTUREY_STRICT = re.compile(r"[=#\[\]()@]")  # a REAL bond/branch/bracket


def _strip_markdown(tok: str) -> str:
    """Remove code/LaTeX wrappers a model puts around a real SMILES:
    backticks, `$...$` math, and stray surrounding `*` or spaces. This recovers
    valid SMILES that were flagged only because the wrapper broke RDKit parsing.
    """
    return tok.strip().strip("`").strip("$").strip("*").strip()


def _plausible_bare_smiles(s: str) -> bool:
    """Precision-first gate for an UNTAGGED SMILES candidate.

    The pilot flag audit showed that bare extraction was 97% false-positive: it
    accepted IUPAC-name locant fragments ("4-diamino", "5-(4-chlorophenyl)"),
    space-group symbols, LaTeX, and split formula fragments as SMILES purely
    because they contained a digit (mistaken for a ring closure). A bare
    candidate is only plausibly a SMILES claim when ALL hold:

      1. it carries a REAL structural token (bond/branch/bracket), not merely a
         digit -- a lone digit is a locant far more often than a ring closure;
      2. parentheses AND square brackets are balanced (kills "[Al", "]Cl");
      3. it starts with a valid SMILES atom (organic-subset letter or "[")
         (kills digit-led locants "4-oxo" and bracket-close-led fragments);
      4. it contains none of the punctuation SMILES effectively never uses in
         generated text but names/space-groups/LaTeX do: hyphen, slash,
         underscore, backslash (kills "benzene-1", "UiO-66", "Fm-3m",
         "P\\(\\bar", "Pm\\overline").

    Ring-closure-only structures (e.g. bare "c1ccccc1") still pass because they
    carry aromatic lowercase atoms AND a digit that pairs; those are handled by
    the caller's aromatic-atom check plus balance, so we allow a digit to count
    as structural ONLY when the token has >=2 lowercase aromatic atoms.
    """
    if not _SMILES_START.match(s):
        return False
    if s.count("(") != s.count(")") or s.count("[") != s.count("]"):
        return False
    # Name/space-group/LaTeX punctuation kills a candidate -- but charges and
    # isotopes live INSIDE bracket atoms ([O-], [N+], [13C]), so remove bracket
    # contents before testing. What remains carrying -, /, _, or \ is a name
    # ("benzene-1", "UiO-66"), a space group ("P4_2/mnm"), or LaTeX ("Pm\over").
    core = re.sub(r"\[[^\]]*\]", "X", s)
    if any(ch in core for ch in "-/_\\"):
        return False
    # count real structural tokens (bonds/branches/brackets)
    n_struct = len(re.findall(r"[=#\[\]()@]", s))
    if n_struct == 0:
        # no bond/branch/bracket: accept only a ring-closure aromatic system
        # (>=2 lowercase aromatic atoms AND a ring-closure digit), e.g. c1ccccc1
        aromatic = len(re.findall(r"[bcnops]", s))
        return aromatic >= 2 and bool(re.search(r"[a-z]\d", s))
    # has structure: accept if RDKit parses it (a real SMILES claim, will PASS),
    # OR it is structurally complex enough (>=3 tokens) to be a genuine ATTEMPTED
    # SMILES the model got wrong -- this keeps true invalid-SMILES catches while
    # dropping short abbreviations like [BMIM] or O(BDC) that merely look bracket-y.
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        if Chem.MolFromSmiles(s) is not None:
            return True
    except Exception:
        pass
    return n_struct >= 3


def extract_smiles(text: str) -> list[Claim]:
    seen_spans: list[tuple[int, int]] = []
    out: list[Claim] = []
    for m in _TAGGED.finditer(text):
        s = _strip_markdown(m.group(1).rstrip(".,;"))
        # strip a trailing ")" that has no matching "(" — the tag regex can grab
        # the closing paren of an enclosing phrase, e.g. "(H2O, SMILES: O)" -> "O)".
        while s.endswith(")") and s.count(")") > s.count("("):
            s = s[:-1]
        while s.startswith("(") and s.count("(") > s.count(")"):
            s = s[1:]
        s = s.strip(".,;")
        if not s:
            continue
        out.append(Claim("smiles", s, (m.start(1), m.start(1) + len(s)), {"tagged": True}))
        seen_spans.append(m.span(1))
    for m in _BARE.finditer(text):
        s = m.group(1)
        start = m.start(1)
        # skip if already captured by a tagged match
        if any(a <= start < b for a, b in seen_spans):
            continue
        # Normalize surrounding SENTENCE punctuation without harming valid
        # SMILES boundary tokens. [ and ] are real atom-bracket chars, so we
        # never strip them; parens are stripped only when UNBALANCED (a stray
        # "(In" or trailing ")"), and .,; are always sentence noise.
        s = s.strip(".,;")
        # strip a leading "(" only if it has no matching ")"
        while s.startswith("(") and s.count("(") > s.count(")"):
            s = s[1:]
        # strip a trailing ")" only if it has no matching "("
        while s.endswith(")") and s.count(")") > s.count("("):
            s = s[:-1]
        s = _strip_markdown(s.strip(".,;"))
        # strip FULLY-ENCLOSING balanced parens: "(DMSO)"->"DMSO", "(SiO2)"->"SiO2"
        while (
            len(s) >= 2 and s[0] == "(" and s[-1] == ")" and s.count("(") == 1 and s.count(")") == 1
        ):
            s = s[1:-1]
        # recompute start robustly: locate the normalized token near the match
        idx = m.group(1).find(s)
        start = m.start(1) + (idx if idx >= 0 else 0)
        if not s:
            continue
        # an all-uppercase (or upper+digit) token is an ABBREVIATION or a
        # formula acronym (DMSO, DMF, THF, SiO2 written in caps), not SMILES.
        # Real bare SMILES use lowercase aromatic atoms or mixed case with bonds.
        if s.isupper() and len(s) >= 2 and not re.search(r"[=#\[\]()/\\]", s):
            continue
        # a molecular formula is not a SMILES candidate (checked AFTER stripping,
        # so "C6H7N.)" -> "C6H7N" is correctly routed away from the SMILES tier)
        if _looks_like_formula(s):
            continue
        # a Hermann-Mauguin space-group symbol (Fm-3m, P6_3mc) is not SMILES
        if _looks_like_spacegroup(s):
            continue
        # a molecular formula is not a SMILES candidate
        if _looks_like_formula(s):
            continue
        # a Hermann-Mauguin space-group symbol (Fm-3m, P6_3mc) is not SMILES
        if _looks_like_spacegroup(s):
            continue
        # PRECISION GATE (added after the pilot flag audit found bare extraction
        # was 97% false-positive). A bare candidate must look structurally like a
        # SMILES: real bond/branch/bracket token, balanced brackets, valid atom
        # start, and none of the name/space-group/LaTeX punctuation. This single
        # gate replaces the previous permissive "has a digit -> accept" rule that
        # let IUPAC locant fragments ("4-diamino", "5-(4-chlorophenyl)"), split
        # formula fragments ("[Al"), and LaTeX ("Pm\\overline") through.
        if not _plausible_bare_smiles(s):
            continue
        if not re.search(r"[CNOSPFBIclbrnos]", s):
            continue
        out.append(Claim("smiles", s, (start, start + len(s)), {"tagged": False}))
    return out


# ---- name = formula -------------------------------------------------------
_FORMULA = r"[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+"
# name is 1-3 alphabetic words (e.g. "acetic acid", "carbon dioxide"),
# immediately followed by a connector and a formula token.
_NAME_FORMULA = re.compile(
    rf"\b([A-Za-z]+(?:\s[a-z]+){{0,2}})\s*,?\s*"
    rf"(?:has (?:the )?formula|formula|is|=)\s+"
    rf"\(?({_FORMULA})\)?",
)
_STOPWORDS = {
    "is",
    "the",
    "a",
    "an",
    "which",
    "and",
    "of",
    "with",
    "it",
    "consider",
    "adding",
    "so",
    "another",
    "candidate",
    "gives",
    "starting",
    "scaffold",
    "good",
    "has",
    "formula",
    "finally",
    "first",
    "second",
    "third",
    "next",
    "then",
}


# Labeled-field form: "Molecular formula: C8H9NO2" / "Formula: Al2O3". Common in
# structured model answers where the compound name is on a preceding line/header,
# so we bind to the nearest KNOWN compound name (same strategy as properties).
_LABELED_FORMULA = re.compile(
    r"(?:molecular\s+formula|chemical\s+formula|formula)\s*[:=]\s*\(?(" + _FORMULA + r")\)?",
    re.IGNORECASE,
)


def extract_formulas(text: str, known_names: list[str] | None = None) -> list[Claim]:
    known_names = known_names or []
    out: list[Claim] = []
    seen: list[tuple[int, int]] = []
    for m in _NAME_FORMULA.finditer(text):
        name = m.group(1).strip().lower()
        # if the "name" is actually a structure-string label ("SMILES of naproxen",
        # "InChI of X"), the captured token is a SMILES/InChI, not a formula. Skip.
        if "smiles" in name or "inchi" in name:
            continue
        # drop leading AND trailing stopwords so "water has" -> "water",
        # "good starting scaffold is benzene" -> "benzene"
        words = [w for w in name.split()]
        while words and words[0] in _STOPWORDS:
            words.pop(0)
        while words and words[-1] in _STOPWORDS:
            words.pop()
        name = " ".join(words)
        if not name:
            continue
        formula = m.group(2)
        if not _plausible_formula(formula):
            continue
        if _preceded_by_smiles(text, m.start(2)):
            continue
        out.append(Claim("formula", m.group(0), m.span(), {"name": name, "formula": formula}))
        seen.append(m.span())
    # labeled-field form: bind formula to nearest known compound name before it
    for m in _LABELED_FORMULA.finditer(text):
        if any(a <= m.start() < b for a, b in seen):
            continue
        formula = m.group(1)
        if not _plausible_formula(formula):
            continue
        if _preceded_by_smiles(text, m.start(1)):
            continue
        # nearest known name whose whole-word occurrence ends at/before the label
        bound, best = None, -1
        window = text[: m.start()].lower()
        for kn in known_names:
            for mm in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(kn) + r"(?![A-Za-z0-9])", window):
                if mm.end() > best:
                    best, bound = mm.end(), kn
        if bound is None:
            continue
        out.append(Claim("formula", m.group(0), m.span(), {"name": bound, "formula": formula}))
    # header form: "Dopamine:  C8H11NO2" / "Warfarin - C19H16O4" -- a known name
    # immediately followed by a formula, no "formula" keyword. Only fires for
    # KNOWN names (so arbitrary "Word: C6..." prose isn't grabbed) and only if the
    # span isn't already covered.
    covered = [s for s in seen] + [c.span for c in out]
    names_with_formula = {c.payload.get("name") for c in out}
    for kn in known_names:
        # if a labeled/name formula for this compound already exists, the header
        # form would only add noise (e.g. a fragment from a self-correcting
        # "No -- corrected: ..." retraction). Skip it.
        if kn in names_with_formula:
            continue
        for hm in re.finditer(
            r"(?<![A-Za-z0-9])"
            + re.escape(kn)
            + r"(?![A-Za-z0-9])\s*[:\-–]\s*\(?("
            + _FORMULA
            + r")\)?",
            text,
            re.IGNORECASE,
        ):
            if any(a <= hm.start() < b for a, b in covered):
                continue
            formula = hm.group(1)
            if not _plausible_formula(formula):
                continue
            if _preceded_by_smiles(text, hm.start(1)):
                continue
            out.append(Claim("formula", hm.group(0), hm.span(), {"name": kn, "formula": formula}))
            covered.append(hm.span())
    return out


# ---- name property = value ------------------------------------------------
_PROP_WORDS = {
    "dipole": ("dipole_moment", "debye"),
    "dipole moment": ("dipole_moment", "debye"),
    "homo-lumo gap": ("homo_lumo_gap", "ev"),
    "homo lumo gap": ("homo_lumo_gap", "ev"),
    "gap": ("homo_lumo_gap", "ev"),
}
_PROP = re.compile(
    r"([A-Za-z][A-Za-z0-9\-\s]{1,25}?)\s+(?:has (?:a )?|,\s*)?"
    r"(dipole moment|dipole|homo[- ]lumo gap|gap)\s*"
    r"(?:of\s+[A-Za-z0-9\-\(\), ]{0,30}?\s+is|of|=|is|:)?\s*"
    r"(?:approximately|about|~|approx\.?)?\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*(D|debye|eV)?",
    re.IGNORECASE,
)
# labeled-field form: "Dipole moment: 1.04 D", "Dipole moment = 3.92 D",
# "Dipole moment approximately 1.60 D". Value may be prefixed by "approximately".
# (the "≈" approx sign is already stripped to a space in normalize_text.)
_LABELED_PROP = re.compile(
    r"(dipole moment|dipole|homo[- ]lumo gap|band gap|gap)\s*"
    r"(?:of|=|is|:|,)?\s*(?:approximately|about|~|approx\.?)?\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*(D|debye|eV)\b",
    re.IGNORECASE,
)


def extract_properties(text: str, known_names: list[str] | None = None) -> list[Claim]:
    """Extract `name property = value` claims.

    The grammatical subject of a property is often separated from the property
    phrase ("water ... with a dipole moment of 1.85 D"). A greedy regex grabs
    filler ("with a") as the name. To bind correctly we resolve the name to the
    nearest KNOWN compound name occurring at or before the match; regex capture
    is only a fallback.
    """
    known_names = known_names or []
    out: list[Claim] = []
    seen: list[tuple[int, int]] = []
    for m in _PROP.finditer(text):
        raw_name = m.group(1).strip().lower()
        # Bind to the closest known compound name whose WHOLE-WORD occurrence ends
        # nearest to (at/before) this property phrase (see _nearest_known_name).
        bound = _nearest_known_name(text, m.start(2), known_names)
        name = bound if bound is not None else raw_name
        prop_key, unit = _PROP_WORDS.get(m.group(2).lower().strip(), (m.group(2).lower(), None))
        val = float(m.group(3))
        out.append(
            Claim(
                "property",
                m.group(0),
                m.span(),
                {"name": name, "prop": prop_key, "value": val, "unit": (m.group(4) or unit)},
            )
        )
        seen.append(m.span())
    # labeled-field form: "Dipole moment: 1.04 D" / "Dipole moment = 3.92 D" with
    # the compound name on a preceding header line. Bind to nearest known name.
    for m in _LABELED_PROP.finditer(text):
        if any(a <= m.start() < b for a, b in seen):
            continue
        prop_word = m.group(1).lower().strip()
        prop_key, unit = _PROP_WORDS.get(prop_word, ("dipole_moment", "debye"))
        bound = _nearest_known_name(text, m.start(), known_names)
        if bound is None:
            continue
        out.append(
            Claim(
                "property",
                m.group(0),
                m.span(),
                {
                    "name": bound,
                    "prop": prop_key,
                    "value": float(m.group(2)),
                    "unit": (m.group(3) or unit),
                },
            )
        )
    return out


# ---- crystalline claims ---------------------------------------------------
# Hermann-Mauguin symbols start with a Bravais-lattice letter and are short.
_HM = r"[PABCIFR][a-zA-Z0-9_/\-]{1,9}"
# A space-group + number claim needs an explicit cue so molecular formulas
# (e.g. "C13H18O2") can't masquerade as "symbol C13H18O, number 2":
#   (a) the literal words "space group" before the symbol, OR
#   (b) the number wrapped as (#225), (225), or #225.
# A HM symbol followed by an explicit number cue. The number must be introduced
# by a cue -- "(", "#", "No.", or "number" -- so the digit INSIDE the symbol
# (the "3" of "Fm-3m") is never mistaken for the space-group number. The symbol
# is the _HM run; the number is a cued 1-3 digit token after it.
# Cues are case-sensitive on purpose: "No." / "No" / "number" with a capital or
# lowercase n, but NOT the "NO" inside a formula like "C8H9NO2". We list the exact
# acceptable forms rather than use IGNORECASE (which let "NO2" match the "No" cue
# and "formula" match the [PABCIFR] lattice-letter class).
# The number MUST be introduced by a cue, so a formula digit ("C8H9NO2") is never
# read as a space-group number. Accepted cues, each optionally wrapped in "(":
#   "(225)"  "( 225 )"  "#225"  "(No. 167)"  ", No 225"  ", number 225"
# A lone "(" counts as a cue (the parenthesized-number convention); "No"/"number"
# require following whitespace so the "NO" in a formula cannot match.
_NUMCUE = (
    r"(?:"
    r"\(\s*#?\s*(?:No\.?\s*|number\s+)?"  # "(225", "(No. 167", "(#225"
    r"|"
    r"#\s*"  # "#225"
    r"|"
    r",?\s*(?:No\.?|number)\s+"  # ", No 225", "number 225"
    r")"
)
_SG_NUM = re.compile(
    r"(?:[Ss]pace group\s+)?(" + _HM + r")\s*" + _NUMCUE + r"(\d{1,3})\s*\)?",
)
# labeled-field form (very common in structured answers):
#   "Space group symbol: Fm-3m    Space group number: 225"
#   "space group symbol Fd-3m and space group number 227"
# symbol and number carry their OWN labels and may be separated by markup/newlines
# (normalized to spaces). Require the "space group ... symbol" and "... number"
# labels so a bare formula can't match. Symbol is an _HM run; number is 1-3 digits.
_SG_LABELED = re.compile(
    r"space group\s*(?:symbol)?\s*(?:is|[:=])?\s*(" + _HM + r")\b"
    r"[\s\S]{0,60}?"
    r"space group\s*(?:number|no\.?|#)\s*(?:is|[:=])?\s*(\d{1,3})\b",
    re.IGNORECASE,
)
# crystal-system claim:  "cubic Fm-3m" / "P6_3mc is hexagonal" / "space group 225 is cubic"
_SYS_WORDS = r"(triclinic|monoclinic|orthorhombic|tetragonal|trigonal|rhombohedral|hexagonal|cubic)"
# symbol side must be a HM symbol WITH a rotation/inversion token, or a bare
# space-group NUMBER -- never a plain word or formula.
_SGREF = r"(?:[PABCIFR][a-zA-Z]*[0-9_/\-][a-zA-Z0-9_/\-]*|\d{1,3})"
_SG_SYS_A = re.compile(_SYS_WORDS + r"\s+(?:space group\s+)?(" + _SGREF + r")")
_SG_SYS_B = re.compile(
    r"(?:space group\s+)(" + _SGREF + r")\s+(?:is|,)\s+" + _SYS_WORDS, re.IGNORECASE
)
# lattice parameters + system:  "cubic ... a = 5.64 Å" (angles optional -> assume 90)
_LATT = re.compile(
    r"(triclinic|monoclinic|orthorhombic|tetragonal|trigonal|rhombohedral|hexagonal|cubic)"
    r"[^.]*?a\s*=\s*([\d.]+)\s*(?:[,\s]+b\s*=\s*([\d.]+))?\s*(?:[,\s]+c\s*=\s*([\d.]+))?"
    r"(?:[^.]*?alpha\s*=\s*([\d.]+))?(?:[^.]*?beta\s*=\s*([\d.]+))?(?:[^.]*?gamma\s*=\s*([\d.]+))?",
    re.IGNORECASE,
)
_SG_TOKEN = re.compile(r"^[A-Z][a-zA-Z0-9_/\-]*$")


def _valid_sg_symbol(sym: str) -> bool:
    """A real HM symbol: passes the SG-token shape AND carries a rotation/
    inversion/glide token (so a bare formula fragment is rejected)."""
    return bool(_SG_TOKEN.match(sym)) and bool(re.search(r"[\-/0-9_]", sym))


def extract_crystal(text: str) -> list[Claim]:
    out: list[Claim] = []
    claimed_spans: list[tuple[int, int]] = []
    # labeled-field form first (symbol and number carry their own labels)
    for m in _SG_LABELED.finditer(text):
        sym, num = m.group(1), int(m.group(2))
        if not (1 <= num <= 230) or not _valid_sg_symbol(sym):
            continue
        out.append(Claim("sg_number", m.group(0), m.span(), {"symbol": sym, "number": num}))
        claimed_spans.append(m.span())
    for m in _SG_NUM.finditer(text):
        # group 1 = HM symbol, group 2 = cued space-group number
        sym = m.group(1)
        num_s = m.group(2)
        if sym is None or num_s is None:
            continue
        num = int(num_s)
        if not (1 <= num <= 230):
            continue
        if not _valid_sg_symbol(sym):
            continue
        # skip if already covered by a labeled-field match
        if any(a <= m.start() < b for a, b in claimed_spans):
            continue
        out.append(Claim("sg_number", m.group(0), m.span(), {"symbol": sym, "number": num}))
    for rgx, order in ((_SG_SYS_A, "sys_first"), (_SG_SYS_B, "sg_first")):
        for m in rgx.finditer(text):
            if order == "sys_first":
                system, sg = m.group(1), m.group(2)
            else:
                sg, system = m.group(1), m.group(2)
            out.append(
                Claim("sg_system", m.group(0), m.span(), {"sg": sg, "system": system.lower()})
            )

    def _f(g: str | None, default: float | None = None) -> float | None:
        if not g:
            return default
        return float(g.rstrip("."))

    for m in _LATT.finditer(text):
        system = m.group(1).lower()
        a = _f(m.group(2))
        b = _f(m.group(3), a)
        c = _f(m.group(4), a)
        alpha = _f(m.group(5), 90.0)
        beta = _f(m.group(6), 90.0)
        # gamma defaults to 120 for hex/trig, else 90
        gamma = _f(m.group(7), 120.0 if system in ("hexagonal", "trigonal") else 90.0)
        out.append(
            Claim(
                "lattice",
                m.group(0)[:60],
                m.span(),
                {
                    "system": system,
                    "a": a,
                    "b": b,
                    "c": c,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                },
            )
        )
    return out


def extract_all(text: str, known_names: list[str] | None = None) -> list[Claim]:
    # normalize Unicode subscripts + markdown BEFORE extraction so formula /
    # space-group / property claims written as "C₈H₉NO₂" or
    # "**R-3c, No. 167**" are visible to the ASCII regexes. Length-preserving,
    # so spans still index into a same-length string.
    text = normalize_text(text)
    claims = (
        extract_smiles(text)
        + extract_formulas(text, known_names=known_names)
        + extract_properties(text, known_names=known_names)
        + extract_crystal(text)
    )
    claims.sort(key=lambda c: c.span[0])
    return claims


def harvest_names(text: str) -> list[str]:
    """Collect candidate compound NAMES a trace refers to, so a live provider
    (PubChem MCP) can pre-resolve them before verification.

    Sources: the subject of every `name = formula` and `name property = value`
    claim. Deliberately over-collects; unresolvable names simply return None at
    lookup time.
    """
    names = set()
    for c in extract_formulas(text):
        names.add(c.payload["name"])
    # property names need a known_names pass to bind well, but the raw regex
    # subject is still a useful candidate here
    for c in extract_properties(text):
        n = c.payload["name"]
        if n and len(n) > 2:
            names.add(n)
    return sorted(names)
