"""Audit of what was actually handed to Boltz-2.

A Boltz-2 run is only as meaningful as the sequence it was given. The inputs for
this project were written by ten near-identical generator scripts, each carrying a
hard-coded sequence pasted from a web page. This module reads the generated inputs
back and reports, in counts, what those sequences turned out to contain:
expression-construct scaffold that is not part of the human protein, point
mutations that make a crystallography construct a different molecule, and
oligomeric states assigned per target with no stated rule.

None of this is recoverable from the scored outputs, which is the point: the
defects live in the inputs and are invisible downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Purification and expression scaffold. These are cloning artefacts: they are
# present in the construct that was crystallised, absent from the protein in a
# cell, and they lengthen the chain the model folds.
TAG_MOTIFS: dict[str, str] = {
    "His-tag":   r"HHHHHH",       # His6 and longer runs (His8, His12)
    "Strep-tag": r"WSHPQFEK",     # Strep-tag II; "twin-Strep" is two copies
    "thrombin":  r"LVPRGS",       # thrombin cleavage site
    "TEV":       r"ENLYFQ",       # TEV protease site
    "FLAG":      r"DYKDDDD",      # DYKDDDDK, and the K->A variant used in 8HN1
    "GS-linker": r"GGGSGGGS",     # flexible linker joining tags
}

AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class TagHit:
    name: str
    start: int
    end: int
    motif: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class ArmDiff:
    """How an input sequence differs from the canonical UniProt sequence.

    The distinction that matters is truncation versus substitution. A truncated
    construct is still the same protein with its ends removed; a construct
    carrying thermostabilising point mutations and mutated glycosylation sequons
    is a *different molecule*, and the two cannot be compared head to head as
    though the only difference were length.

    Computed from a proper alignment (``difflib`` opcodes) rather than by walking
    positions from a single anchor: an internal deletion shifts every downstream
    residue, and a position-walking diff reports that shift as hundreds of
    substitutions that are not there.
    """
    input_length: int
    canonical_length: int
    matched: int                                        # residues aligning exactly
    substitutions: list[tuple[int, str, str]]           # (canonical position, canonical, input)
    insertions: list[tuple[int, str]]                   # (canonical position, inserted segment)
    deletions: list[tuple[int, int]]                    # 1-based inclusive canonical spans absent from the input
    tags: list[TagHit] = field(default_factory=list)

    @property
    def identity(self) -> float:
        """Fraction of the input that aligns exactly to the canonical sequence."""
        return self.matched / self.input_length if self.input_length else 0.0

    @property
    def inserted_residues(self) -> int:
        """Residues present in the input but absent from the human protein."""
        return sum(len(s) for _, s in self.insertions)

    @property
    def deleted_residues(self) -> int:
        # Spans are 1-based and inclusive on both ends, so a span (41, 60) is 20
        # residues, not 19.
        return sum(e - s + 1 for s, e in self.deletions)

    @property
    def tag_residues(self) -> int:
        return sum(t.length for t in self.tags)

    @property
    def is_canonical_fragment(self) -> bool:
        """True when the input is the canonical protein, only shorter."""
        return not self.substitutions and not self.insertions

    @property
    def verdict(self) -> str:
        if self.tags and self.substitutions:
            return "engineered construct: purification scaffold and point substitutions"
        if self.tags:
            return f"canonical sequence carrying {self.inserted_residues} residues of purification scaffold"
        if self.substitutions:
            return f"construct with {len(self.substitutions)} point substitutions vs canonical"
        if self.deletions:
            return "clean truncation of the canonical sequence"
        return "canonical sequence"


def find_construct_tags(seq: str) -> list[TagHit]:
    """Locate every known expression-construct motif in a sequence.

    Overlapping His runs are merged so that His12 is reported once, not seven
    times, and the residue count reflects the real amount of scaffold.
    """
    hits: list[TagHit] = []
    for name, motif in TAG_MOTIFS.items():
        pattern = motif + "+" if name == "His-tag" else motif
        for m in re.finditer(pattern, seq):
            hits.append(TagHit(name, m.start(), m.end(), motif))
    return sorted(hits, key=lambda h: h.start)


def strip_construct_tags(seq: str, canonical: str) -> str:
    """Return the part of `seq` that is genuinely present in `canonical`.

    Used by the generator when an arm is declared `strip_tags: true`. Deliberately
    conservative: it finds the longest run of the input that appears verbatim in
    the canonical sequence rather than trying to excise motifs one at a time,
    because excision leaves linker debris behind.
    """
    best = ""
    for start in range(len(seq)):
        if len(seq) - start <= len(best):
            break
        lo, hi = len(best), len(seq) - start
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if seq[start:start + mid] in canonical:
                lo = mid
            else:
                hi = mid - 1
        if lo > len(best):
            best = seq[start:start + lo]
    return best


def diff_arms(input_seq: str, canonical_seq: str) -> ArmDiff:
    """Compare a model input against the canonical UniProt sequence for that gene.

    Alignment is by ``difflib.SequenceMatcher``, which is exact for sequences that
    are mostly identical — the case here, since every input started life as a
    copy-paste of the real protein. Equal-length replaced blocks are reported as
    point substitutions; longer input-only blocks are insertions, which is what
    purification scaffold looks like.
    """
    from difflib import SequenceMatcher

    matched = 0
    subs: list[tuple[int, str, str]] = []
    ins: list[tuple[int, str]] = []
    dels: list[tuple[int, int]] = []

    sm = SequenceMatcher(None, canonical_seq, input_seq, autojunk=False)
    for op, c0, c1, i0, i1 in sm.get_opcodes():
        if op == "equal":
            matched += c1 - c0
        elif op == "replace":
            if c1 - c0 == i1 - i0:
                for k in range(c1 - c0):
                    subs.append((c0 + k + 1, canonical_seq[c0 + k], input_seq[i0 + k]))
            else:                       # a substituted block of unequal length
                subs.append((c0 + 1, canonical_seq[c0:c1], input_seq[i0:i1]))
        elif op == "insert":
            ins.append((c0 + 1, input_seq[i0:i1]))
        elif op == "delete":
            dels.append((c0 + 1, c1))

    return ArmDiff(
        input_length=len(input_seq),
        canonical_length=len(canonical_seq),
        matched=matched,
        substitutions=subs,
        insertions=ins,
        deletions=dels,
        tags=find_construct_tags(input_seq),
    )


def read_input_yaml(path: Path) -> dict:
    """Parse a Boltz-2 input without a YAML dependency.

    The files are machine-written from one of three fixed templates, so a regex
    read is exact here and avoids depending on a parser to audit the parser's
    input.
    """
    text = Path(path).read_text(errors="ignore")
    seqs = re.findall(r'sequence:\s*"([A-Z]+)"', text)
    smiles = re.findall(r'smiles:\s*"([^"]*)"', text)
    ids = [i.strip('"') for i in re.findall(r"- protein:\n      id: (.+)\n", text)]
    return {
        "path": Path(path),
        "name": Path(path).name,
        "directory": Path(path).parent.name,
        "sequences": seqs,
        "chain_ids": ids,
        "n_chains": len(seqs),
        "smiles": smiles[0] if smiles else None,
        "oligomeric_state": "homodimer" if len(seqs) == 2 else "monomer",
    }


def audit_inputs(root: Path) -> "pd.DataFrame":  # noqa: F821
    """Audit every ``*.yaml`` under `root`. One row per input file."""
    import pandas as pd

    rows = []
    for path in sorted(Path(root).rglob("*.yaml")):
        rec = read_input_yaml(path)
        seq = rec["sequences"][0] if rec["sequences"] else ""
        tags = find_construct_tags(seq)
        rows.append({
            "file": rec["name"],
            "directory": rec["directory"],
            "n_chains": rec["n_chains"],
            "oligomeric_state": rec["oligomeric_state"],
            "seq_length": len(seq),
            "tags": ",".join(sorted({t.name for t in tags})),
            "tag_residues": sum(t.length for t in tags),
            "has_tags": bool(tags),
        })
    return pd.DataFrame(rows)
