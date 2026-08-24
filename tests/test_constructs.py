"""The input audit: every known motif detected, every count stable."""
import pytest
import yaml

from offtarget.constructs import (TAG_MOTIFS, audit_inputs, diff_arms,
                                  find_construct_tags, strip_construct_tags)


@pytest.mark.parametrize("name,motif", sorted(TAG_MOTIFS.items()))
def test_every_known_motif_is_detected(name, motif):
    seq = "MKTAYIAKQRQISFVKSHFSRQ" + motif.replace("+", "") + "GGSTLEENLYFQ"
    assert name in {h.name for h in find_construct_tags(seq)}


def test_his_runs_merge_into_one_hit():
    """His12 is one tag of twelve residues, not seven overlapping His6 hits."""
    hits = [h for h in find_construct_tags("MKTAYIAK" + "H" * 12) if h.name == "His-tag"]
    assert len(hits) == 1
    assert hits[0].length == 12


def test_untagged_sequence_is_clean():
    assert find_construct_tags("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ") == []


def test_audit_counts_are_stable(repo):
    """The headline numbers from the input audit."""
    df = audit_inputs(repo / "inputs/generated")
    assert len(df) == 158
    assert int(df.has_tags.sum()) == 29
    assert int((df.n_chains == 2).sum()) == 102
    assert int((df.n_chains == 1).sum()) == 56


def test_slc6a4_carries_55_residues_of_scaffold(repo):
    """The construct behind every SLC6A4 claim in the March 2026 deck."""
    import gzip, re
    cfg = yaml.safe_load((repo / "inputs/targets.yaml").read_text())
    seq = cfg["targets"]["SLC6A4"]["arms"]["pdb"]["sequence"]

    canonical, header, buf = None, None, []
    for line in gzip.open(repo / "data/raw/uniprot_reviewed_2025-11-10.fasta.gz", "rt"):
        line = line.strip()
        if line.startswith(">"):
            if header and "P31645" in header:
                canonical = "".join(buf)
            header, buf = line, []
        else:
            buf.append(line)
    if header and "P31645" in header:
        canonical = "".join(buf)

    d = diff_arms(seq, canonical)
    assert d.substitutions == []                  # the human sequence, unmodified
    assert d.inserted_residues == 55              # followed by 55 residues of scaffold
    assert {"His-tag", "Strep-tag", "thrombin", "GS-linker"} <= {t.name for t in d.tags}


def test_diff_arms_reports_truncation_not_substitutions():
    """An internal deletion must not be reported as hundreds of point substitutions.

    A position-walking diff reports the shift caused by a deletion as a substitution
    at every downstream residue. The sequence here is non-repeating so the alignment
    is unambiguous.
    """
    canonical = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL"
                 "PDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMG")
    d = diff_arms(canonical[:40] + canonical[60:], canonical)
    assert d.substitutions == []
    assert d.deleted_residues == 20
    assert d.identity == 1.0
    assert d.is_canonical_fragment


def test_strip_construct_tags_recovers_the_protein():
    canonical = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    assert strip_construct_tags(canonical + "GGGSGGGS" + "H" * 8, canonical) == canonical
