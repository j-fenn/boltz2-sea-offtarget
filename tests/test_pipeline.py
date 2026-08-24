"""Row counts, label hygiene and the named exclusions."""
import json

import pandas as pd
import pytest

from offtarget.pipeline import KNOWN_BAD_INPUTS, LABEL_MAP, normalise_label

EXPECTED_ROWS = {
    "safety_targets": 73,      # S2
    "uniprot_map": 117,        # reviewed FASTA
    "drugs": 656,              # S1
    "predictions": 4759,       # Predictions.dat, before de-duplication
    "confirmed": 174,          # S3
    "pairs_all": 710,
    "pairs_labelled": 708,
    "pairs_primary": 690,      # ADRA2A removed
    "dimer_panel": 107,        # Raw + MAOA + Transporters
}


@pytest.mark.parametrize("stage,n", sorted(EXPECTED_ROWS.items()))
def test_stage_row_counts(processed, stage, n):
    manifest = json.loads((processed / "manifest.json").read_text())
    assert manifest["stages"][stage] == n


def test_every_input_is_hashed(processed):
    manifest = json.loads((processed / "manifest.json").read_text())
    assert manifest["inputs"]
    assert all(len(h) == 64 for h in manifest["inputs"].values())


def test_labels_are_renamed_everywhere(processed):
    """`True positive` must not survive into anything an analysis reads."""
    for name in ["pairs", "pairs_audited", "dimer_panel"]:
        df = pd.read_parquet(processed / f"{name}.parquet")
        assert set(df.label.dropna().unique()) <= {"assay_active", "assay_inactive"}


def test_label_normalisation_is_case_and_whitespace_insensitive():
    """`'inactive '` and `'Inactive'` are the same label; the original merge disagreed."""
    for variant in ["inactive", "Inactive", "inactive ", " INACTIVE"]:
        assert normalise_label(variant) == "assay_inactive"
    for variant in ["True positive", "true positive", "TRUE POSITIVE "]:
        assert normalise_label(variant) == "assay_active"
    assert normalise_label("CC(=O)NC1=CC=C(C=C1)O") is None
    assert normalise_label(None) is None


def test_known_bad_input_is_excluded_with_a_reason(processed):
    audited = pd.read_parquet(processed / "pairs_audited.parquet")
    primary = pd.read_parquet(processed / "pairs.parquet")
    assert "ADRA2A" in set(audited.Target)          # kept in the record
    assert "ADRA2A" not in set(primary.Target)      # excluded from the analysis
    assert "32-residue fragment" in KNOWN_BAD_INPUTS["ADRA2A"]   # the reason is written down


def test_dimer_labels_were_all_recovered(processed):
    """15 rows had a SMILES in the label column; 39 more had no label column at all."""
    df = pd.read_parquet(processed / "dimer_panel.parquet")
    assert df.label.isna().sum() == 0
    assert set(df.sheet.unique()) == {"Raw", "MAOA", "Transporters"}


def test_smiles_comparison_is_stable(processed):
    """The cost of re-deriving structures that were already published in S1."""
    s = pd.read_parquet(processed / "smiles_comparison.parquet")
    counts = s.status.value_counts().to_dict()
    assert len(s) == 220
    assert counts["identical"] == 216
    assert counts["different_compound"] == 3
    assert counts["tautomer_or_stereo"] == 1
    assert (s.name_match == "case_insensitive").sum() == 14
