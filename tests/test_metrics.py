"""The headline numbers, pinned."""
import numpy as np
import pandas as pd
import pytest

from offtarget.metrics import auc_table, bootstrap_auc_ci, stratified_auc


@pytest.fixture(scope="module")
def audited(processed):
    df = pd.read_parquet(processed / "pairs_audited.parquet")
    return df[df.label.notna()]


def test_the_dataset_is_what_the_readme_says(audited):
    assert len(audited) == 708
    assert int((audited.label == "assay_active").sum()) == 147
    assert int((audited.label == "assay_inactive").sum()) == 561


@pytest.mark.parametrize("score,pooled,within", [
    ("confidence",           0.563, 0.527),
    ("iptm",                 0.592, 0.486),
    ("affinity_probability", 0.731, 0.683),
    ("neg_affinity_score",   0.771, 0.710),
])
def test_auc_table(audited, score, pooled, within):
    row = auc_table(audited).set_index("score")
    from offtarget.metrics import SCORES
    r = row.loc[SCORES[score]]
    assert r.pooled_auc == pytest.approx(pooled, abs=0.001)
    assert r.within_target_auc == pytest.approx(within, abs=0.001)
    assert r.targets_used == 34


def test_the_finding(audited):
    """Structural confidence at chance, affinity not — stated as an assertion."""
    conf = stratified_auc(audited, "confidence")[0]
    iptm = stratified_auc(audited, "iptm")[0]
    aff = stratified_auc(audited, "neg_affinity_score")[0]

    for score in ["confidence", "iptm"]:
        lo, hi = bootstrap_auc_ci(audited, score)
        assert lo < 0.5 < hi, f"{score} interval no longer includes chance"

    lo, hi = bootstrap_auc_ci(audited, "neg_affinity_score")
    assert lo > 0.5
    assert aff > max(conf, iptm) + 0.15


def test_stratification_matters(audited):
    """Pooled AUC flatters the confidence scores; that gap is why we stratify."""
    t = auc_table(audited)
    conf = t[t.score.str.startswith("Confidence")].iloc[0]
    assert conf.pooled_auc - conf.within_target_auc > 0.03


def test_only_two_class_groups_contribute():
    df = pd.DataFrame({
        "Target": ["A"] * 4 + ["B"] * 3,
        "label": ["assay_active", "assay_inactive"] * 2 + ["assay_active"] * 3,
        "s": [1.0, 0.0, 0.9, 0.1, 0.5, 0.4, 0.3],
    })
    auc, used = stratified_auc(df, "s")
    assert used == 1                  # B contributes only actives
    assert auc == pytest.approx(1.0)


def test_bootstrap_resamples_targets_not_rows(audited):
    """A row-level bootstrap would give a far tighter interval than the data supports."""
    lo, hi = bootstrap_auc_ci(audited, "iptm", n=2000, seed=1)
    assert hi - lo > 0.05
    assert bootstrap_auc_ci(audited, "iptm", n=2000, seed=1) == (lo, hi)   # deterministic
