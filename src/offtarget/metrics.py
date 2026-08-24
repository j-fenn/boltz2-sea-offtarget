"""Discrimination and calibration measures for scored target-drug pairs.

The central function is :func:`stratified_auc`. Pooled AUC over a table like this
one is close to meaningless: targets differ enormously in how many of their tested
drugs turned out to be active, so a score that has learned nothing except *which
target this is* still ranks well pooled. Stratifying within target removes that
channel. On this dataset it is the difference between a confidence score looking
mediocre (0.563) and a confidence score being at chance (0.527).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Boltz-2's confidence is 0.8*pLDDT + 0.2*ipTM, so it is dominated by fold quality.
# ipTM is the interface term on its own, and is the metric a structural biologist
# reaches for when asked whether a complex is real -- which is why it is here.
SCORES = {
    "confidence":           "Confidence (0.8*pLDDT + 0.2*ipTM)",
    "iptm":                 "ipTM (interface only)",
    "affinity_probability": "Affinity probability",
    "neg_affinity_score":   "-Affinity score (predicted log affinity)",
}


def stratified_auc(df: pd.DataFrame, score: str, label: str = "label",
                   group: str = "Target", positive: str = "assay_active"
                   ) -> tuple[float, int]:
    """AUC computed within each group, combined weighted by that group's pos*neg pairs.

    Weighting by ``n_pos * n_neg`` makes the result the fraction of all
    *within-target* active/inactive pairs that the score orders correctly, which is
    the quantity the pooled AUC is usually mistaken for. Groups contributing only
    one class carry no information about ranking and are dropped; on this dataset
    34 of 63 targets survive that filter.

    Returns ``(auc, n_groups_used)``.
    """
    y = (df[label] == positive).astype(int)
    num = den = 0.0
    used = 0
    for _, idx in df.groupby(group).groups.items():
        yy, ss = y.loc[idx], df.loc[idx, score]
        if yy.nunique() < 2:
            continue
        w = float(yy.sum() * (1 - yy).sum())
        num += roc_auc_score(yy, ss) * w
        den += w
        used += 1
    return (num / den if den else float("nan")), used


def auc_table(df: pd.DataFrame, scores: dict[str, str] = None,
              label: str = "label", group: str = "Target") -> pd.DataFrame:
    """Pooled and within-target AUC for every score, side by side."""
    scores = scores or SCORES
    y = (df[label] == "assay_active").astype(int)
    rows = []
    for col, name in scores.items():
        strat, used = stratified_auc(df, col, label=label, group=group)
        rows.append({"score": name,
                     "pooled_auc": roc_auc_score(y, df[col]),
                     "within_target_auc": strat,
                     "targets_used": used})
    return pd.DataFrame(rows)


def per_target_auc(df: pd.DataFrame, score: str, label: str = "label",
                   group: str = "Target", positive: str = "assay_active") -> pd.DataFrame:
    """One row per target that contributes both classes: its AUC and its weight."""
    y = (df[label] == positive).astype(int)
    rows = []
    for g, idx in df.groupby(group).groups.items():
        yy, ss = y.loc[idx], df.loc[idx, score]
        if yy.nunique() < 2:
            continue
        rows.append({group: g, "n": len(idx), "n_pos": int(yy.sum()),
                     "auc": roc_auc_score(yy, ss),
                     "weight": float(yy.sum() * (1 - yy).sum())})
    return pd.DataFrame(rows)


def bootstrap_auc_ci(df: pd.DataFrame, score: str, label: str = "label",
                     group: str = "Target", n: int = 4000, seed: int = 0
                     ) -> tuple[float, float]:
    """Percentile bootstrap CI for the within-target AUC, resampling whole targets.

    Targets, not rows, are the unit of resampling: pairs within a target share a
    protein, a construct and an oligomeric state, so resampling rows would treat
    34 targets' worth of information as 708 independent observations and return an
    interval far tighter than the data supports.

    Because the estimator is a weighted mean of per-target AUCs, the bootstrap can
    resample the (auc, weight) pairs directly instead of recomputing ROC curves.
    """
    per = per_target_auc(df, score, label=label, group=group)
    if per.empty:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n, len(per)))
    a, w = per.auc.to_numpy(), per.weight.to_numpy()
    vals = (a[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def sequence_length_regression(df: pd.DataFrame, score: str = "confidence",
                               length: str = "seq_length") -> dict:
    """Regress a score on modelled sequence length.

    The stated hypothesis for the sub-chance UniProt confidence result was that
    Boltz-2 degrades on long chains. This returns the slope so the hypothesis has a
    number attached. A slope is not a mechanism: length is confounded with target
    identity, target class and construct choice, none of which are controlled here.
    """
    d = df.dropna(subset=[score, length])
    x, y = d[length].astype(float), d[score].astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"n": len(d), "slope_per_residue": float(slope),
            "slope_per_100_residues": float(slope) * 100,
            "intercept": float(intercept), "pearson_r": r, "r_squared": r ** 2,
            "median_length": float(x.median()),
            "frac_over_800": float((x > 800).mean())}
