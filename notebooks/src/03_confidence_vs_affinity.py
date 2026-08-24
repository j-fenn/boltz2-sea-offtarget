# %% [markdown]
# # 03 — Confidence, ipTM, and the affinity head
#
# The question: given a target–drug pair that Boltz-2 has modelled, does any of its output
# separate the pairs the 2012 assay called active from the ones it called inactive?
#
# The answer depends entirely on which output you read, and the difference is the whole
# result.

# %%
import sys, os
from pathlib import Path

# Work from the repository root wherever the notebook is launched from.
_root = Path.cwd().resolve()
while not (_root / "src" / "offtarget").exists():
    _root = _root.parent
os.chdir(_root)
sys.path.insert(0, str(_root / "src"))

import pandas as pd
from offtarget.metrics import auc_table, stratified_auc, bootstrap_auc_ci, per_target_auc
from offtarget import figures as F
pd.set_option("display.width", 140)
audited = pd.read_parquet("data/processed/pairs_audited.parquet")
audited = audited[audited.label.notna()]
print(f"n = {len(audited)}   assay-active = {int((audited.label == 'assay_active').sum())}   "
      f"assay-inactive = {int((audited.label == 'assay_inactive').sum())}   "
      f"targets = {audited.Target.nunique()}")

# %% [markdown]
# ## Why stratify
#
# Pooled AUC across this table is close to meaningless. Targets differ enormously in how many
# of their tested drugs turned out to be active, so a score that has learned nothing except
# *which target this is* still ranks well pooled.

# %%
hit = audited.groupby("Target").agg(n=("drug", "size"),
                                    active=("label", lambda s: (s == "assay_active").sum()))
hit["hit_rate"] = hit.active / hit.n
hit.sort_values("hit_rate", ascending=False).head(8)

# %% [markdown]
# Stratifying within target removes that channel. Only targets contributing both classes can
# say anything about ranking — 34 of the 63 here.

# %%
table = auc_table(audited)
table.round(3)

# %% [markdown]
# | Score | Pooled | Within-target |
# |---|---|---|
# | Confidence (0.8·pLDDT + 0.2·ipTM) | 0.563 | **0.527** |
# | ipTM (interface only) | 0.592 | **0.486** |
# | Affinity probability | 0.731 | 0.683 |
# | −Affinity score | 0.771 | **0.710** |
#
# **Both structural confidence metrics sit at chance. Both affinity outputs discriminate.**
#
# The obvious objection to an earlier version of this result was that Boltz-2's confidence is
# `0.8·pLDDT + 0.2·ipTM` and therefore dominated by fold quality — that the interface metric,
# ipTM, would tell a different story. It does tell a different story: ipTM is the *worst* of
# the four. It is not that the wrong number was read.

# %% [markdown]
# ## How much of this is noise
#
# The interval resamples targets, not pairs. Pairs within a target share a protein, a
# construct and an oligomeric state; resampling rows would treat 34 targets' worth of
# information as 708 independent observations.

# %%
ci = pd.DataFrame([{"score": name, "within-target AUC": round(stratified_auc(audited, c)[0], 3),
                    "95% CI": tuple(round(v, 3) for v in bootstrap_auc_ci(audited, c))}
                   for c, name in
                   [("confidence", "Confidence"), ("iptm", "ipTM"),
                    ("affinity_probability", "Affinity probability"),
                    ("neg_affinity_score", "-Affinity score")]])
ci

# %% [markdown]
# Both confidence intervals include 0.5. Neither affinity interval does. The intervals are
# wide — the whole table rests on 34 proteins — and that width is part of the result.

# %%
F.roc_panel(audited, Path("results/figures/fig1_roc_confidence_vs_affinity.png"))
F.within_target_bars(audited, Path("results/figures/fig2_within_target_auc.png"))
from IPython.display import Image, display
display(Image("results/figures/fig2_within_target_auc.png"))

# %% [markdown]
# ## Per target
#
# The pooled figure hides that ipTM is below chance on more targets than it is above.

# %%
per = per_target_auc(audited, "iptm").merge(
    per_target_auc(audited, "neg_affinity_score"), on="Target", suffixes=("_iptm", "_affinity"))
print(f"targets where ipTM beats chance:     {int((per.auc_iptm > 0.5).sum())} of {len(per)}")
print(f"targets where affinity beats chance: {int((per.auc_affinity > 0.5).sum())} of {len(per)}")
per.sort_values("weight_iptm", ascending=False)[
    ["Target", "n_iptm", "n_pos_iptm", "auc_iptm", "auc_affinity"]].head(12).round(3)

# %% [markdown]
# ## Robustness: dropping the target with the known broken input
#
# One target, ADRA2A, was modelled against a 32-residue fragment of a helix (see notebook 01).
# Removing its 20 rows changes nothing that matters.

# %%
primary = pd.read_parquet("data/processed/pairs.parquet")
primary = primary[primary.label.notna()]
comparison = auc_table(audited)[["score", "within_target_auc"]].merge(
    auc_table(primary)[["score", "within_target_auc"]], on="score",
    suffixes=(" (n=708, as audited)", " (n=688, ADRA2A dropped)"))
comparison.round(3)

# %% [markdown]
# ## Disagreement with the earlier extraction
#
# The master workbook used above is a later, ipTM-carrying extraction. The repository also
# holds the earlier one, `boltzresults_individual.xlsx`, where the two sequence arms are in
# separate sheets. Running the same measure there:

# %%
xl = pd.ExcelFile("data/boltz_outputs/boltzresults_individual.xlsx")

def arm(true_sheet, false_sheet):
    frames = []
    for sheet, lab in [(true_sheet, "assay_active"), (false_sheet, "assay_inactive")]:
        d = xl.parse(sheet)
        d.columns = [str(c).strip() for c in d.columns]
        d = d.rename(columns={"Confidence_score": "confidence", "Affinity": "affinity",
                              "Affinity_score": "affinity_probability",
                              "Affinity, boltz2": "affinity"})
        d["Target"] = d.Target.astype(str).str.strip().str.upper()
        d["label"] = lab
        for c in ["confidence", "affinity", "affinity_probability"]:
            d[c] = pd.to_numeric(d.get(c), errors="coerce")
        frames.append(d[["Target", "Drug", "confidence", "affinity",
                         "affinity_probability", "label"]])
    out = pd.concat(frames, ignore_index=True)
    out["neg_affinity_score"] = -out.affinity
    return out

rows = []
for name, (t, f) in {"PDB arm": ("TruePDB", "FalsePDB"),
                     "UniProt arm": ("TrueUni", "FalseUni")}.items():
    d = arm(t, f)
    for c in ["confidence", "affinity_probability", "neg_affinity_score"]:
        dd = d.dropna(subset=[c])
        auc, used = stratified_auc(dd, c)
        rows.append({"extraction": name, "score": c, "within-target AUC": round(auc, 3),
                     "n": len(dd), "targets": used})
pd.DataFrame(rows)

# %% [markdown]
# The two extractions agree on the finding and not on the digits: confidence 0.509 (PDB) and
# 0.600 (UniProt) against 0.527 for the merged confidence column, affinity 0.701–0.707 against
# 0.710. Different row sets, different arms, one carrying ipTM and one not.
#
# Two figures from earlier write-ups of this work do **not** reproduce and are not used
# anywhere in this repository:
#
# - *"UniProt confidence 0.410"* — the UniProt arm gives 0.600 here. The reported value is
#   below chance where this one is above it, which is the signature of a sign error somewhere
#   in the earlier extraction.
# - *"SEA max Tc 0.104"* — an AUC of 0.104 means a score that is strongly *anti*-predictive,
#   which SEA's own similarity measure is not. Computed from the primary source in notebook 01
#   it is 0.536. The earlier number appears to have compared `Combined Tc` (the only Tc column
#   in the actives sheet) against `Max Tc` (the only one in the inactives sheet) — two
#   different quantities.

# %% [markdown]
# ## Sequence length
#
# The stated hypothesis for the sub-chance confidence result was that Boltz-2 degrades on long
# chains. It deserves a number rather than a mention.

# %%
from offtarget.metrics import sequence_length_regression

aa = pd.read_excel("data/boltz_outputs/all_with_smiles_and_AA.xlsx", sheet_name="Sheet1")
aa.columns = [str(c).strip() for c in aa.columns]
aa["Target"] = aa.Target.astype(str).str.strip().str.upper()
aa["drug_key"] = aa.Drug.astype(str).str.strip().str.lower()
aa["seq_length"] = aa["AA sequence (Uniprot)"].astype(str).str.len()
aa = aa[aa.seq_length > 10]

merged = audited.merge(aa[["Target", "drug_key", "seq_length"]].drop_duplicates(
    ["Target", "drug_key"]), on=["Target", "drug_key"], how="left")
pd.DataFrame([{"score": s, **sequence_length_regression(merged, score=s)}
              for s in ["confidence", "iptm", "affinity_probability"]]).round(4)

# %% [markdown]
# Confidence barely moves with length: **−0.0015 per 100 residues, r = −0.05, R² = 0.003.**
# ipTM does move — −0.012 per 100 residues, r = −0.35 — but ipTM is also the score that
# cannot tell binders from non-binders at all, so a length dependence in it explains nothing
# about discrimination.
#
# A slope is not a mechanism. Length here is confounded with target identity, target class and
# construct choice, none of which are controlled. The hypothesis is not supported by this
# regression, and it is not ruled out by it either.
