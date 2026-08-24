# %% [markdown]
# # 01 — Rebuilding the dataset from the published supplement
#
# Everything downstream starts here: a single published file, `NIHMS373071-supplement-3.xls`
# (Lounkine et al., *Nature* 486:361–367, 2012), plus the raw SEA prediction dump that came
# with it. This notebook runs the pipeline and checks the row count at every stage.
#
# The original version of this was thirteen scripts that had to be run in a particular order,
# with files renamed by hand in between, absolute paths pointing at one laptop, and one script
# that read a `.fasta` which only ever existed on disk as `.fasta.gz`.

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
from offtarget.pipeline import build, sea_baseline
pd.set_option("display.width", 140)
manifest = build()
pd.Series(manifest.stages, name="rows").to_frame()

# %% [markdown]
# ## What the source file contains
#
# Eight sheets. Four of them matter here.

# %%
xl = pd.ExcelFile("data/raw/NIHMS373071-supplement-3.xls")
pd.DataFrame([{"sheet": s, "rows": len(xl.parse(s))} for s in xl.sheet_names])

# %% [markdown]
# ## The premise: how often SEA was wrong
#
# The project exists because SEA's predictions were taken to the bench and most of them
# failed. That number is worth deriving rather than quoting, and deriving it turns up a
# counting trap.
#
# `Predictions.dat` has 4,759 rows but only 2,768 distinct `Prediction_Id`: one drug–target
# prediction appears once per ChEMBL gene entry it maps to. Counting rows inflates every
# total by about 1.7×.

# %%
pred = pd.read_csv("data/raw/Predictions.dat", sep="\t")
print(f"rows in Predictions.dat        {len(pred):,}")
print(f"distinct Prediction_Id         {pred.Prediction_Id.nunique():,}")

sea = sea_baseline(Path("data/raw/Predictions.dat"), Path("data/raw/pid_confirmation_status.dat"))
assayed = sea[sea.label.notna()]
n_conf = int((assayed.label == "assay_active").sum())
n_dis = int((assayed.label == "assay_inactive").sum())
print(f"\npredictions taken to the bench {len(assayed):,}")
print(f"  confirmed active             {n_conf:,}")
print(f"  not confirmed                {n_dis:,}  ({n_dis / len(assayed):.1%})")

# %% [markdown]
# **80.8% of the SEA predictions that were assayed did not confirm.** That is the premise of
# the project, and it is a stronger premise than the 46% figure that earlier write-ups of this
# work quoted. 46% does not reproduce from these files under any denominator I could construct;
# it is not used anywhere in this repository.

# %% [markdown]
# ## SEA's own scores as a baseline
#
# Before asking whether a structure model can rank these pairs, it is worth asking whether the
# thing that generated them can. Within target, SEA's own similarity and E-value barely order
# its confirmed predictions above its disproved ones.

# %%
from offtarget.metrics import stratified_auc
from sklearn.metrics import roc_auc_score

a = assayed.dropna(subset=["Max Tc", "neg_log_evalue"])
y = (a.label == "assay_active").astype(int)
pd.DataFrame([
    {"score": c,
     "pooled AUC": round(roc_auc_score(y, a[c]), 3),
     "within-target AUC": round(stratified_auc(a, c)[0], 3),
     "targets used": stratified_auc(a, c)[1]}
    for c in ["Max Tc", "neg_log_evalue"]
])

# %% [markdown]
# ## The label rename
#
# The upstream scripts mapped SEA's bench outcome to `True positive` / `False positive`,
# meaning *SEA predicted this and the assay agreed / disagreed*. Those names then travelled
# into an analysis that computes ROC curves, where "true positive" means something else
# entirely — one name, two meanings, in the same table. Everything is renamed on the way in.

# %%
from offtarget.pipeline import LABEL_MAP
pd.Series(LABEL_MAP, name="renamed to").to_frame()

# %% [markdown]
# Case and trailing whitespace are load-bearing in the source workbooks: `all_with_both_AA.xlsx`
# carries 576 rows of `'inactive '` and a single `'Inactive'`. A case-sensitive comparison
# silently drops one of them.

# %%
raw = pd.read_excel("data/boltz_outputs/all_with_both_AA.xlsx")
raw.Label.value_counts().to_frame("rows")

# %% [markdown]
# ## The SMILES step, and what re-deriving it cost
#
# The upstream pipeline resolved every drug name through PubChem with
# `pcp.get_compounds(name, "name")[0]` and no validation — while all 656 drugs already carry
# published SMILES in sheet S1 of the file it was already reading. S1 is authoritative here.
#
# Comparing the two by InChIKey (RDKit, so the comparison is chemical rather than textual):

# %%
smi = pd.read_parquet("data/processed/smiles_comparison.parquet")
print(smi.status.value_counts().to_string())
print()
print(smi.name_match.value_counts().to_string())

# %%
smi[smi.status != "identical"][
    ["drug", "matched_S1_entry", "status", "formula_pubchem", "formula_S1"]]

# %% [markdown]
# Three of 220 resolved to a genuinely different compound, and one to a different tautomer:
#
# - **Anecortave** — PubChem returned the free alcohol; the drug is anecortave *acetate*.
# - **Olmesartan** — PubChem returned the free acid; the drug is the medoxomil prodrug ester.
# - **Testosterone_Propionate** — the name had to be truncated to match S1 at all, and then
#   matched plain testosterone rather than the propionate ester.
# - **Warfarin** — same formula, different tautomer (open-chain keto vs the 4-hydroxycoumarin enol).
#
# Fourteen more matched only because the lookup here is case-insensitive; the original merge
# was not, so those fourteen would have gone through as unmatched.
#
# Small numbers, and that is the useful part of the result: the shortcut mostly worked. It is
# still a shortcut that reintroduced avoidable error into a table whose correct values were
# one sheet away.

# %% [markdown]
# ## The excluded rows
#
# One target is filtered by name rather than silently dropped.

# %%
for note in manifest.notes:
    print("•", note, "\n")

# %% [markdown]
# ## Manifest
#
# Every input carries a content hash, so a rebuilt table can be traced to the exact bytes it
# came from.

# %%
import json
json.loads(Path("data/processed/manifest.json").read_text())["inputs"]
