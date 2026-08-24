# %% [markdown]
# # 05 — Chemical similarity, and which direction it points
#
# SEA nominates a drug–target pair from ligand similarity: the drug resembles known ligands of
# the target. A natural check is whether that similarity predicts which nominations survive
# the bench. It does not — and an earlier write-up of this work reported it pointing the wrong
# way, which is worth pinning down because a sign error and a real inversion look identical in
# a table.

# %%
import sys, os
from pathlib import Path

# Work from the repository root wherever the notebook is launched from.
_root = Path.cwd().resolve()
while not (_root / "src" / "offtarget").exists():
    _root = _root.parent
os.chdir(_root)
sys.path.insert(0, str(_root / "src"))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from offtarget.pipeline import sea_baseline
from offtarget.metrics import stratified_auc

pd.set_option("display.width", 140)
sea = sea_baseline(Path("data/raw/Predictions.dat"), Path("data/raw/pid_confirmation_status.dat"))
assayed = sea[sea.label.notna()].dropna(subset=["Max Tc", "neg_log_evalue"])
print(f"assayed predictions: {len(assayed)}   "
      f"confirmed: {int((assayed.label == 'assay_active').sum())}")

# %% [markdown]
# ## SEA's own scores against its own outcomes
#
# One source, one Tc column, one E-value column, so there is nothing to mismatch.

# %%
rows = []
for c in ["Max Tc", "neg_log_evalue", "Charge Probability"]:
    d = assayed.dropna(subset=[c])
    yy = (d.label == "assay_active").astype(int)
    auc, used = stratified_auc(d, c)
    rows.append({"score": c, "n": len(d),
                 "pooled AUC": round(roc_auc_score(yy, d[c]), 3),
                 "within-target AUC": round(auc, 3), "targets used": used})
pd.DataFrame(rows)

# %% [markdown]
# Max Tc reaches 0.536 within target, the E-value 0.580. Both are close to chance and both are
# **above** it. Chemical similarity to known ligands is barely informative about whether the
# nominated interaction is real — but it is not anti-informative.
#
# ## Where 0.104 came from
#
# An earlier write-up reported a SEA Tc AUC of 0.104. An AUC of 0.104 is a strongly
# *anti*-predictive score: it would mean the more a drug resembles known ligands of a target,
# the less likely it is to bind it. That is not a finding, it is a bug — and the source of it
# is visible in the file that number came from.

# %%
xl = pd.ExcelFile("data/boltz_outputs/boltzresults_individual.xlsx")
true_sea, false_sea = xl.parse("TrueSea"), xl.parse("FalseSea")
print("TrueSea  similarity columns:",
      [c for c in true_sea.columns if "Tc" in str(c)])
print("FalseSea similarity columns:",
      [c for c in false_sea.columns if "Tc" in str(c)])

# %% [markdown]
# The actives sheet carries `ECFP_4 Tc` and `Combined Tc`. The inactives sheet carries
# `Max Tc`. There is no column common to both, so any AUC computed across them compared two
# different quantities — and `Combined Tc` and `Max Tc` are on different scales.

# %%
pd.DataFrame({
    "Combined Tc (actives sheet)": true_sea["Combined Tc"].describe(),
    "Max Tc (inactives sheet)": false_sea["Max Tc"].describe(),
}).round(3)

# %% [markdown]
# The medians differ by enough to produce an AUC near zero on their own, whichever way the
# classes fall. **The inversion was an artefact of joining two different columns**, and the
# repository does not carry the 0.104 figure anywhere.
#
# ## The honest version
#
# Within target, SEA's similarity score has an AUC of 0.536 for distinguishing its own
# confirmed predictions from its own disproved ones. That is a much less exciting sentence
# than an inversion, and it is the one the data supports.

# %%
per = assayed.groupby("Target").agg(n=("Max Tc", "size"),
                                    active=("label", lambda s: (s == "assay_active").sum()))
per = per[(per.active > 0) & (per.active < per.n)]
print(f"targets contributing both classes: {len(per)}")
per.sort_values("n", ascending=False).head(10)
