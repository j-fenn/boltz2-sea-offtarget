# %% [markdown]
# # 07 — Monomer versus dimer, and a bar that measures the target
#
# A second experiment, absent from the earlier repository: the same drugs modelled against the
# same targets with interface-specific metrics — `Ligand_ipTM`, `Interface_PDE`,
# `Interface_pLDDT`. It was run to check whether the interface scores would behave better than
# the global confidence score. The answer is interesting and not the one that was wanted.

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
import yaml
from pathlib import Path
from offtarget import figures as F

pd.set_option("display.width", 140)
dimer = pd.read_parquet("data/processed/dimer_panel.parquet")
print(f"{len(dimer)} modelled pairs across {dimer.Target.nunique()} targets")
dimer.groupby(["sheet", "Target"]).size().to_frame("pairs")

# %% [markdown]
# ## Three sheets, two defects
#
# The workbook splits one experiment across sheets that grew as it ran: `Raw` (four targets),
# `MAOA`, and `Transporters`. A fourth sheet, `-PDE>5`, is `Raw` again with a filter applied
# and seven blank rows — reading it would double-count 51 of the 107 pairs, so `pipeline.py`
# does not.
#
# Two defects are repaired on the way in. `Raw` has a column-shift: 15 SLC6A2 rows carry a
# **SMILES string** in the `Experiment result` column, a paste that landed one column to the
# left. The `MAOA` sheet has no label column at all. Both are recovered by joining back to the
# master table on (target, drug) rather than by retyping values into a spreadsheet.

# %%
raw = pd.read_excel("data/boltz_outputs/PDBdimerresult.xlsx", sheet_name="Raw")
raw.columns = [c.strip() for c in raw.columns]
bad = raw[~raw["Experiment result"].astype(str).str.lower().isin(
    ["inactive", "true positive", "nan"])]
print(f"{len(bad)} rows with a SMILES string in the label column")
bad[["Target", "Drug", "Experiment result"]].head(5)

# %%
print(f"labels recovered: {int(dimer.label.notna().sum())} of {len(dimer)}")
dimer.label.value_counts(dropna=False).to_frame("rows")

# %% [markdown]
# ## The pass rate is a property of the target
#
# The bar cited in the March 2026 work as indicating crystal-structure-quality interface
# prediction was `Interface_PDE < 1` and `Ligand_ipTM > 0.9`.

# %%
d = dimer.dropna(subset=["Ligand_ipTM", "Interface_PDE"]).copy()
d["clears_bar"] = (d.Interface_PDE < 1) & (d.Ligand_ipTM > 0.9)
summary = d.groupby("Target").agg(n=("drug", "size"), passed=("clears_bar", "sum"),
                                  median_ipDE=("Interface_PDE", "median"),
                                  median_ipTM=("Ligand_ipTM", "median"))
summary["pass_rate"] = (summary.passed / summary.n).round(2)
summary.sort_values("pass_rate", ascending=False).round(3)

# %%
F.dimer_pass_rates(d, Path("results/figures/fig4_dimer_pass_rates.png"))
from IPython.display import Image, display
display(Image("results/figures/fig4_dimer_pass_rates.png"))

# %% [markdown]
# **SLC29A1 passes 2 of 2. SLC6A4 passes 7 of 16. MAOA passes 10 of 24. SLC6A3 passes 0 of 14**,
# with a median interface PDE of 3.54 — well outside the bar, for every drug tested.
#
# SLC6A3 and SLC6A4 are the same transporter family with the same fold. One of them fails
# completely and the other passes about half the time. Whatever this bar measures, it is
# tracking the target rather than the drug — which makes it useless for the job it was used
# for, which was picking out drugs.

# %% [markdown]
# ## The pass rate tracks the modelled oligomeric state
#
# Joining the pass rate to what `targets.yaml` says was actually modelled explains most of the
# spread across targets.

# %%
state = {t: v["arms"]["pdb"]["oligomeric_state"]
         for t, v in yaml.safe_load(Path("inputs/targets.yaml").read_text())["targets"].items()
         if "pdb" in v["arms"]}
d["modelled_as"] = d.Target.map(state)
d.groupby(["modelled_as", "Target"]).agg(n=("drug", "size"), passed=("clears_bar", "sum"),
                                         median_ipDE=("Interface_PDE", "median")).round(2)

# %%
by_state = d.groupby("modelled_as").agg(n=("drug", "size"), passed=("clears_bar", "sum"),
                                        median_ipDE=("Interface_PDE", "median"),
                                        median_ipTM=("Ligand_ipTM", "median"))
by_state["pass_rate"] = (by_state.passed / by_state.n).round(3)
by_state.round(3)

# %% [markdown]
# **Inputs modelled as homodimers pass 2 of 51 (4%). Inputs modelled as monomers pass 19 of 56
# (34%).** Median interface PDE is 5.28 for the dimers against 1.18 for the monomers.
#
# There is a mechanical reading of this that fits: given two protein chains, the interface
# metrics have a large protein–protein interface to describe as well as the ligand pocket, and
# the reported value is no longer about the ligand. That would explain why a bar built from
# these metrics tracks the target rather than the drug — the dimers are the targets that fail.
#
# This is an association across eight targets with oligomeric state assigned by hand, not a
# controlled experiment, and it cannot separate oligomeric state from target identity. It is
# the hypothesis that fits, offered as one.

# %% [markdown]
# ## Do the interface metrics discriminate at all?
#
# The reason for running this experiment was the hope that interface-specific metrics would
# succeed where the global confidence score failed. They do not.

# %%
from offtarget.metrics import stratified_auc
labelled = d[d.label.notna()].copy()
labelled["neg_interface_PDE"] = -labelled.Interface_PDE
print(f"n = {len(labelled)}   "
      f"assay-active = {int((labelled.label == 'assay_active').sum())}")
pd.DataFrame([{"score": c, "within-target AUC": round(stratified_auc(labelled, c)[0], 3),
               "targets used": stratified_auc(labelled, c)[1]}
              for c in ["Ligand_ipTM", "neg_interface_PDE", "Interface_pLDDT"]])

# %% [markdown]
# All three sit at chance: **0.488, 0.502, 0.493**. This is a separate set of metrics from a
# separate set of runs, and it reaches the same place as notebook 03. The problem is not that
# the wrong confidence number was read; every structural quality score available here fails to
# separate confirmed binders from confirmed non-binders on this dataset.

# %% [markdown]
# ## And the targets differ in their inputs
#
# From the audit in notebook 02: SLC6A4's input is the canonical sequence plus 55 residues of
# purification scaffold, modelled as a monomer. SLC6A3's input aligns to only 67% of the human
# dopamine transporter — 22 substitutions, 91 inserted residues in nine blocks — also modelled
# as a monomer. Meanwhile SLC6A2, the third member of the same family, is modelled as a
# **homodimer**.

# %%
cfg = yaml.safe_load(Path("inputs/targets.yaml").read_text())
rows = []
for t in ["SLC6A2", "SLC6A3", "SLC6A4"]:
    for arm_name, arm in cfg["targets"][t]["arms"].items():
        rows.append({"target": t, "arm": arm_name, "pdb": arm["pdb_id"],
                     "oligomeric state": arm["oligomeric_state"],
                     "identity to canonical": arm["audit"]["identity_to_canonical"],
                     "inserted residues": arm["audit"]["inserted_residues"],
                     "tags": ", ".join(arm["audit"]["construct_tags"]) or "—"})
pd.DataFrame(rows)

# %% [markdown]
# Three members of one transporter family, three different modelling decisions, no stated
# rule for any of them — and the metric they are being scored on varies more across those
# three targets than it does across the drugs being tested.
#
# This is a confounded comparison, not a controlled experiment: oligomeric state, construct
# identity and target are varied together, and there are five targets. It cannot attribute the
# effect to oligomeric state. What it does establish is that **the interface bar was reading
# something other than ligand binding**, which is enough to disqualify it as a nomination
# criterion in notebook 06.
