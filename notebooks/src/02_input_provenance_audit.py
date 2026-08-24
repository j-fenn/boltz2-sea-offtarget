# %% [markdown]
# # 02 — What was actually handed to the model
#
# A Boltz-2 run means nothing beyond the sequence it was given. The 158 inputs behind this
# project were written by ten near-identical generator scripts, each carrying one protein
# sequence pasted from a web page into a string literal. Nobody could review those decisions
# because there was nothing to review: the choices existed only as differences between copies
# of a script.
#
# This notebook reads the inputs back and counts what is in them.

# %%
import sys, os
from pathlib import Path

# Work from the repository root wherever the notebook is launched from.
_root = Path.cwd().resolve()
while not (_root / "src" / "offtarget").exists():
    _root = _root.parent
os.chdir(_root)
sys.path.insert(0, str(_root / "src"))

import gzip, re, yaml
import pandas as pd
from pathlib import Path
from offtarget.constructs import audit_inputs, diff_arms, find_construct_tags, TAG_MOTIFS

pd.set_option("display.width", 150)
pd.set_option("display.max_colwidth", 60)

audit = audit_inputs(Path("inputs/generated"))
print(f"{len(audit)} model inputs")

# %% [markdown]
# ## 1. Expression-construct scaffold
#
# Purification tags are cloning artefacts. They are in the construct that was crystallised,
# absent from the protein in a cell, and they lengthen the chain the model folds.

# %%
pd.Series(TAG_MOTIFS, name="motif").to_frame()

# %%
n_tag = int(audit.has_tags.sum())
print(f"inputs carrying expression-construct scaffold: {n_tag} of {len(audit)} "
      f"({n_tag / len(audit):.0%})")
audit[audit.has_tags].groupby(["directory", "tags"]).size().to_frame("inputs")

# %% [markdown]
# **29 of 158 inputs (18%) were run against a sequence containing purification scaffold.**
# All 16 SLC6A4 files carry a thrombin site, twin Strep-tag II, GS linkers and His12. All 13
# ADRA1A PDB files carry an N-terminal FLAG tag and a C-terminal His8.

# %% [markdown]
# ## 2. Oligomeric state, assigned per target with no stated rule

# %%
audit.groupby(["directory", "oligomeric_state"]).size().to_frame("inputs")

# %%
print(f"homodimer: {int((audit.n_chains == 2).sum())}    monomer: {int((audit.n_chains == 1).sum())}")

# %% [markdown]
# **102 inputs are homodimers and 56 are monomers.** SLC6A2 is modelled as a dimer while
# SLC6A3 and SLC6A4 — the same transporter family, the same fold — are monomers. ADRA1A, a
# GPCR, is a homodimer. Notebook 07 shows this choice moves the interface metrics more than
# the ligand does.

# %% [markdown]
# ## 3. What each arm actually is
#
# The project ran two arms per target and compared them as though the only difference were
# length: a "PDB" arm from a structure and a "UniProt" arm from the canonical sequence.
# Aligning each input against the reviewed UniProt record — from this project's own FASTA —
# says otherwise.

# %%
def canonical_sequences(path="data/raw/uniprot_reviewed_2025-11-10.fasta.gz"):
    out, header, buf = {}, None, []
    for line in gzip.open(path, "rt"):
        line = line.strip()
        if line.startswith(">"):
            if header: out[header] = "".join(buf)
            header, buf = line, []
        else: buf.append(line)
    out[header] = "".join(buf)
    return {m.group(1): s for h, s in out.items()
            if (m := re.match(r">\w+\|([A-Z0-9]+)\|", h))}

canon = canonical_sequences()
cfg = yaml.safe_load(Path("inputs/targets.yaml").read_text())

rows = []
for target, tcfg in cfg["targets"].items():
    ref = canon[tcfg["accession"]]
    for arm_name, arm in tcfg["arms"].items():
        d = diff_arms(arm["sequence"], ref)
        rows.append({"target": target, "arm": arm_name, "pdb": arm["pdb_id"],
                     "n_inputs": arm["n_inputs"], "chains": len(arm["chain_ids"]),
                     "len": d.input_length, "canonical_len": d.canonical_length,
                     "identity": round(d.identity, 3), "subs": len(d.substitutions),
                     "inserted": d.inserted_residues, "deleted": d.deleted_residues,
                     "verdict": d.verdict})
arms = pd.DataFrame(rows).sort_values(["target", "arm"])
arms

# %% [markdown]
# Five things fall out of that table.
#
# **SLC6A4's construct carries exactly 55 residues of scaffold.** Zero substitutions: the
# sequence is P31645 in full, followed by a thrombin site, two Strep-tags, GS linkers and
# His12. Every SLC6A4 claim in the March 2026 deck was made against this input.

# %%
slc6a4 = cfg["targets"]["SLC6A4"]["arms"]["pdb"]["sequence"]
d = diff_arms(slc6a4, canon["P31645"])
for pos, seg in d.insertions:
    print(f"inserted after canonical residue {pos - 1}: {len(seg)} residues\n  {seg}")
print("\nmotifs found:", [(t.name, t.start, t.end) for t in find_construct_tags(slc6a4)])

# %% [markdown]
# **The ADRA1A "PDB" arm is not the human protein.** It is a GPCR crystallography construct:
# three glycosylation sequons mutated, two thermostabilising substitutions in TM3, ICL3 cut
# out with a fusion remnant left behind, and tags at both ends. It was compared head-to-head
# with the "UniProt" arm as though the difference were length.

# %%
d = diff_arms(cfg["targets"]["ADRA1A"]["arms"]["pdb"]["sequence"], canon["P35348"])
print("point substitutions:", [f"{a}{p}{b}" for p, a, b in d.substitutions])
print("insertions:", [(p, s) for p, s in d.insertions])
print("deleted spans:", d.deletions)

# %% [markdown]
# **The two SLC6A2 arms are swapped.** The arm labelled "PDB 8HFE" is the canonical UniProt
# sequence, verbatim — 617 residues, 100% identity, nothing changed. The arm labelled
# "UniProt" is a 566-residue hand-trim. Whatever the comparison between them measured, it was
# not what its labels say.

# %%
arms[arms.target == "SLC6A2"]

# %% [markdown]
# **The "human PTGS1" arm is not human.** 33 substitutions spread evenly across the whole
# chain — median spacing 10 residues — with no tags and no engineering signature. Dispersed
# substitution at 93.9% identity is what an ortholog looks like, not a mutated human
# construct: PDB 3KK6 is the ovine enzyme. The UniProt arm for the same target *is* human, so
# the arm comparison for PTGS1 is also a cross-species comparison.

# %%
import numpy as np
d = diff_arms(cfg["targets"]["PTGS1"]["arms"]["pdb"]["sequence"], canon["P23219"])
pos = [p for p, _, _ in d.substitutions if isinstance(p, int)]
print(f"{len(d.substitutions)} substitutions between residues {min(pos)} and {max(pos)}")
print(f"median spacing: {np.median(np.diff(sorted(pos))):.0f} residues")
print(f"tags found: {[t.name for t in d.tags] or 'none'}")

# %% [markdown]
# **The "UniProt" arm is often not the UniProt sequence.** For ADRA1A only 70.4% of the arm
# aligns to the reviewed P35348 record; 42 residues, including a 19-residue block in ICL3,
# appear nowhere in it. The generator names it `# Clean ADRA1A Sequence (Residues ~20-370)`.

# %%
d = diff_arms(cfg["targets"]["ADRA1A"]["arms"]["uniprot"]["sequence"], canon["P35348"])
pd.DataFrame([{"inserted at canonical position": p, "residues": len(s),
               "present elsewhere in P35348": s in canon["P35348"], "segment": s}
              for p, s in d.insertions])

# %% [markdown]
# ## 4. Scaffold the motif table misses
#
# The six audited motifs give 29 of 158. The alignment finds one more construct the motif
# table does not match: SLC29A1 carries an HRV-3C protease site and linker at the C-terminus.

# %%
d = diff_arms(cfg["targets"]["SLC29A1"]["arms"]["pdb"]["sequence"], canon["Q99808"])
print("SLC29A1 insertions:", d.insertions)
print("\nmotif-based count:      29 of 158 (18%)")
print("including HRV-3C:       31 of 158 (20%)")

# %% [markdown]
# ## 5. The generator that does not generate its own output
#
# `generate_SLC29A1_yamls.py` sits in the directory containing the SLC29A1 inputs. Its header,
# its `pdb_sequence` and its `target_filter` all say SLC6A4 / 9HCO. It cannot have produced
# the files beside it, and there is no script in the tree that did.
#
# ## 6. Did any of it matter?
#
# The audit is only worth the space if the defects move the scores. Comparing tagged against
# untagged inputs, and dimers against monomers, with target held as a covariate:

# %%
pairs = pd.read_parquet("data/processed/pairs_audited.parquet")
pairs = pairs[pairs.label.notna()]

per_target_state = {}
for target, tcfg in cfg["targets"].items():
    for arm_name, arm in tcfg["arms"].items():
        d2 = diff_arms(arm["sequence"], canon[tcfg["accession"]])
        per_target_state[(target, arm_name)] = {
            "tagged": bool(d2.tags), "dimer": len(arm["chain_ids"]) == 2}

state = pd.DataFrame([{"Target": t, "arm": a, **v} for (t, a), v in per_target_state.items()])
modelled = state.groupby("Target").agg(any_tag=("tagged", "any"), any_dimer=("dimer", "any"))
sub = pairs.merge(modelled, left_on="Target", right_index=True, how="inner")
print(f"{len(sub)} pairs on the {sub.Target.nunique()} targets that have input files\n")
sub.groupby(["any_tag", "label"])[["confidence", "iptm", "affinity_probability"]].mean().round(3)

# %% [markdown]
# The comparison is badly confounded — tag status is a property of the target, not of the
# pair, and only eight targets have inputs in the tree at all — so the honest report is that
# **this dataset cannot separate a tag effect from a target effect.** The audit stands on
# what the inputs contain, not on a downstream effect it is not powered to measure. That is
# the negative result, and it is stated as one.

# %%
sub.groupby(["any_dimer"])[["confidence", "iptm", "affinity_probability"]].agg(["mean", "count"]).round(3)
