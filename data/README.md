# Data provenance

Everything in this repository descends from one published file. This document records the
chain, the row count at each step, and the defects found along the way. It is the document to
read first, because it decides whether anything else here is worth trusting.

Rebuild all of it with `offtarget build`, which writes `data/processed/` and a
`manifest.json` carrying a SHA-256 for every input and a row count for every stage.

---

## The chain

```
data/raw/NIHMS373071-supplement-3.xls
  Lounkine et al., "Large-scale prediction and testing of drug activity on
  side-effect targets", Nature 486:361-367 (2012).  doi:10.1038/nature11159
  │
  ├── S1  Drugs used in study ............   656 drugs, SMILES for all 656
  ├── S2  Safety targets .................    73 targets + Entrez gene IDs + class hierarchy
  ├── S3  Confirmed predictions ..........   174 rows: SEA E-value, Tc, confirmed activity [µM]
  ├── S4  Comparison to 1NN ..............   322
  ├── S5  Target-ADR associations ........ 3,257
  ├── S6  Novel off-target and ADRs ......   116
  ├── S7  Target promiscuity .............    73
  └── S8  Drug promiscuity ...............   656

data/raw/Predictions.dat ...............  4,759 rows -> 2,768 distinct predictions
data/raw/pid_confirmation_status.dat ...  2,768 predictions with a bench outcome
data/raw/inactive.dat ..................  1,328 rows for the 827 disproved predictions
data/raw/uniprot_reviewed_2025-11-10.fasta.gz .. 117 reviewed human sequences

  -> join predictions to their bench outcome ......... 1,023 assayed predictions
  -> Boltz-2, two sequence arms, 8 targets with committed inputs
  -> data/boltz_outputs/alltargetSHOICHET.xlsx ....... 710 scored pairs, 708 labelled
```

## The premise

**Of the 1,023 SEA predictions that were taken to the bench, 827 (80.8%) did not confirm.**
That failure rate is why this project exists. It is derived in `notebooks/01`, not quoted.

> ⚠️ Earlier write-ups of this work stated "46% of SEA's predicted interactions were disproved
> in the wet lab". That figure does not reproduce from these files under any denominator I
> could construct, and it is not used anywhere in this repository.

### The counting trap behind it

`Predictions.dat` has 4,759 rows but only **2,768 distinct `Prediction_Id`**: a single
drug–target prediction is repeated once per ChEMBL gene entry it maps to. Counting rows rather
than predictions inflates every total by about 1.7×. `pipeline.sea_baseline` de-duplicates
once, at the source.

## Ground truth is positive–unlabeled

Lounkine measured **functional activity**. A pair recorded as inactive is a pair that did not
move that particular readout at the concentrations tested — not a pair shown not to bind. A
neutral antagonist can occupy an orthosteric site and produce no functional signal.

This matters for every number here. Some fraction of the "inactive" class may be binders, so
the discrimination figures in `notebooks/03` are best read as a **lower bound on the model and
an upper bound on the label quality**. It also matters for `notebooks/06`, which is about what
happens when you try to identify those binders from a structure score.

## The label rename

The upstream pipeline mapped SEA's bench outcome onto `True positive` / `False positive`,
meaning *SEA predicted this and the assay agreed / disagreed*. Those names then travelled into
an analysis that computes ROC curves, where "true positive" means something else entirely —
one name carrying two meanings inside one table.

| Source value | Renamed to |
|---|---|
| `True positive`, `active` | `assay_active` |
| `False positive`, `inactive`, `inactive inactive` | `assay_inactive` |

Every table is renamed on ingest and the originals appear nowhere downstream. Case and
trailing whitespace are load-bearing in the source workbooks — `all_with_both_AA.xlsx` carries
576 rows of `'inactive '` and a single `'Inactive'` — so normalisation is case- and
whitespace-insensitive.

## Known defects, and what was done about each

| Defect | Where | Handling |
|---|---|---|
| **ADRA2A pocket sequence is a 32-residue helix fragment** — `TSSIVHLCAISLDRYWSITQAIEYNLKRTPRR`, pasted where a binding-pocket sequence belonged | `all_with_both_AA.xlsx`, PDB arm | Kept in `raw/` and `boltz_outputs/` as the record. Excluded **by name with a logged reason** in `pipeline.KNOWN_BAD_INPUTS`, never silently dropped. `pairs_audited.parquet` keeps its 20 rows; `pairs.parquet` does not. `notebooks/03` shows the result is unchanged either way. |
| **Column shift** — 15 SLC6A2 rows carry a SMILES string in the `Experiment result` column | `PDBdimerresult.xlsx`, `Raw` sheet | Detected as labels that fail to parse; recovered by joining back to the master table on (target, drug). |
| **No label column at all** — 39 rows | `PDBdimerresult.xlsx`, `MAOA` sheet | Same recovery. |
| **A duplicate filtered view** — the `-PDE>5` sheet is `Raw` again with a filter and seven blank rows | `PDBdimerresult.xlsx` | Not read. Reading it would double-count 51 of the 107 pairs. |
| **Side-by-side tables with no join key** — the PDB run, the UniProt run and the assay result are three tables pasted into one sheet, aligned by row position | `boltz2_result.xlsx` | Read by column position in one documented function, `transforms.load_paired_affinities`, rather than at each call site. |
| **Inconsistent back-transform** — `=POWER(10,x)` for one arm, `=EXP(x)` for the other, on the same quantity | `boltz2_result.xlsx`, columns H and O | Reproduced deliberately in `transforms.workbook_legacy_uniprot`, called only from `notebooks/04`. See the root README. |
| **SMILES re-derived by fuzzy PubChem name lookup** when all 656 drugs already carry published SMILES in S1 | `add_smiles_from_pubchem.py` | S1 is authoritative. The PubChem output is kept as `pubchem_smiles_expanded.xlsx` and compared against S1 by InChIKey in `notebooks/01`. |
| **Two different similarity columns compared as one** — `Combined Tc` in the actives sheet, `Max Tc` in the inactives sheet | `boltzresults_individual.xlsx` | The SEA baseline is computed from the primary source instead. See `notebooks/05`. |

Every one of these is a spreadsheet failure mode, and each is an argument for the same thing:
building tables in code, where a defect can be detected, named and tested against.

## What the run counts do and do not reconcile

Earlier write-ups quote **1,085 Boltz-2 runs**. That number is reachable — 668 UniProt-arm runs
(128 active + 540 inactive) plus 417 PDB-arm runs — but it does not reconcile against what is
on disk:

| Artifact | Rows |
|---|---|
| `boltz2_result.xlsx` — the paired PDB/UniProt extraction | 128 active + 539 inactive = **667** |
| `rcsb_result.xlsx` — the PDB arm | **417** |
| `alltargetSHOICHET.xlsx` — the later master, the only file carrying ipTM | **710** (63 targets) |
| Committed model inputs (`inputs/generated`) | **158** (8 targets) |

**There is no run manifest.** Nothing on disk records which runs were submitted, with what
settings, or in what order, so the 1,085 figure cannot be checked and is not used in this
repository. Only 158 of the inputs survive — about 15% — and they cover 8 of the 63 targets in
the master table. Everything in `notebooks/02` is therefore an audit of a sample of the inputs,
and it is described as one.

The reproducible counts are the ones in `data/processed/manifest.json`.

## Contents

### `raw/` — primary sources, unmodified

| File | What it is |
|---|---|
| `NIHMS373071-supplement-3.xls` | Lounkine 2012 supplementary data, eight sheets |
| `Predictions.dat` | 4,759 rows / 2,768 SEA predictions with E-value, Max Tc, charge filter |
| `pid_confirmation_status.dat` | bench outcome per prediction |
| `inactive.dat` | the disproved subset |
| `uniprot_reviewed_2025-11-10.fasta.gz` | 117 reviewed human sequences for the safety targets |

### `boltz_outputs/` — scored model outputs, as produced

Kept unmodified, including their defects; they are the record. `alltargetSHOICHET.xlsx` is the
only one carrying **ipTM**, and it is the basis of the headline result.

### `processed/` — rebuilt by `offtarget build`, not committed

`safety_targets`, `uniprot_map`, `drugs`, `predictions`, `confirmed`, `pairs`,
`pairs_audited`, `dimer_panel`, `smiles_comparison` — all Parquet, plus `manifest.json`.
`pairs.parquet` is the analysis set; `pairs_audited.parquet` retains the ADRA2A rows so the
published numbers can be reproduced exactly.
