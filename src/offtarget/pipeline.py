"""Rebuild every analysis table from the published Lounkine 2012 supplement.

Ports thirteen loose scripts into one module with a row count at every stage and a
content hash for every input. The originals hard-coded absolute paths, depended on
files being renamed by hand between steps, and one of them read a ``.fasta`` that
only ever existed as ``.fasta.gz``. None of that survives here.

Stages, with the counts they must produce:

    safety_targets     S2                      ->    73 targets
    uniprot_map        reviewed FASTA          ->   117 sequences
    drugs              S1                      ->   656 drugs, 656 published SMILES
    predictions        Predictions.dat         -> 4,759 raw SEA predictions
    confirmed          S3                      ->   174 assayed predictions
    pairs              scored Boltz-2 outputs  ->   710 rows, 708 labelled

Run with::

    offtarget build --si data/raw/NIHMS373071-supplement-3.xls --out data/processed/
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("offtarget.pipeline")

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Labels.
#
# The upstream scripts mapped SEA's outcome to "True positive" / "False positive",
# meaning *SEA predicted this and the assay agreed / disagreed*. Those names then
# travelled into an analysis that computes ROC curves, where "true positive" means
# something else entirely. One name, two meanings, in the same table.
#
# Every label is renamed on the way in. The originals never appear downstream.
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "true positive":      "assay_active",
    "active":             "assay_active",
    "false positive":     "assay_inactive",
    "inactive":           "assay_inactive",
    "inactive inactive":  "assay_inactive",   # a doubled cell in the master workbook
}

# Excluded by name, with the reason, rather than silently dropped.
KNOWN_BAD_INPUTS = {
    "ADRA2A": (
        "The PDB-arm sequence for ADRA2A is 'TSSIVHLCAISLDRYWSITQAIEYNLKRTPRR' - a "
        "32-residue fragment of transmembrane helix 3 and the second intracellular "
        "loop, pasted where a binding-pocket sequence belonged. It is what "
        "copy-pasting from a structure page by hand produces. The rows are kept in "
        "data/raw and data/boltz_outputs as the record and excluded here by name."
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifest:
    """Row counts per stage and a content hash per input, written beside the outputs."""
    inputs: dict[str, str] = field(default_factory=dict)
    stages: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def record_input(self, path: Path) -> Path:
        path = Path(path).resolve()
        try:
            key = str(path.relative_to(REPO))
        except ValueError:          # an input from outside the repo tree
            key = str(path)
        self.inputs[key] = sha256(path)
        return path

    def record_stage(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        self.stages[name] = len(df)
        log.info("%-16s %6d rows", name, len(df))
        return df

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"inputs": self.inputs, "stages": self.stages, "notes": self.notes}, indent=2))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def safety_targets(si: Path, m: Manifest) -> pd.DataFrame:
    """S2 -> the 73 safety targets, their Entrez ids and their class hierarchy."""
    df = pd.read_excel(m.record_input(si), sheet_name="S2 Safety targets")
    df.columns = [c.strip() for c in df.columns]
    df = df[["Target", "Human Entrez gene IDs", "Target class level 1",
             "Target class level 2", "Target class level 3"]].copy()
    df = df.rename(columns={"Human Entrez gene IDs": "gene_id",
                            "Target class level 1": "class_1",
                            "Target class level 2": "class_2",
                            "Target class level 3": "class_3"})
    df["gene_id"] = df.gene_id.astype(str).str.replace(";", "", regex=False).str.strip()
    df["Target"] = df.Target.astype(str).str.strip().str.upper()
    return m.record_stage("safety_targets", df)


def uniprot_map(fasta_gz: Path, m: Manifest) -> pd.DataFrame:
    """Reviewed FASTA -> (target symbol, accession, sequence).

    Reads the gzip directly. The original script read an unzipped ``.fasta`` that
    is not in the tree, so the pipeline could not be run as committed.
    """
    records: list[tuple[str, str]] = []
    header, buf = None, []
    with gzip.open(m.record_input(fasta_gz), "rt") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(buf)))
                header, buf = line, []
            else:
                buf.append(line)
    if header:
        records.append((header, "".join(buf)))

    rows = []
    for h, seq in records:
        mm = re.match(r">\w+\|([A-Z0-9]+)\|(\w+)_HUMAN", h)
        if mm:
            rows.append({"accession": mm.group(1), "entry_name": mm.group(2),
                         "sequence": seq, "seq_length": len(seq)})
    return m.record_stage("uniprot_map", pd.DataFrame(rows))


def drugs(si: Path, m: Manifest) -> pd.DataFrame:
    """S1 -> 656 drugs with their published SMILES.

    The published SMILES are authoritative. The upstream pipeline ignored them and
    re-derived structures by fuzzy PubChem name lookup instead -- from sheet S1 of
    the very file it already had open. :func:`smiles_disagreements` measures what
    that cost.
    """
    df = pd.read_excel(m.record_input(si), sheet_name="S1 Drugs used in study")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Drug name": "drug", "SMILES": "smiles_published"})
    df["drug"] = df.drug.astype(str).str.strip()
    df["drug_key"] = df.drug.str.lower()
    return m.record_stage("drugs", df[["drug", "drug_key", "smiles_published",
                                       "Known human targets (Drugbank, ChEMBL)",
                                       "Target classes"]])


def predictions(pred_dat: Path, m: Manifest) -> pd.DataFrame:
    """Predictions.dat -> 4,759 raw SEA predictions."""
    df = pd.read_csv(m.record_input(pred_dat), sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df["chemblgene"] = df.chemblgene.astype(str).str.upper().str.strip()
    df["gene_id"] = df.gene_id.astype(str).str.strip()
    df = df.rename(columns={"o_name": "drug"})
    return m.record_stage("predictions", df)


def confirmed(si: Path, m: Manifest) -> pd.DataFrame:
    """S3 -> the 174 predictions that were taken to the bench, with measured activity."""
    df = pd.read_excel(m.record_input(si), sheet_name="S3 Confirmed predictions")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Drug name": "drug", "Target": "Target",
                            "Confirmed activity [uM]": "confirmed_activity_uM",
                            "SEA E-value": "sea_evalue", "Combined Tc": "combined_tc",
                            "ECFP_4 Tc": "ecfp4_tc"})
    df["drug_key"] = df.drug.astype(str).str.strip().str.lower()
    df["Target"] = df.Target.astype(str).str.strip().str.upper()
    return m.record_stage("confirmed", df)


def normalise_label(value) -> str | None:
    """Map an experimental-result cell onto ``assay_active`` / ``assay_inactive``.

    Case and trailing whitespace are load-bearing in the source workbooks: one file
    carries 576 rows of ``'inactive '`` and a single ``'Inactive'``, and a
    case-sensitive comparison silently drops one of them.
    """
    if not isinstance(value, str):
        return None
    return LABEL_MAP.get(value.strip().lower())


def pairs(master: Path, m: Manifest, exclude_known_bad: bool = True) -> pd.DataFrame:
    """The scored target-drug table: Boltz-2 outputs joined to the assay outcome.

    One row per target-drug pair actually modelled, carrying both structural
    confidence outputs (confidence, ipTM) and both affinity outputs (predicted log
    affinity, affinity probability).
    """
    df = pd.read_excel(m.record_input(master), sheet_name="Sheet1")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Drug_Name": "drug", "Confidence": "confidence",
                            "IPTM": "iptm", "Affinity Score": "affinity_score",
                            "Affinity Probability": "affinity_probability",
                            "Experimental Result": "raw_label",
                            "Drug SMILES": "smiles_as_run"})
    df["Target"] = df.Target.astype(str).str.strip().str.upper()
    df["drug"] = df.drug.astype(str).str.strip()
    df["drug_key"] = df.drug.str.lower()
    df["label"] = df.raw_label.map(normalise_label)

    # Boltz-2's affinity score is a predicted log10(IC50 in uM): lower is tighter.
    # Negate it so that, like every other column here, larger means more likely to bind.
    df["neg_affinity_score"] = -df.affinity_score

    m.record_stage("pairs_all", df)
    if exclude_known_bad:
        for target, reason in KNOWN_BAD_INPUTS.items():
            n = int((df.Target == target).sum())
            if n:
                log.warning("excluding %d %s rows: %s", n, target, reason)
                m.notes.append(f"excluded {n} {target} rows: {reason}")
                df = df[df.Target != target]
    labelled = df[df.label.notna()].copy()
    m.record_stage("pairs_labelled", labelled)
    return df


def dimer_panel(path: Path, m: Manifest, master: pd.DataFrame | None = None) -> pd.DataFrame:
    """The interface-metric experiment, consolidated from three sheets and repaired.

    ``PDBdimerresult.xlsx`` splits one experiment across sheets that grew as it ran:
    ``Raw`` (four targets), ``MAOA`` and ``Transporters``. A fourth sheet,
    ``-PDE>5``, is ``Raw`` again with a filter applied and seven blank rows, so it
    is not read -- reading it would double-count 51 of the 107 rows.

    Two defects are repaired here:

    * ``Raw`` has a column shift -- 15 SLC6A2 rows carry a SMILES string in the
      ``Experiment result`` column, a paste that landed one column left. The label
      is recovered by joining back to the master table rather than retyped.
    * The ``MAOA`` sheet has no label column at all; those labels are recovered the
      same way.
    """
    SHEETS = {"Raw": "Experiment result", "MAOA": None, "Transporters": "Unnamed: 5"}
    frames = []
    xl = pd.ExcelFile(m.record_input(path))
    for sheet, label_col in SHEETS.items():
        d = xl.parse(sheet)
        d.columns = [str(c).strip() for c in d.columns]
        keep = ["Target", "Drug", "Ligand_ipTM", "Interface_PDE", "Interface_pLDDT"]
        out = d[keep].copy()
        out["raw_label"] = d[label_col] if label_col and label_col in d else None
        out["sheet"] = sheet
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Target", "Ligand_ipTM"])
    df["Target"] = df.Target.astype(str).str.strip().str.upper()

    # Drug cells on some sheets are prefixed with the PDB id, e.g. "8HFE_Atropine".
    # Strip only a genuine 4-character PDB code, so that names which legitimately
    # contain an underscore ("Fenfluramine_active_compound") survive intact.
    df["drug"] = df.Drug.astype(str).str.strip().str.replace(
        r"^[0-9][A-Za-z0-9]{3}_", "", regex=True)
    df["drug_key"] = df.drug.str.lower()
    df["label"] = df.raw_label.map(normalise_label)

    unlabelled = df.label.isna()
    if master is not None and unlabelled.any():
        key = (master.drop_duplicates(["Target", "drug_key"])
                     .set_index(["Target", "drug_key"]).label)
        idx = pd.MultiIndex.from_frame(df.loc[unlabelled, ["Target", "drug_key"]])
        df.loc[unlabelled, "label"] = key.reindex(idx).to_numpy()

    shifted = int((df.raw_label.notna() & df.raw_label.map(normalise_label).isna()).sum())
    m.notes.append(f"dimer sheets: {shifted} rows carried a SMILES string in the label "
                   f"column and {int(unlabelled.sum())} rows had no label column; "
                   f"labels recovered by joining to the master table "
                   f"({int(df.label.isna().sum())} still unresolved)")
    return m.record_stage("dimer_panel", df)


def smiles_disagreements(si: Path, pubchem_table: Path, m: Manifest) -> pd.DataFrame:
    """Compare the PubChem-derived structures against the published S1 SMILES.

    Both sides are canonicalised to an InChIKey with RDKit, so the comparison is
    chemical rather than textual. Classification:

    ``identical``            same InChIKey
    ``tautomer_or_stereo``   same molecular formula, different InChIKey
    ``different_compound``   different formula -- a salt, ester or prodrug variant
    ``name_case_mismatch``   only a case difference separated the two names
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")

    s1 = pd.read_excel(m.record_input(si), sheet_name="S1 Drugs used in study")
    s1.columns = [c.strip() for c in s1.columns]
    published = {str(k).strip().lower(): str(v).strip()
                 for k, v in zip(s1["Drug name"], s1["SMILES"])}
    exact = {str(k).strip() for k in s1["Drug name"]}

    pc = pd.read_excel(m.record_input(pubchem_table), header=None, skiprows=1)
    pc = pc[[2, 4]].rename(columns={2: "drug", 4: "smiles_pubchem"}).dropna()
    pc["drug"] = pc.drug.astype(str).str.strip()
    pc = pc.drop_duplicates("drug")

    def describe(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        return Chem.MolToInchiKey(mol), CalcMolFormula(mol)

    rows = []
    for _, r in pc.iterrows():
        name, resolved = r.drug, r.drug.lower()
        how = "exact"
        if resolved not in published:                       # "Testosterone_Propionate"
            resolved, how = r.drug.split("_")[0].lower(), "suffix_stripped"
        if resolved not in published:
            rows.append({"drug": name, "status": "absent_from_S1"})
            continue
        if name not in exact and how == "exact":
            how = "case_insensitive"

        ik_pc, f_pc = describe(r.smiles_pubchem)
        ik_s1, f_s1 = describe(published[resolved])
        if ik_pc is None or ik_s1 is None:
            status = "unparseable"
        elif ik_pc == ik_s1:
            status = "identical"
        elif f_pc == f_s1:
            status = "tautomer_or_stereo"
        else:
            status = "different_compound"
        rows.append({"drug": name, "matched_S1_entry": resolved, "name_match": how,
                     "status": status, "formula_pubchem": f_pc, "formula_S1": f_s1,
                     "smiles_pubchem": r.smiles_pubchem, "smiles_S1": published[resolved]})
    out = pd.DataFrame(rows)
    return m.record_stage("smiles_comparison", out)


def build(si: Path = REPO / "data/raw/NIHMS373071-supplement-3.xls",
          out: Path = REPO / "data/processed") -> Manifest:
    """Run every stage and write the outputs plus ``manifest.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = Manifest()

    tgt = safety_targets(si, m)
    uni = uniprot_map(REPO / "data/raw/uniprot_reviewed_2025-11-10.fasta.gz", m)
    drg = drugs(si, m)
    prd = predictions(REPO / "data/raw/Predictions.dat", m)
    cnf = confirmed(si, m)
    audited = pairs(REPO / "data/boltz_outputs/alltargetSHOICHET.xlsx", m,
                    exclude_known_bad=False)
    prs = audited[audited.Target != "ADRA2A"].copy()
    for target, reason in KNOWN_BAD_INPUTS.items():
        n = int((audited.Target == target).sum())
        log.warning("excluding %d %s rows from pairs.parquet: %s", n, target, reason)
        m.notes.append(f"excluded {n} {target} rows from pairs.parquet: {reason}")
    m.record_stage("pairs_primary", prs)
    dim = dimer_panel(REPO / "data/boltz_outputs/PDBdimerresult.xlsx", m, master=prs)
    smi = smiles_disagreements(si, REPO / "data/boltz_outputs/pubchem_smiles_expanded.xlsx", m)

    # Attach target class and the published structures to both pair tables.
    def annotate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(tgt[["Target", "class_1", "class_2", "gene_id"]], on="Target", how="left")
        return df.merge(drg[["drug_key", "smiles_published"]], on="drug_key", how="left")

    prs, audited = annotate(prs), annotate(audited)

    for name, df in [("safety_targets", tgt), ("uniprot_map", uni), ("drugs", drg),
                     ("predictions", prd), ("confirmed", cnf), ("pairs", prs),
                     ("dimer_panel", dim), ("smiles_comparison", smi),
                     ("pairs_audited", audited)]:
        df.to_parquet(out / f"{name}.parquet", index=False)
    m.write(out / "manifest.json")
    return m


def sea_baseline(pred_dat: Path, status_dat: Path, m: Manifest | None = None) -> pd.DataFrame:
    """SEA's own predictions with their bench outcome, one row per *prediction*.

    ``Predictions.dat`` has 4,759 rows but only 2,768 distinct ``Prediction_Id``:
    a single drug-target prediction is repeated once per ChEMBL gene entry it maps
    to. Counting rows instead of predictions inflates every total by roughly 1.7x,
    so the de-duplication happens here, once, rather than in each analysis.

    ``status`` is SEA's bench outcome: ``confirmed`` (196 predictions) and
    ``inactive`` (827) were assayed; ``known``, ``known_chembl`` and ``no data``
    were not, and carry no label.
    """
    m = m or Manifest()
    pred = pd.read_csv(m.record_input(pred_dat), sep="\t")
    status = pd.read_csv(m.record_input(status_dat), sep="\t")
    pred = pred.drop_duplicates("Prediction_Id")
    df = pred.merge(status, on="Prediction_Id", how="inner")
    df["Target"] = df.chemblgene.astype(str).str.upper().str.strip()
    df["label"] = df.status.map({"confirmed": "assay_active", "inactive": "assay_inactive"})
    df["neg_log_evalue"] = -np.log10(df["E-value"].clip(lower=1e-300))
    return m.record_stage("sea_baseline", df)
