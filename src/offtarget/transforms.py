"""Converting Boltz-2's affinity output into a concentration, and one way to get it wrong.

Boltz-2's affinity head emits a predicted log10(IC50) with the concentration in
micromolar. Recovering the concentration is ``10**x``. The master workbook does
that for the PDB arm -- and uses ``EXP(x)`` for the UniProt arm, in the adjacent
column, on the same quantity. The two arms were then compared and the UniProt arm
declared more accurate.

It is more accurate the way a shorter ruler makes a room smaller. Since
``e**x == 10**(log10(e) * x)``, the exponential path returns a number whose
log10 is ``0.4343 * x``: every prediction is pulled toward 1 uM, and predictions
are wrong mostly by being far from 1 uM. On this dataset the reported mean
absolute log10 error drops from 1.244 to 0.898 without the model changing at all,
and the "UniProt is more accurate" conclusion evaporates: 1.244 versus 1.232 for
the PDB arm, a difference of 0.012 log units in the other direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOG10_E = float(np.log10(np.e))   # 0.4342944819...


def affinity_to_uM(affinity_score: "np.ndarray | pd.Series | float"):
    """Correct back-transform: predicted log10(IC50 [uM]) -> IC50 in uM."""
    return np.power(10.0, affinity_score)


def workbook_legacy_uniprot(affinity_score: "np.ndarray | pd.Series | float"):
    """Reproduce the defect. Do not use for anything but demonstrating it.

    This is the ``=EXP(M2)`` in column O of ``boltz2_result.xlsx``. It is kept so
    the reported figure of 0.898 can be regenerated from code rather than taken on
    trust, and it is called from ``notebooks/04`` and nowhere else.
    """
    return np.exp(affinity_score)


def log_error(predicted_uM, actual_uM):
    """Absolute error in log10 concentration units. 1.0 == off by ten-fold."""
    return np.abs(np.log10(predicted_uM) - np.log10(actual_uM))


def calibration_table(df: pd.DataFrame, actual: str = "actual_uM",
                      pdb: str = "pdb_affinity", uni: str = "uni_affinity") -> pd.DataFrame:
    """Mean absolute log10 error for each arm, under each back-transform.

    Returns one row per (arm, transform) so the legacy number sits next to the
    corrected one and the comparison is visible rather than asserted.
    """
    d = df.dropna(subset=[actual, pdb, uni])
    d = d[d[actual] > 0]
    a = d[actual]
    rows = [
        {"arm": "UniProt", "transform": "EXP(x) - as computed in the workbook",
         "mean_abs_log10_error": log_error(workbook_legacy_uniprot(d[uni]), a).mean()},
        {"arm": "UniProt", "transform": "10**x - consistent",
         "mean_abs_log10_error": log_error(affinity_to_uM(d[uni]), a).mean()},
        {"arm": "PDB", "transform": "10**x - consistent",
         "mean_abs_log10_error": log_error(affinity_to_uM(d[pdb]), a).mean()},
    ]
    out = pd.DataFrame(rows)
    out["n"] = len(d)
    out["fold_error"] = 10 ** out.mean_abs_log10_error
    return out


def load_paired_affinities(path) -> pd.DataFrame:
    """Pull the side-by-side PDB / UniProt / measured-activity block out of the workbook.

    ``boltz2_result.xlsx`` is three tables pasted side by side in one sheet -- the
    PDB run in columns B:H, the UniProt run in J:O, and the Lounkine assay result
    in Q:AA -- aligned by row position with no key. Reading it by position is what
    the file supports; that fragility is itself an argument for building tables in
    code, and it is why this function exists instead of a ``read_excel`` call at
    the call site.
    """
    tp = pd.read_excel(path, sheet_name="True Positive")
    tp.columns = [str(c).strip() for c in tp.columns]
    pdb_block = tp.iloc[:, 1:8]
    uni_block = tp.iloc[:, 9:15]
    sea_block = tp.iloc[:, 16:27]
    num = lambda s: pd.to_numeric(s, errors="coerce")
    return pd.DataFrame({
        "Target": pdb_block.iloc[:, 0].astype(str).str.strip().str.upper(),
        "drug": pdb_block.iloc[:, 1].astype(str).str.strip(),
        "pdb_affinity": num(pdb_block.iloc[:, 3]),
        "pdb_confidence": num(pdb_block.iloc[:, 2]),
        "uni_affinity": num(uni_block.iloc[:, 3]),
        "uni_confidence": num(uni_block.iloc[:, 2]),
        "actual_uM": num(sea_block["Confirmed activity [uM]"]),
    })
