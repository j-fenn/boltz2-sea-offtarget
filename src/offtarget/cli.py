"""Command line entry point: ``offtarget <command>``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _build(a) -> int:
    from .pipeline import build
    m = build(si=a.si, out=a.out)
    print(f"\nwrote {a.out}")
    for stage, n in m.stages.items():
        print(f"  {stage:20s} {n:6d}")
    for note in m.notes:
        print(f"  note: {note}")
    return 0


def _audit_inputs(a) -> int:
    from .constructs import audit_inputs
    df = audit_inputs(a.root)
    print(f"{len(df)} inputs")
    print(f"  carrying expression-construct scaffold: {int(df.has_tags.sum())} "
          f"({df.has_tags.mean():.0%})")
    print(f"  modelled as homodimer: {int((df.n_chains == 2).sum())}   "
          f"monomer: {int((df.n_chains == 1).sum())}")
    print()
    print(df[df.has_tags].groupby(["directory", "tags"]).size().to_string())
    return 0


def _metrics(a) -> int:
    import pandas as pd
    from .metrics import auc_table
    df = pd.read_parquet(a.pairs)
    print(auc_table(df[df.label.notna()]).round(3).to_string(index=False))
    return 0


def _figures(a) -> int:
    import pandas as pd
    from . import figures as F
    from .transforms import load_paired_affinities
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # The audited set (ADRA2A included) is what every quoted number refers to;
    # notebook 03 repeats the whole table with those 20 rows removed.
    pairs = pd.read_parquet(REPO / "data/processed/pairs_audited.parquet")
    labelled = pairs[pairs.label.notna()]
    dimer = pd.read_parquet(REPO / "data/processed/dimer_panel.parquet")
    paired = load_paired_affinities(REPO / "data/boltz_outputs/boltz2_result.xlsx")
    made = [
        F.roc_panel(labelled, out / "fig1_roc_confidence_vs_affinity.png"),
        F.within_target_bars(labelled, out / "fig2_within_target_auc.png"),
        F.calibration_scatter(paired, out / "fig3_calibration.png"),
        F.dimer_pass_rates(dimer, out / "fig4_dimer_pass_rates.png"),
        F.claim_background(labelled, [("MAOA", "Fenoterol"), ("SLC6A4", "Atropine"),
                                      ("SLC6A4", "Ramipril"), ("SLC6A4", "Benzquinamide")],
                           out / "fig5_claim_background.png"),
        F.structure_hero(
            REPO / "assets/structures/MAOA_2Z5X_Fenoterol_model_0.cif",
            out / "fig0_hero_structure.png",
            "Boltz-2 model of MAO-A with fenoterol",
            "A prediction, not a validated complex. Fenoterol was recorded inactive against\n"
            "MAO-A in the 2012 assay; on this panel it is the median confirmed non-binder by\n"
            "ipTM, and ipTM does not separate binders from non-binders here (AUC 0.486)."),
    ]
    for p in made:
        print("wrote", p.relative_to(REPO))
    return 0


def _tables(a) -> int:
    """Write the tables the README quotes, so every number in it has a file behind it."""
    import pandas as pd
    from .metrics import auc_table, bootstrap_auc_ci, per_target_auc
    from .transforms import calibration_table, load_paired_affinities

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    audited = pd.read_parquet(REPO / "data/processed/pairs_audited.parquet")
    audited = audited[audited.label.notna()]
    primary = pd.read_parquet(REPO / "data/processed/pairs.parquet")
    primary = primary[primary.label.notna()]

    auc = auc_table(audited)
    ci = [bootstrap_auc_ci(audited, c) for c in ["confidence", "iptm",
                                                 "affinity_probability", "neg_affinity_score"]]
    auc["ci_low"] = [c[0] for c in ci]
    auc["ci_high"] = [c[1] for c in ci]
    auc["within_target_auc_adra2a_dropped"] = auc_table(primary).within_target_auc

    dimer = pd.read_parquet(REPO / "data/processed/dimer_panel.parquet")
    dimer = dimer.dropna(subset=["Ligand_ipTM", "Interface_PDE"]).copy()
    dimer["clears_bar"] = (dimer.Interface_PDE < 1) & (dimer.Ligand_ipTM > 0.9)
    bar = dimer.groupby("Target").agg(n=("drug", "size"), passed=("clears_bar", "sum"),
                                      median_interface_PDE=("Interface_PDE", "median"))
    bar["pass_rate"] = (bar.passed / bar.n).round(3)

    written = {
        "table1_auc.csv": auc.round(4),
        "table2_calibration.csv": calibration_table(
            load_paired_affinities(REPO / "data/boltz_outputs/boltz2_result.xlsx")).round(4),
        "table3_per_target_auc.csv": per_target_auc(audited, "iptm").merge(
            per_target_auc(audited, "neg_affinity_score"), on="Target",
            suffixes=("_iptm", "_affinity")).round(4),
        "table4_interface_bar_by_target.csv": bar,
        "table5_smiles_comparison.csv": pd.read_parquet(
            REPO / "data/processed/smiles_comparison.parquet"),
    }
    for name, df in written.items():
        df.to_csv(out / name, index=(name == "table4_interface_bar_by_target.csv"))
        print("wrote", (out / name).relative_to(REPO))
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser(prog="offtarget", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="rebuild every analysis table from the published SI")
    b.add_argument("--si", type=Path, default=REPO / "data/raw/NIHMS373071-supplement-3.xls")
    b.add_argument("--out", type=Path, default=REPO / "data/processed")
    b.set_defaults(fn=_build)

    ai = sub.add_parser("audit-inputs", help="tag / oligomer audit over the model inputs")
    ai.add_argument("--root", type=Path, default=REPO / "inputs/generated")
    ai.set_defaults(fn=_audit_inputs)

    mt = sub.add_parser("metrics", help="pooled and within-target AUC for every score")
    mt.add_argument("--pairs", type=Path, default=REPO / "data/processed/pairs.parquet")
    mt.set_defaults(fn=_metrics)

    fg = sub.add_parser("figures", help="regenerate every figure")
    fg.add_argument("--out", type=Path, default=REPO / "results/figures")
    fg.set_defaults(fn=_figures)

    tb = sub.add_parser("tables", help="write the tables the README quotes")
    tb.add_argument("--out", type=Path, default=REPO / "results/tables")
    tb.set_defaults(fn=_tables)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
