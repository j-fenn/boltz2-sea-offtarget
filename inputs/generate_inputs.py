#!/usr/bin/env python3
"""Generate every Boltz-2 input from `targets.yaml` and the drug table.

Replaces ten near-identical generator scripts. Each of those carried one protein
sequence pasted into a string literal and one YAML template pasted into another,
so the per-target modelling decisions -- which construct, how many chains, which
tags -- existed only as a difference between copies of a script. Nobody could
review them because there was nothing to review.

Here the decisions live in `targets.yaml` and the code is the same for every
target. Running this reproduces all 158 committed inputs byte for byte, which is
the point: the configuration is a faithful description of what was actually run,
including the parts that were wrong.

    python inputs/generate_inputs.py --out inputs/generated
    python inputs/generate_inputs.py --check inputs/generated   # byte-for-byte diff
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent


def render(arm: dict, smiles: str) -> str:
    """Render one input file.

    Three template shapes appear in the committed inputs -- a bare monomer, a
    dimer with a `# Source PDB:` comment and quoted `A_<pdbid>` chain ids, and a
    dimer with unquoted `A`/`B` ids. They differ only in the header comment and
    whether chain ids are quoted, both of which are config here.
    """
    q = '"{}"'.format if arm["quote_chain_ids"] else "{}".format
    parts = ["version: 1\n"]
    if arm.get("header_comment"):
        parts.append(arm["header_comment"] + "\n")
    parts.append("sequences:\n")
    for cid in arm["chain_ids"]:
        parts.append("  - protein:\n")
        parts.append(f"      id: {q(cid)}\n")
        parts.append(f'      sequence: "{arm["sequence"]}"\n')
    parts.append("  - ligand:\n      id: LIG\n")
    parts.append(f'      smiles: "{smiles}"\n')
    parts.append("      copies: 1\n")
    return "".join(parts)


def load_config(path: Path = REPO / "inputs" / "targets.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def load_drugs(cfg: dict) -> pd.DataFrame:
    t = cfg["drug_table"]
    df = pd.read_excel(REPO / t["path"], sheet_name=t["sheet"])
    df.columns = df.columns.str.strip()
    df = df.rename(columns={t["name_column"]: "drug",
                            t["smiles_column"]: "smiles",
                            t["target_column"]: "target"})
    df["target"] = df["target"].astype(str).str.strip().str.upper()
    df["drug"] = df["drug"].astype(str).str.strip()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    return df[~df.drug.isin(["nan", ""]) & ~df.smiles.isin(["nan", ""])]


def generate(out_root: Path, cfg: dict | None = None) -> list[Path]:
    cfg = cfg or load_config()
    drugs = load_drugs(cfg)
    sanitise = re.compile(cfg["filename_sanitise_pattern"])
    written: list[Path] = []

    for target, tcfg in cfg["targets"].items():
        subset = drugs[drugs.target == target.upper()]
        for arm_name, arm in tcfg["arms"].items():
            if arm.get("strip_tags"):
                raise NotImplementedError(
                    f"{target}/{arm_name}: strip_tags is declared but these inputs were "
                    "run with the scaffold in place; stripping would not reproduce them. "
                    "Use offtarget.constructs.strip_construct_tags for a new run."
                )
            d = out_root / arm["output_dir"]
            d.mkdir(parents=True, exist_ok=True)
            for _, row in subset.iterrows():
                fname = arm["filename_pattern"].format(drug=sanitise.sub("_", row.drug))
                p = d / fname
                p.write_text(render(arm, row.smiles))
                written.append(p)
    return written


def check(reference_root: Path, cfg: dict | None = None) -> tuple[int, list[str]]:
    """Regenerate in memory and compare against `reference_root` byte for byte."""
    cfg = cfg or load_config()
    drugs = load_drugs(cfg)
    sanitise = re.compile(cfg["filename_sanitise_pattern"])
    ok, problems = 0, []

    expected: set[Path] = set()
    for target, tcfg in cfg["targets"].items():
        subset = drugs[drugs.target == target.upper()]
        for arm_name, arm in tcfg["arms"].items():
            for _, row in subset.iterrows():
                fname = arm["filename_pattern"].format(drug=sanitise.sub("_", row.drug))
                p = reference_root / arm["output_dir"] / fname
                expected.add(p)
                if not p.exists():
                    problems.append(f"missing: {p.relative_to(reference_root)}")
                    continue
                if p.read_text() == render(arm, row.smiles):
                    ok += 1
                else:
                    problems.append(f"differs: {p.relative_to(reference_root)}")

    for p in sorted(reference_root.rglob("*.yaml")):
        if p not in expected:
            problems.append(f"unexpected: {p.relative_to(reference_root)}")
    return ok, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "inputs" / "generated")
    ap.add_argument("--check", type=Path, metavar="DIR",
                    help="compare against an existing tree instead of writing")
    a = ap.parse_args()

    if a.check:
        ok, problems = check(a.check)
        print(f"{ok} inputs reproduce byte for byte; {len(problems)} problems")
        for p in problems[:20]:
            print("  " + p)
        return 1 if problems else 0

    written = generate(a.out)
    print(f"wrote {len(written)} inputs to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
