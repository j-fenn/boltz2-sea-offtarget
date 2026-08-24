"""Figures. Every one is regenerated from `data/` by `offtarget figures`."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .metrics import SCORES, stratified_auc

# Two structural-confidence scores in warm grey, two affinity scores in blue:
# the visual grouping is the finding.
COLOURS = {"confidence": "#b08968", "iptm": "#7f5539",
           "affinity_probability": "#4c72b0", "neg_affinity_score": "#2a4d80"}


def roc_panel(df: pd.DataFrame, out: Path, label: str = "label") -> Path:
    """ROC curves for all four scores, pooled, with the within-target AUC beside each.

    Both numbers are shown deliberately. The pooled curve is what a reader expects;
    the within-target figure in the legend is the one that means anything, and the
    gap between them for the confidence scores is the point of the figure.
    """
    y = (df[label] == "assay_active").astype(int)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for col, name in SCORES.items():
        fpr, tpr, _ = roc_curve(y, df[col])
        strat, _ = stratified_auc(df, col, label=label)
        ax.plot(fpr, tpr, lw=2, color=COLOURS[col],
                label=f"{name}\n     pooled {roc_auc_score(y, df[col]):.3f}  ·  within-target {strat:.3f}")
    ax.plot([0, 1], [0, 1], ls=":", c="0.5", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Separating confirmed binders from confirmed non-binders\n"
                 f"n = {len(df)} pairs, {int(y.sum())} assay-active", fontsize=11)
    ax.legend(fontsize=7.2, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def within_target_bars(df: pd.DataFrame, out: Path, label: str = "label") -> Path:
    """Within-target AUC per score with a bootstrap interval, and the chance line.

    The interval resamples targets, not pairs. It is wide, and it should be: the
    whole table rests on 34 proteins.
    """
    from .metrics import bootstrap_auc_ci
    rows = []
    for col, name in SCORES.items():
        v = stratified_auc(df, col, label=label)[0]
        lo, hi = bootstrap_auc_ci(df, col, label=label)
        rows.append((name.split(" (")[0], v, lo, hi, COLOURS[col]))

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    err = np.array([[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]])
    ax.barh(names, vals, color=[r[4] for r in rows], height=0.6,
            xerr=err, error_kw=dict(ecolor="0.25", capsize=3, lw=1.1))
    ax.axvline(0.5, color="0.25", ls="--", lw=1.2, zorder=0)
    for i, r in enumerate(rows):
        ax.text(r[3] + 0.008, i, f"{r[1]:.3f}   [{r[2]:.2f}, {r[3]:.2f}]",
                va="center", fontsize=8.5)
    ax.set_xlim(0.35, 0.92)
    ax.set_xlabel("Within-target AUC   (bars: 95% bootstrap interval over targets)")
    ax.text(0.505, -0.42, "chance", fontsize=8.5, color="0.25", va="center")
    ax.set_title("Structural confidence is at chance; the affinity head is not", fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def calibration_scatter(paired: pd.DataFrame, out: Path) -> Path:
    """Predicted versus measured potency, both arms, on a consistent back-transform."""
    d = paired.dropna(subset=["actual_uM", "pdb_affinity", "uni_affinity"])
    d = d[d.actual_uM > 0]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lo, hi = -4, 3
    ax.plot([lo, hi], [lo, hi], ls=":", c="0.5", lw=1, zorder=0)
    for c in (1, -1):
        ax.plot([lo, hi], [lo + c, hi + c], ls=":", c="0.8", lw=1, zorder=0)
    ax.scatter(np.log10(d.actual_uM), d.pdb_affinity, s=16, alpha=.7,
               color="#b08968", label="PDB arm")
    ax.scatter(np.log10(d.actual_uM), d.uni_affinity, s=16, alpha=.7,
               color="#4c72b0", label="UniProt arm")
    ax.set_xlabel("Measured activity, log₁₀ µM  (Lounkine 2012)")
    ax.set_ylabel("Boltz-2 predicted affinity, log₁₀ µM")
    ax.set_title(f"Calibration, n = {len(d)}\nmean absolute error ≈ 1.23 log units (~17-fold)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def claim_background(df: pd.DataFrame, claims: list[tuple[str, str]], out: Path) -> Path:
    """Where the four nominated pairs sit in the ipTM distribution of known non-binders.

    The figure exists to answer one question: does the score that nominated them
    put them anywhere unusual? The histogram is every confirmed non-binder.
    """
    inact = df[df.label == "assay_inactive"]
    act = df[df.label == "assay_active"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bins = np.linspace(0.80, 1.0, 41)
    ax.hist(inact.iptm.clip(lower=0.80), bins=bins, color="0.80",
            label=f"confirmed non-binders (n={len(inact)})")
    ax.hist(act.iptm.clip(lower=0.80), bins=bins, histtype="step", lw=1.8,
            color="#2a4d80", label=f"confirmed binders (n={len(act)})")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.30)

    n97 = int((inact.iptm > 0.97).sum())
    ax.axvline(0.97, color="0.2", ls="--", lw=1.2)
    ax.text(0.9715, top * 0.62, f"ipTM > 0.97\ncleared by {n97} of {len(inact)}\nknown non-binders",
            fontsize=8.5, color="0.2", ha="left", va="top")

    for i, (tgt, drug) in enumerate(claims):
        row = df[(df.Target == tgt) & df.drug.str.contains(drug, case=False, na=False)]
        if row.empty:
            continue
        v = float(row.iloc[0].iptm)
        pct = 100 * (inact.iptm < v).mean()
        y = top * (1.25 - 0.085 * i)
        ax.vlines(v, 0, y, color="#c1443f", lw=1.3)
        ax.plot([v], [y], marker="o", ms=3.5, color="#c1443f")
        ax.annotate(f"{tgt}–{drug}   ipTM {v:.3f},  {pct:.0f}th percentile of non-binders",
                    xy=(v, y), xytext=(-8, 0), textcoords="offset points",
                    fontsize=8.2, color="#c1443f", ha="right", va="center")

    ax.set_xlim(0.80, 1.005)
    ax.set_xlabel("ipTM  (values below 0.80 pooled into the first bin)")
    ax.set_ylabel("pairs")
    ax.set_title("The four nominated pairs, against the background this metric produces",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def dimer_pass_rates(dimer: pd.DataFrame, out: Path) -> Path:
    """Pass rate for the "crystal-quality" interface bar, by target."""
    d = dimer.dropna(subset=["Ligand_ipTM", "Interface_PDE"]).copy()
    d["passes"] = (d.Interface_PDE < 1) & (d.Ligand_ipTM > 0.9)
    g = d.groupby("Target").agg(n=("Drug", "size"), passed=("passes", "sum"),
                                median_pde=("Interface_PDE", "median"))
    g["rate"] = g.passed / g.n
    g = g.sort_values("rate")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.barh(g.index, g.rate, color="#4c72b0", height=0.6)
    for i, (t, r) in enumerate(g.iterrows()):
        ax.text(r.rate + 0.012, i, f"{int(r.passed)}/{int(r.n)}   median iPDE {r.median_pde:.2f}",
                va="center", fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("fraction clearing iPDE < 1 and ipTM > 0.9")
    ax.set_title("The interface-quality bar is a property of the target, not the drug",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def parse_model_cif(path: Path) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Pull the CA trace, its per-residue pLDDT, and the ligand atoms out of a Boltz-2 model.

    Written against the ModelCIF that Boltz-2 emits, which is machine-written and
    column-stable, so a whitespace split is exact. A general mmCIF parser would be
    the wrong dependency for reading four files this repository ships.

    Columns 10-12 are x/y/z; the second-to-last column is the per-atom pLDDT.
    """
    ca, plddt, lig = [], [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM"):
            f = line.split()
            if f[3] == "CA":
                ca.append([float(f[10]), float(f[11]), float(f[12])])
                plddt.append(float(f[-2]))
        elif line.startswith("HETATM"):
            f = line.split()
            lig.append([float(f[10]), float(f[11]), float(f[12])])
    return np.array(ca), np.array(plddt), np.array(lig)


def structure_hero(cif: Path, out: Path, title: str, subtitle: str) -> Path:
    """Cα trace coloured by pLDDT, with the ligand drawn where the model puts it.

    The caption is doing as much work as the picture. This is a prediction, and the
    rest of this repository is about how little a confident-looking prediction tells
    you on this dataset.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    ca, plddt, lig = parse_model_cif(cif)
    fig = plt.figure(figsize=(6.8, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([-0.02, 0.10, 0.88, 0.84])

    seg = np.stack([ca[:-1], ca[1:]], axis=1)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(50, 100)
    for s, v in zip(seg, plddt[:-1]):
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color=cmap(norm(v)), lw=1.5)
    if len(lig):
        ax.scatter(lig[:, 0], lig[:, 1], lig[:, 2], s=42, color="#c1443f",
                   depthshade=False, edgecolors="none")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("pLDDT", fontsize=9)

    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    fig.text(0.5, 0.015, subtitle, ha="center", va="bottom", fontsize=8.2, color="0.3")
    # A cube around the fold, trimmed so the one long helix does not shrink everything.
    lim = np.percentile(np.abs(ca - ca.mean(0)), 99.5)
    c = ca.mean(0)
    for setter, i in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        setter(c[i] - lim, c[i] + lim)
    ax.view_init(elev=16, azim=42)
    ax.set_box_aspect((1, 1, 1), zoom=1.55)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
