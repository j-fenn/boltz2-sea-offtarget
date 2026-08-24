"""The back-transform defect must stay reproducible, and stay quarantined."""
import numpy as np
import pytest

from offtarget.transforms import (LOG10_E, affinity_to_uM, calibration_table,
                                  load_paired_affinities, workbook_legacy_uniprot)


def test_exp_path_scales_log_space_by_log10_e():
    """The exact identity behind the defect: exp(x) has log10 equal to 0.4343*x.

    This is what makes the workbook's UniProt error look smaller. It is an identity,
    not an empirical ratio, so it is asserted exactly rather than within a tolerance.
    """
    x = np.linspace(-4, 3, 71)
    assert np.allclose(np.log10(workbook_legacy_uniprot(x)), LOG10_E * x)
    assert LOG10_E == pytest.approx(0.4343, abs=1e-4)


def test_correct_transform_round_trips():
    x = np.linspace(-4, 3, 71)
    assert np.allclose(np.log10(affinity_to_uM(x)), x)


def test_azatadine_spot_check(repo):
    """The two workbook columns, reproduced from the affinity outputs they came from."""
    d = load_paired_affinities(repo / "data/boltz_outputs/boltz2_result.xlsx")
    row = d[(d.Target == "ADRA1A") & d.drug.str.contains("Azatadine", case=False)].iloc[0]
    assert row.pdb_affinity == pytest.approx(-1.6217, abs=1e-3)
    assert 10 ** row.pdb_affinity == pytest.approx(0.0239, abs=1e-3)
    assert row.uni_affinity == pytest.approx(-0.7548, abs=1e-3)
    assert np.exp(row.uni_affinity) == pytest.approx(0.4702, abs=1e-3)


def test_calibration_conclusion_reverses(repo):
    """On a consistent transform the reported 'UniProt is more accurate' result goes away."""
    d = load_paired_affinities(repo / "data/boltz_outputs/boltz2_result.xlsx")
    t = calibration_table(d).set_index(["arm", "transform"]).mean_abs_log10_error

    legacy = t["UniProt", "EXP(x) - as computed in the workbook"]
    uni = t["UniProt", "10**x - consistent"]
    pdb = t["PDB", "10**x - consistent"]

    assert legacy == pytest.approx(0.898, abs=0.005)
    assert uni == pytest.approx(1.244, abs=0.005)
    assert pdb == pytest.approx(1.232, abs=0.005)
    assert legacy < pdb                      # the reported comparison
    assert pdb < uni                         # the comparison on a consistent transform
