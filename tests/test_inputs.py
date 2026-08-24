"""`targets.yaml` must be a faithful description of the inputs that were run."""
import re

import pytest
import yaml

from generate_inputs import check, generate, load_config, render


def test_config_reproduces_every_input_byte_for_byte(repo):
    """The whole point of the rewrite.

    Ten generator scripts each carried one sequence and one template in a string
    literal. If the config cannot rebuild what they produced, exactly, then it is a
    description of something other than what was run.
    """
    ok, problems = check(repo / "inputs/generated")
    assert problems == []
    assert ok == 158


def test_regeneration_is_deterministic(repo, tmp_path):
    generate(tmp_path)
    first = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.yaml")}
    generate(tmp_path)
    second = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.yaml")}
    assert first == second
    assert len(first) == 158


def test_no_input_carries_scaffold_unless_declared(repo):
    """A tag may be present, but only where the config says so and says why.

    This is the guard that stops a future edit from quietly reintroducing
    purification scaffold: adding a tagged sequence without recording it in the
    audit block fails here.
    """
    from offtarget.constructs import find_construct_tags
    cfg = load_config()
    for target, tcfg in cfg["targets"].items():
        for arm_name, arm in tcfg["arms"].items():
            found = sorted({t.name for t in find_construct_tags(arm["sequence"])})
            declared = sorted(arm["audit"]["construct_tags"])
            assert found == declared, f"{target}/{arm_name}: {found} != declared {declared}"
            if found:
                assert arm["rationale"], f"{target}/{arm_name} carries {found} with no rationale"


def test_every_arm_declares_its_oligomeric_state(repo):
    cfg = load_config()
    for target, tcfg in cfg["targets"].items():
        for arm_name, arm in tcfg["arms"].items():
            assert arm["oligomeric_state"] in {"monomer", "homodimer"}
            assert len(arm["chain_ids"]) == (2 if arm["oligomeric_state"] == "homodimer" else 1)


def test_render_is_pure(repo):
    cfg = load_config()
    arm = cfg["targets"]["MAOA"]["arms"]["pdb"]
    out = render(arm, "CC(=O)NC1=CC=C(C=C1)O")
    assert out.startswith("version: 1\n")
    assert out.endswith("      copies: 1\n")
    assert out.count("- protein:") == 1
