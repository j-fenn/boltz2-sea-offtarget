import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "inputs"))


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def processed(repo):
    """Build the processed tables once if they are not already there."""
    out = repo / "data/processed"
    if not (out / "pairs_audited.parquet").exists():
        from offtarget.pipeline import build
        build(out=out)
    return out
