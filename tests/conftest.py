"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory holding small test fixture files."""
    return FIXTURES_DIR


@pytest.fixture
def fake_reference_file(tmp_path: Path) -> Path:
    """A fake 96 x signatures reference (random profiles, columns sum to 1), COSMIC format.

    Lets app tests run in CI without the COSMIC file. It carries every signature named in the
    config's cancer-type subsets, so any subset can be fitted. Not real signatures.
    """
    import numpy as np
    import yaml

    from fingerprints.profile import SBS96_CHANNELS

    config = yaml.safe_load((Path(__file__).parents[1] / "config" / "config.yaml").read_text())
    names = sorted({s for sigs in config["fitting"]["signature_subsets"].values() for s in sigs})
    rng = np.random.default_rng(0)
    values = rng.dirichlet(np.ones(96) * 0.3, size=len(names)).T
    import pandas as pd

    frame = pd.DataFrame(values, index=SBS96_CHANNELS, columns=names)
    frame.index.name = "Type"
    path = tmp_path / "fake_cosmic.txt"
    frame.to_csv(path, sep="\t")
    return path
