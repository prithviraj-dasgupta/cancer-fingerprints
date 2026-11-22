"""Offline checks on the cancer-type signature subsets in config/config.yaml.

The COSMIC file is not read here (it is not in the repo). Instead, the valid COSMIC v3.4 SBS
names are hard-coded, copied from the header of COSMIC_v3.4_SBS_GRCh38_exome.txt bundled
with SigProfilerAssignment 1.1.5 (86 signatures).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

# Header of COSMIC_v3.4_SBS_GRCh38_exome.txt (SigProfilerAssignment 1.1.5), minus "Type".
_COSMIC_V34_HEADER = """
SBS1 SBS2 SBS3 SBS4 SBS5 SBS6 SBS7a SBS7b SBS7c SBS7d SBS8 SBS9 SBS10a SBS10b SBS10c
SBS10d SBS11 SBS12 SBS13 SBS14 SBS15 SBS16 SBS17a SBS17b SBS18 SBS19 SBS20 SBS21
SBS22a SBS22b SBS23 SBS24 SBS25 SBS26 SBS27 SBS28 SBS29 SBS30 SBS31 SBS32 SBS33 SBS34
SBS35 SBS36 SBS37 SBS38 SBS39 SBS40a SBS40b SBS40c SBS41 SBS42 SBS43 SBS44 SBS45 SBS46
SBS47 SBS48 SBS49 SBS50 SBS51 SBS52 SBS53 SBS54 SBS55 SBS56 SBS57 SBS58 SBS59 SBS60
SBS84 SBS85 SBS86 SBS87 SBS88 SBS89 SBS90 SBS91 SBS92 SBS93 SBS94 SBS95 SBS96 SBS97
SBS98 SBS99
"""
COSMIC_V34_SBS = frozenset(_COSMIC_V34_HEADER.split())

NAME_RE = re.compile(r"^SBS\d+[a-d]?$")
MMR_REQUIRED = {"SBS6", "SBS15", "SBS26", "SBS44"}  # the MMR set named in CLAUDE.md


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def subsets(config: dict) -> dict[str, list[str]]:
    return config["fitting"]["signature_subsets"]


def test_hardcoded_cosmic_list_has_86_names() -> None:
    assert len(COSMIC_V34_SBS) == 86


def test_every_project_has_a_nonempty_subset(config: dict, subsets: dict) -> None:
    project_ids = {p["id"] for p in config["projects"]}
    assert project_ids <= set(subsets), f"missing subsets: {project_ids - set(subsets)}"
    assert set(subsets) <= project_ids, f"unknown projects: {set(subsets) - project_ids}"
    for pid in project_ids:
        assert isinstance(subsets[pid], list) and subsets[pid], pid


def test_names_are_valid_v34_signatures(subsets: dict) -> None:
    for pid, sigs in subsets.items():
        for sig in sigs:
            assert NAME_RE.match(sig), f"{pid}: badly formed name {sig!r}"
            assert sig in COSMIC_V34_SBS, f"{pid}: {sig} is not a COSMIC v3.4 SBS name"


def test_no_duplicates(subsets: dict) -> None:
    for pid, sigs in subsets.items():
        assert len(sigs) == len(set(sigs)), f"{pid}: duplicate signatures"


def test_no_artefact_signatures(config: dict, subsets: dict) -> None:
    artefacts = set(config["fitting"]["exclude_artefacts"])
    for pid, sigs in subsets.items():
        assert not artefacts & set(sigs), f"{pid}: artefacts {artefacts & set(sigs)}"


def test_colorectal_includes_mmr_and_is_shared(config: dict, subsets: dict) -> None:
    crc = [p["id"] for p in config["projects"] if p["cancer_type"] == "colorectal"]
    assert crc, "no colorectal projects configured"
    for pid in crc:
        missing = MMR_REQUIRED - set(subsets[pid])
        assert not missing, f"{pid}: missing MMR signatures {missing}"
    assert all(subsets[pid] == subsets[crc[0]] for pid in crc), "COAD/READ subsets differ"


@pytest.mark.parametrize(
    ("project", "required"),
    [
        ("TCGA-LUAD", {"SBS4"}),
        ("TCGA-LUSC", {"SBS4"}),
        ("TCGA-SKCM", {"SBS7a", "SBS7b"}),
    ],
)
def test_validation_signatures_present(subsets: dict, project: str, required: set) -> None:
    """Milestone 4 validation depends on these; editing subsets must not drop them."""
    missing = required - set(subsets[project])
    assert not missing, f"{project}: missing validation signatures {missing}"


def test_subsets_marked_proposed(config: dict) -> None:
    assert "proposed" in config["fitting"]["signature_subsets_status"]
