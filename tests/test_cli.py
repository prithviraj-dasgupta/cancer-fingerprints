"""Tests for the command-line interface (no dependency on the real profile module)."""

from __future__ import annotations

import subprocess
import sys
import types

import pandas as pd
import pytest

import fingerprints
from fingerprints import cli


class FakeProfile:
    """Records calls to a fake build_sbs96_matrix; can be told to raise."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.raises: Exception | None = None

    def build_sbs96_matrix(self, maf_path, sample_col="Tumor_Sample_Barcode"):
        self.calls.append({"maf_path": maf_path, "sample_col": sample_col})
        if self.raises is not None:
            raise self.raises
        index = pd.Index([f"CH{i}" for i in range(96)], name="MutationType")
        return pd.DataFrame({"S1": range(96), "S2": [0] * 96}, index=index)


@pytest.fixture
def fake_profile(monkeypatch) -> FakeProfile:
    fake = FakeProfile()
    module = types.ModuleType("fingerprints.profile")
    module.build_sbs96_matrix = fake.build_sbs96_matrix
    monkeypatch.setitem(sys.modules, "fingerprints.profile", module)
    monkeypatch.setattr(fingerprints, "profile", module, raising=False)
    return fake


@pytest.fixture
def maf(tmp_path):
    path = tmp_path / "in.maf"
    path.write_text("")
    return path


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("profile", "fit", "explain"):
        assert command in out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert fingerprints.__version__ in capsys.readouterr().out


def test_python_m_version():
    result = subprocess.run(
        [sys.executable, "-m", "fingerprints", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert fingerprints.__version__ in result.stdout


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["fit", "explain"])
def test_stub_commands_exit_3(command, capsys):
    assert cli.main([command]) == cli.EXIT_NOT_IMPLEMENTED == 3
    assert "not implemented yet (Phase 2)" in capsys.readouterr().err


def test_profile_requires_output(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["profile", "in.maf"])
    assert exc.value.code == 2
    assert "--output" in capsys.readouterr().err


def test_profile_writes_tsv(tmp_path, maf, fake_profile):
    out = tmp_path / "nested" / "sbs96.tsv"

    assert cli.main(["profile", str(maf), "--output", str(out)]) == 0

    assert fake_profile.calls == [{"maf_path": maf, "sample_col": "Tumor_Sample_Barcode"}]
    written = pd.read_csv(out, sep="\t", index_col=0)
    assert written.shape == (96, 2)
    assert list(written.columns) == ["S1", "S2"]
    assert written.index.name == "MutationType"


def test_profile_passes_sample_col(tmp_path, maf, fake_profile):
    out = tmp_path / "sbs96.tsv"
    code = cli.main(["profile", str(maf), "-o", str(out), "--sample-col", "sample_id"])
    assert code == 0
    assert out.is_file()
    assert fake_profile.calls[0]["sample_col"] == "sample_id"


def test_profile_missing_maf_is_clean_error(tmp_path, fake_profile, capsys):
    missing = tmp_path / "nope.maf"
    out = tmp_path / "sbs96.tsv"

    assert cli.main(["profile", str(missing), "-o", str(out)]) == cli.EXIT_ERROR == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert str(missing) in err
    assert "Traceback" not in err
    assert not out.exists()
    assert fake_profile.calls == []


@pytest.mark.parametrize(
    "exc",
    [ValueError("MAF has no CONTEXT column"), FileNotFoundError("reference not found")],
)
def test_profile_build_errors_are_clean(tmp_path, maf, fake_profile, capsys, exc):
    fake_profile.raises = exc
    out = tmp_path / "sbs96.tsv"

    assert cli.main(["profile", str(maf), "-o", str(out)]) == 1

    err = capsys.readouterr().err
    assert err == f"error: {exc}\n"
    assert not out.exists()
