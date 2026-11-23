"""Tests for the command-line interface (no dependency on the real profile module)."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

import fingerprints
from fingerprints import cli
from fingerprints.fit import FitResult


class FakeProfile:
    """Records calls to a fake build_sbs96_matrix; can be told to raise."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.raises: Exception | None = None

    def build_sbs96_matrix(
        self, maf_path, sample_col="Tumor_Sample_Barcode", adjacent_snvs="exclude"
    ):
        self.calls.append(
            {"maf_path": maf_path, "sample_col": sample_col, "adjacent_snvs": adjacent_snvs}
        )
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


def test_explain_stub_exits_3(capsys):
    assert cli.main(["explain"]) == cli.EXIT_NOT_IMPLEMENTED == 3
    assert "not implemented yet (Phase 2)" in capsys.readouterr().err


class FakeFit:
    """Records calls to fake fit functions; can be told to raise."""

    METHOD_DESCRIPTION = "fake NNLS"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.raises: Exception | None = None

    def load_reference(self, path):
        self.calls.append({"reference": path})
        return pd.DataFrame({"SBS1": [1.0]})

    def run_cohort_fit(self, project, config, matrix, reference, min_snvs=None):
        self.calls.append({"project": project, "min_snvs": min_snvs, "shape": matrix.shape})
        if self.raises is not None:
            raise self.raises
        expo = pd.DataFrame({"SBS1": [5.0]}, index=["S1"])
        return FitResult(
            project=project,
            signatures=["SBS1"],
            exposures=expo,
            fractions=expo / 5.0,
            cosine=pd.Series([1.0]),
            qc=pd.DataFrame(),
            excluded_table=pd.DataFrame(),
            n_input=2,
            min_snvs=min_snvs or 50,
            excluded=["S2"],
            sensitivity=pd.DataFrame({"min_snvs": [50, 20], "n_excluded": [1, 0]}),
        )

    def save_results(self, result, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"exposures_{result.project}.tsv"
        path.write_text("x")
        return [path]


@pytest.fixture
def fake_fit(monkeypatch) -> FakeFit:
    fake = FakeFit()
    module = types.ModuleType("fingerprints.fit")
    module.METHOD_DESCRIPTION = fake.METHOD_DESCRIPTION
    module.load_reference = fake.load_reference
    module.run_cohort_fit = fake.run_cohort_fit
    module.save_results = fake.save_results
    monkeypatch.setitem(sys.modules, "fingerprints.fit", module)
    monkeypatch.setattr(fingerprints, "fit", module, raising=False)
    return fake


@pytest.fixture
def fit_inputs(tmp_path):
    matrix = tmp_path / "m.tsv"
    matrix.write_text("MutationType\tS1\tS2\nA[C>A]A\t5\t0\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "paths:\n  raw_signatures: sigs\nsignatures:\n  file: ref.txt\nfitting:\n  min_snvs: 50\n",
        encoding="utf-8",
    )
    return matrix, cfg


def test_fit_wiring(tmp_path, fake_fit, fit_inputs, capsys):
    matrix, cfg = fit_inputs
    out = tmp_path / "res"
    code = cli.main(
        ["fit", str(matrix), "--project", "TCGA-LUAD", "--config", str(cfg),
         "--min-snvs", "30", "--output-dir", str(out)]
    )  # fmt: skip
    assert code == 0
    assert fake_fit.calls[0] == {"reference": Path("sigs") / "ref.txt"}
    assert fake_fit.calls[1] == {"project": "TCGA-LUAD", "min_snvs": 30, "shape": (1, 2)}
    assert (out / "exposures_TCGA-LUAD.tsv").is_file()
    err = capsys.readouterr().err
    assert f"config={cfg}" in err
    assert "1 of 2 excluded" in err
    assert "<20: 0" in err


def test_fit_requires_project_and_output(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["fit", "m.tsv"])
    assert exc.value.code == 2


def test_fit_explicit_missing_config_is_error(tmp_path, fake_fit, fit_inputs, capsys):
    matrix, _ = fit_inputs
    missing = tmp_path / "nope.yaml"
    code = cli.main(
        ["fit", str(matrix), "--project", "P", "--config", str(missing), "-o", str(tmp_path / "o")]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and str(missing) in err
    assert fake_fit.calls == []


def test_fit_missing_matrix_is_error(tmp_path, fake_fit, capsys):
    code = cli.main(["fit", str(tmp_path / "x.tsv"), "--project", "P", "-o", str(tmp_path / "o")])
    assert code == 1
    assert "matrix file not found" in capsys.readouterr().err


def test_fit_value_error_is_clean(tmp_path, fake_fit, fit_inputs, capsys):
    matrix, cfg = fit_inputs
    fake_fit.raises = ValueError("no signature subset for project P")
    code = cli.main(
        ["fit", str(matrix), "--project", "P", "--config", str(cfg), "-o", str(tmp_path / "o")]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: no signature subset")
    assert "Traceback" not in err


def test_fit_has_no_reference_flag(tmp_path, fit_inputs, capsys):
    matrix, cfg = fit_inputs
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["fit", str(matrix), "--project", "P", "--config", str(cfg),
             "--reference", "x.txt", "-o", str(tmp_path / "o")]
        )  # fmt: skip
    assert exc.value.code == 2


def test_fit_help_mentions_relative_paths(capsys):
    with pytest.raises(SystemExit):
        cli.main(["fit", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "relative to the current directory" in out
    assert "--reference" not in out
