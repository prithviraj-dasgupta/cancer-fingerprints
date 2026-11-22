"""Offline tests for scripts/download_signatures.py.

Uses a tiny fake 96 x 2 matrix. Never imports SigProfilerAssignment.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from fingerprints.profile import SBS96_CHANNELS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


ds = _load_script("download_signatures")


def write_matrix(path: Path, channels=None, columns=("SBS1", "SBS7a"), values=None) -> Path:
    """Write a fake COSMIC-style matrix. Default: uniform columns summing to 1."""
    channels = list(SBS96_CHANNELS) if channels is None else list(channels)
    n = len(channels)
    lines = ["Type\t" + "\t".join(columns)]
    for i, ch in enumerate(channels):
        row = [values(i, j) if values else 1.0 / n for j in range(len(columns))]
        lines.append(ch + "\t" + "\t".join(f"{v:.10g}" for v in row))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_valid_matrix_in_any_row_order(tmp_path):
    p = write_matrix(tmp_path / "sig.txt", channels=sorted(SBS96_CHANNELS, reverse=True))
    res = ds.validate_signature_matrix(p)
    assert res.ok, res.errors
    assert (res.n_channels, res.n_signatures) == (96, 2)


def test_wrong_row_count(tmp_path):
    p = write_matrix(tmp_path / "sig.txt", channels=list(SBS96_CHANNELS)[:95])
    res = ds.validate_signature_matrix(p)
    assert not res.ok
    assert any("expected 96 rows" in e for e in res.errors)


def test_bad_channel_label(tmp_path):
    chans = list(SBS96_CHANNELS)
    chans[0] = "A[C>X]A"
    res = ds.validate_signature_matrix(write_matrix(tmp_path / "sig.txt", channels=chans))
    assert not res.ok
    assert any("missing" in e for e in res.errors)
    assert any("unexpected" in e for e in res.errors)


def test_bad_column_name(tmp_path):
    p = write_matrix(tmp_path / "sig.txt", columns=("SBS1", "Signature_2"))
    res = ds.validate_signature_matrix(p)
    assert not res.ok and any("SBS<n>" in e for e in res.errors)


def test_column_not_summing_to_one(tmp_path):
    p = write_matrix(tmp_path / "sig.txt", values=lambda i, j: 1 / 96 if j == 0 else 2 / 96)
    res = ds.validate_signature_matrix(p)
    assert not res.ok and any("SBS7a=2.0000" in e for e in res.errors)


def test_negative_values(tmp_path):
    p = write_matrix(tmp_path / "sig.txt", values=lambda i, j: -1.0 if i == 0 else 1 / 96)
    res = ds.validate_signature_matrix(p)
    assert not res.ok and any("negative" in e for e in res.errors)


def _sig_cfg(sha: str | None) -> dict:
    return {
        "source": "SigProfilerAssignment",
        "package_version": "1.1.5",
        "package_path": "data/Reference_Signatures/GRCh38",
        "cosmic_version": "3.4",
        "genome_build": "GRCh38",
        "assay": "exome",
        "file": "COSMIC_v3.4_SBS_GRCh38_exome.txt",
        "sha256": sha,
        "licence": {"summary": "academic only", "terms_url": "https://example.invalid"},
    }


def test_install_copies_and_writes_provenance(tmp_path):
    src = write_matrix(tmp_path / "src.txt")
    sha = ds.sha256_of(src)
    dest = tmp_path / "out" / "COSMIC_v3.4_SBS_GRCh38_exome.txt"
    ds.install_signatures(src, dest, _sig_cfg(sha), "1.1.5")
    assert dest.read_bytes() == src.read_bytes()
    prov = json.loads((dest.parent / (dest.name + ".provenance.json")).read_text())
    assert prov["sha256"] == sha and prov["package_version"] == "1.1.5"
    assert ds.is_up_to_date(dest, sha)
    assert not ds.is_up_to_date(dest, "0" * 64)


def test_install_rejects_sha_mismatch(tmp_path):
    src = write_matrix(tmp_path / "src.txt")
    dest = tmp_path / "out" / "x.txt"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        ds.install_signatures(src, dest, _sig_cfg("0" * 64), "1.1.5")
    assert not dest.exists()


def test_install_rejects_invalid_matrix(tmp_path):
    src = write_matrix(tmp_path / "src.txt", channels=list(SBS96_CHANNELS)[:10])
    with pytest.raises(ValueError, match="failed validation"):
        ds.install_signatures(src, tmp_path / "x.txt", _sig_cfg(None), "1.1.5")


def test_locate_missing_package_gives_install_hint():
    with pytest.raises(ds.SignatureSourceError, match="uv sync --extra signatures"):
        ds.locate_bundled_file("data", "x.txt", package="no_such_package_cf_test")


def test_config_pins_exome_v34():
    cfg = ds.load_config(ds.DEFAULT_CONFIG)["signatures"]
    assert cfg["file"] == "COSMIC_v3.4_SBS_GRCh38_exome.txt"
    assert cfg["cosmic_version"] == "3.4" and cfg["genome_build"] == "GRCh38"
    assert len(cfg["sha256"]) == 64


def test_main_check_without_file(tmp_path, capsys):
    assert ds.main(["--check", "--out", str(tmp_path)]) == 1
    assert "not found" in capsys.readouterr().err


def test_sha256_pin_is_required(tmp_path):
    src = write_matrix(tmp_path / "src.txt")
    for bad in (None, "", "abc"):
        with pytest.raises(ValueError, match="signatures.sha256"):
            ds.install_signatures(src, tmp_path / "x.txt", _sig_cfg(bad), "1.1.5")
    assert not (tmp_path / "x.txt").exists()


def _write_config(tmp_path: Path, sha: str | None) -> Path:
    import yaml

    cfg = {"paths": {"raw_signatures": "sigs"}, "signatures": _sig_cfg(sha)}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_main_errors_without_sha_pin(tmp_path, capsys):
    cfg = _write_config(tmp_path, None)
    assert ds.main(["--config", str(cfg), "--out", str(tmp_path)]) == 1
    assert "signatures.sha256" in capsys.readouterr().err


def test_check_requires_provenance(tmp_path, capsys):
    src = write_matrix(tmp_path / "src.txt")
    sha = ds.sha256_of(src)
    cfg = _write_config(tmp_path, sha)
    dest = tmp_path / "sigs" / "COSMIC_v3.4_SBS_GRCh38_exome.txt"
    ds.install_signatures(src, dest, _sig_cfg(sha), "1.1.5")
    assert ds.main(["--config", str(cfg), "--out", str(tmp_path), "--check"]) == 0
    ds.provenance_path(dest).unlink()
    assert ds.main(["--config", str(cfg), "--out", str(tmp_path), "--check"]) == 1
    assert "provenance.json missing" in capsys.readouterr().err
    assert not ds.is_up_to_date(dest, sha)
