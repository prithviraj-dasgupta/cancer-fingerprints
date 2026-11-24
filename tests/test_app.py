"""Smoke tests for the Streamlit app via streamlit.testing.v1.AppTest.

Skipped when the optional `app` extra (streamlit, plotly) is not installed, so CI without it
still passes. A fake reference (see conftest) lets them run without the COSMIC file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="needs `uv sync --extra app`")
pytest.importorskip("plotly", reason="needs `uv sync --extra app`")
from streamlit.testing.v1 import AppTest  # noqa: E402

from fingerprints import appcore  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _run(monkeypatch, reference: Path) -> AppTest:
    monkeypatch.setenv(appcore.REFERENCE_ENV, str(reference))
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_missing_reference_shows_setup_command(monkeypatch, tmp_path):
    at = _run(monkeypatch, tmp_path / "absent.txt")
    assert not at.exception
    assert any("download_signatures.py" in e.value for e in at.error)
    assert any("research and education only" in i.value.lower() for i in at.info)


def test_no_upload_asks_for_a_file(monkeypatch, fake_reference_file):
    at = _run(monkeypatch, fake_reference_file)
    assert not at.exception
    assert any("Upload a MAF" in i.value for i in at.info)
    assert not at.error


def test_example_runs_end_to_end(monkeypatch, fake_reference_file):
    at = _run(monkeypatch, fake_reference_file)
    at.checkbox(key="use_example").check().run()
    assert not at.exception
    assert not at.error
    assert at.selectbox(key="cancer_type").value == "melanoma"
    assert any("SYNTHETIC-0001" in s.value for s in at.subheader)


def test_other_cancer_type_selectable(monkeypatch, fake_reference_file):
    at = _run(monkeypatch, fake_reference_file)
    at.checkbox(key="use_example").check().run()
    at.selectbox(key="cancer_type").select("lung").run()
    assert not at.exception
    assert not at.error


@pytest.mark.skipif(
    not appcore.reference_path(appcore.load_config()).is_file(),
    reason="COSMIC reference not downloaded",
)
def test_example_with_real_cosmic(monkeypatch):
    monkeypatch.delenv(appcore.REFERENCE_ENV, raising=False)
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.checkbox(key="use_example").check().run()
    assert not at.exception and not at.error
