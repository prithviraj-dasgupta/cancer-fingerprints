"""Tests for the figure helpers (plotly parts skip when the `app` extra is absent)."""

from __future__ import annotations

import pandas as pd
import pytest

from fingerprints import plots
from fingerprints.profile import SBS96_CHANNELS, SBS96_PLOT_ORDER


def test_spectrum_frame_order_and_colours():
    counts = pd.Series({"A[C>A]A": 3, "T[T>G]T": 5})
    frame = plots.spectrum_frame(counts)
    assert list(frame["channel"]) == SBS96_PLOT_ORDER
    assert frame["count"].sum() == 8
    assert frame.loc[frame["channel"] == "A[C>A]A", "colour"].item() == "#03BCEE"
    assert frame.loc[frame["channel"] == "T[T>G]T", "substitution"].item() == "T>G"
    assert frame.loc[frame["channel"] == "A[C>A]C", "context"].item() == "A_C"
    assert set(frame["colour"]) == set(plots.SUBSTITUTION_COLOURS.values())


def test_spectrum_frame_accepts_matrix_order_and_rejects_unknown():
    assert len(plots.spectrum_frame(pd.Series(1, index=SBS96_CHANNELS))) == 96
    with pytest.raises(ValueError, match="not SBS96"):
        plots.spectrum_frame(pd.Series({"X[C>A]A": 1}))


def test_figures_build():
    pytest.importorskip("plotly", reason="needs `uv sync --extra app`")
    fig = plots.spectrum_figure(pd.Series({"A[C>A]A": 3}))
    assert len(fig.data) == 6
    assert sum(sum(t.y) for t in fig.data) == 3
    pie = plots.exposure_figure(pd.Series({"UV": 0.8, "Ageing": 0.2, "None": 0.0}))
    assert list(pie.data[0].labels) == ["UV", "Ageing"]
    bar = plots.signature_figure(pd.Series({"SBS7a": 0.7, "SBS1": 0.3, "SBS5": 0.0}))
    assert list(bar.data[0].y) == ["SBS1", "SBS7a"]
