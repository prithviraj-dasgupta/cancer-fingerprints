"""Figure helpers for the Streamlit app (plotly, imported lazily).

The data preparation is plain pandas (:func:`spectrum_frame`) so it can be tested without
plotly. The ``*_figure`` functions need ``uv sync --extra app``.
"""

from __future__ import annotations

import pandas as pd

from fingerprints.profile import SBS96_PLOT_ORDER, SUBSTITUTIONS

__all__ = [
    "SUBSTITUTION_COLOURS",
    "exposure_figure",
    "signature_figure",
    "spectrum_figure",
    "spectrum_frame",
]

#: Standard COSMIC / SigProfiler colours for the six substitution classes.
SUBSTITUTION_COLOURS: dict[str, str] = {
    "C>A": "#03BCEE",
    "C>G": "#010101",
    "C>T": "#E32926",
    "T>A": "#CAC9C9",
    "T>C": "#A1CE63",
    "T>G": "#EBC6C4",
}


def spectrum_frame(counts: pd.Series) -> pd.DataFrame:
    """96 rows in spectrum-plot order: channel, substitution, context label, colour, count.

    ``counts`` is indexed by SBS96 channel label (any order); missing channels count as 0.
    """
    unknown = set(counts.index) - set(SBS96_PLOT_ORDER)
    if unknown:
        raise ValueError(f"not SBS96 channels: {sorted(unknown)[:3]}")
    values = counts.reindex(SBS96_PLOT_ORDER).fillna(0).astype(float)
    frame = pd.DataFrame({"channel": SBS96_PLOT_ORDER, "count": values.to_numpy()})
    frame["substitution"] = frame["channel"].str.slice(2, 5)
    frame["context"] = frame["channel"].str.replace(r"\[.>.\]", "_", regex=True)
    frame["colour"] = frame["substitution"].map(SUBSTITUTION_COLOURS)
    return frame


def spectrum_figure(counts: pd.Series, title: str = "SBS96 mutation spectrum"):
    """Bar chart of the 96 channels, coloured by substitution class (COSMIC colours)."""
    import plotly.graph_objects as go

    frame = spectrum_frame(counts)
    fig = go.Figure()
    for sub in SUBSTITUTIONS:
        part = frame[frame["substitution"] == sub]
        fig.add_bar(
            x=part["channel"],
            y=part["count"],
            name=sub,
            marker_color=SUBSTITUTION_COLOURS[sub],
            hovertemplate="%{x}: %{y:.0f}<extra></extra>",
        )
    fig.update_layout(
        title=title,
        xaxis={"categoryorder": "array", "categoryarray": list(frame["channel"]), "tickangle": 90},
        xaxis_title="Trinucleotide context (5' [ref>alt] 3')",
        yaxis_title="Number of SNVs",
        legend_title="Substitution",
        bargap=0.1,
        height=420,
    )
    fig.update_xaxes(tickfont={"size": 7})
    return fig


def _positive(series: pd.Series) -> pd.Series:
    return series[series > 0].sort_values(ascending=False)


def exposure_figure(
    group_fractions: pd.Series, title: str = "Signature groups (share of attributed mutations)"
):
    """Pie chart of the fraction of mutations per cause group (zero groups omitted)."""
    import plotly.graph_objects as go

    shown = _positive(group_fractions.astype(float))
    fig = go.Figure(
        go.Pie(
            labels=list(shown.index),
            values=list(shown.to_numpy()),
            sort=False,
            texttemplate="%{label}<br>%{percent}",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=420, showlegend=True)
    return fig


def signature_figure(signature_fractions: pd.Series, title: str = "Individual signatures"):
    """Horizontal bar chart of per-signature fractions (zero signatures omitted)."""
    import plotly.graph_objects as go

    shown = _positive(signature_fractions.astype(float)).iloc[::-1]
    fig = go.Figure(go.Bar(x=list(shown.to_numpy()), y=list(shown.index), orientation="h"))
    fig.update_layout(
        title=title,
        xaxis_title="Fraction of mutations",
        xaxis_tickformat=".0%",
        height=max(240, 40 * len(shown) + 120),
    )
    return fig
