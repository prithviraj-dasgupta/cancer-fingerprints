"""Cancer Fingerprints: local Streamlit demo (research and education only).

Run: ``uv sync --extra app && uv run streamlit run app/streamlit_app.py``.
This file only does layout; the logic lives in ``fingerprints.appcore`` and ``fingerprints.plots``.
"""

from __future__ import annotations

import streamlit as st

from fingerprints import appcore, fit, plots, profile

st.set_page_config(page_title="Cancer Fingerprints", layout="wide")

DISCLAIMER = (
    "**Research and education only. Not for diagnosis or treatment decisions.** "
    "Results show which known mutational signatures best explain a mutation profile; they do "
    "not prove a cause."
)
CITATION = (
    "Signatures: COSMIC v3.4 SBS (GRCh38, exome-adjusted), Alexandrov et al. Nature 578:94-101 "
    "(2020); Sondka et al. Nucleic Acids Res 52:D1210 (2024). COSMIC data are used under the "
    "[COSMIC terms](https://www.cosmickb.org/terms/) for non-commercial research and are not "
    "redistributed here, so this app runs locally only. Fitting is NNLS refitting against the "
    "reference, not a trained ML model."
)


def _set_example_defaults() -> None:
    if st.session_state.get("use_example"):
        st.session_state["cancer_type"] = "melanoma"


@st.cache_data(show_spinner=False)
def _profile(data: bytes, name: str):
    maf = appcore.read_upload(data, name)
    return profile.build_sbs96_matrix(maf, return_dropped=True)


def _load_input():
    """Return (data bytes, name) from the example or the uploader, or None with a message."""
    use_example = st.checkbox(
        "Use the bundled synthetic example (melanoma-like, no real patient data)",
        key="use_example",
        on_change=_set_example_defaults,
    )
    upload = None if use_example else st.file_uploader("Upload a MAF", type=None)
    if use_example:
        return appcore.EXAMPLE_MAF.read_bytes(), appcore.EXAMPLE_MAF.name
    if upload is None:
        st.info("Upload a MAF with a CONTEXT column, or tick the example above.")
        return None
    kind = appcore.classify_upload(upload.name)
    if kind == "vcf":
        st.error(appcore.VCF_MESSAGE)
        return None
    if kind == "unsupported":
        st.error("Unrecognised file type. Please upload a .maf, .maf.gz or .tsv file.")
        return None
    return upload.getvalue(), upload.name


def _show(analysis: appcore.Analysis) -> None:
    report = analysis.report
    st.subheader(f"Summary for {analysis.sample}")
    for message in analysis.warnings:
        st.warning(message)
    st.markdown(analysis.body_markdown())
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plots.exposure_figure(report.group_fractions), width="stretch")
    with right:
        st.plotly_chart(plots.signature_figure(report.signature_fractions), width="stretch")
    st.plotly_chart(plots.spectrum_figure(analysis.counts), width="stretch")
    st.caption(
        f"{analysis.n_snvs} SNVs in the profile. Reconstruction cosine similarity: "
        f"{analysis.cosine:.2f} (closer to 1 means a better fit)."
    )
    left, right = st.columns(2)
    left.download_button(
        "Download report (markdown)",
        analysis.full_markdown(),
        file_name=f"{analysis.sample}_report.md",
        mime="text/markdown",
    )
    right.download_button(
        "Download exposures (TSV)",
        analysis.exposures_tsv(),
        file_name=f"{analysis.sample}_exposures.tsv",
        mime="text/tab-separated-values",
    )


def main() -> None:
    st.title("Cancer Fingerprints")
    st.caption(
        "Reads a tumour's mutations and explains, in plain English, what they are consistent with."
    )
    st.info(DISCLAIMER)

    config = appcore.load_config()
    st.sidebar.header("Settings")
    cancer_type = st.sidebar.selectbox(
        "Cancer type",
        appcore.CANCER_TYPES,
        index=appcore.CANCER_TYPES.index("melanoma"),
        key="cancer_type",
        help="Selects the cohort-specific set of signatures to fit (decision D7).",
    )
    st.sidebar.markdown(CITATION)

    ref_path = appcore.reference_path(config)
    if not ref_path.is_file():
        st.error(
            f"The COSMIC signature file is missing ({ref_path}). Run "
            f"`{appcore.SIGNATURE_SETUP_COMMAND}` once, then reload this page."
        )
        return

    source = _load_input()
    if source is None:
        return
    data, name = source
    try:
        matrix, summary = _profile(data, name)
        reference = fit.load_reference(ref_path)
        samples = list(matrix.columns)
        if not samples:
            st.error("No tumour samples were found in this file.")
            return
        sample = st.selectbox("Tumour sample", samples, index=0) if len(samples) > 1 else samples[0]
        analysis = appcore.analyse_sample(matrix, summary, sample, cancer_type, config, reference)
    except (ValueError, OSError) as exc:
        st.error(f"Could not analyse this file: {exc}")
        return
    _show(analysis)


main()
