"""Signature refitting: assign COSMIC SBS signatures to tumours by NNLS.

Method (D7): non-negative least squares (NNLS) refitting of each tumour's 96-channel
SBS counts against a fixed set of COSMIC v3.4 GRCh38 exome-adjusted reference signatures,
restricted to the approved cohort subset in ``config/config.yaml`` (``fitting.signature_subsets``)
with the artefact signatures excluded. This is signature refitting against published
reference signatures; it is not a trained machine-learning model. COSMIC data stay in
``data/raw/signatures`` (licence: no redistribution).

Why direct NNLS (``scipy.optimize.nnls``) and not SigProfilerAssignment's
``Analyzer.cosmic_fit``: ``cosmic_fit`` runs its own forward/backward selection
(``nnls_add_penalty``, ``nnls_remove_penalty``, ``initial_remove_penalty``), can add
"background" signatures (SBS1/SBS5, ``add_background_signatures``) and "connected"
signatures (``connected_sigs``) on its own, and writes result folders and plots. That
selection step could drop or add validation signatures (SBS4, SBS7, MMR) by its own
rules, which would confound the validation. Plain NNLS on exactly the approved subset is
deterministic and every signature in the subset is eligible in every tumour. The two methods
were cross-checked on real cohort data (see the Phase 2 report). SigProfilerAssignment is
only used by the optional :func:`cross_check_with_sigprofiler` helper, imported lazily
(``uv sync --extra signatures``).

Exposures are NNLS-attributed mutation counts, NOT observed counts: when the fit is poor
their sum can differ from the observed SNV total (a pure SBS7a tumour fitted without SBS7a
leaks into other signatures and over- or under-counts). The per-tumour QC table
(``fit_qc_<project>.tsv``) reports observed SNVs, summed exposure, L1 residual and cosine so
poorly fitted tumours can be seen; :func:`low_cosine_flag` is a reporting flag for a
sensitivity analysis and never uses smoking or MSI status.

The ``min_snvs`` thresholds apply to SBS96 matrix totals, i.e. AFTER the profile step's drops
(non-SNVs, adjacent-SNV exclusion, ...), so they depend on ``profile.adjacent_snvs``.

Rows are always matched to the reference by channel LABEL, never by position (the COSMIC
file is in lexicographic order, our matrix in :data:`SBS96_CHANNELS` order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from fingerprints.profile import SBS96_CHANNELS

METHOD_DESCRIPTION = "NNLS signature refitting (scipy.optimize.nnls) against COSMIC v3.4"
DEFAULT_MIN_SNVS = 50
DEFAULT_SENSITIVITY = (20, 100)

__all__ = [
    "FitError",
    "FitResult",
    "cross_check_with_sigprofiler",
    "fit_exposures",
    "load_reference",
    "low_cosine_flag",
    "run_cohort_fit",
    "save_results",
    "select_tumours",
    "subset_reference",
]


class FitError(ValueError):
    """A fitting input is unusable (clean message for the CLI)."""


@dataclass
class FitResult:
    """Output of :func:`run_cohort_fit`."""

    project: str
    signatures: list[str]
    exposures: pd.DataFrame  # tumours x signatures, NNLS-attributed mutation counts
    fractions: pd.DataFrame  # tumours x signatures, rows sum to 1 (or 0 for no exposure)
    cosine: pd.Series  # per tumour, reconstructed vs observed
    qc: pd.DataFrame  # per tumour: observed_snvs, sum_exposures, residual_l1, cosine
    excluded_table: pd.DataFrame  # all input tumours below the largest threshold
    n_input: int
    min_snvs: int
    excluded: list[str]  # tumours below min_snvs (main threshold)
    sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)  # threshold, n_excluded

    def summary(self) -> pd.DataFrame:
        """One-row-per-statistic table for ``fit_summary_<project>.tsv``."""
        rows: list[tuple[str, Any]] = [
            ("project", self.project),
            ("method", METHOD_DESCRIPTION),
            ("n_signatures_fitted", len(self.signatures)),
            ("n_tumours_input", self.n_input),
            ("min_snvs", self.min_snvs),
            ("n_tumours_fitted", len(self.exposures)),
            ("n_excluded_main", len(self.excluded)),
        ]
        for _, r in self.sensitivity.iterrows():
            rows.append((f"n_excluded_min_snvs_{int(r['min_snvs'])}", int(r["n_excluded"])))
        mean_cos = float(self.cosine.mean()) if len(self.cosine) else float("nan")
        rows.append(("mean_reconstruction_cosine", mean_cos))
        return pd.DataFrame(rows, columns=["statistic", "value"])


def load_reference(path: str | Path) -> pd.DataFrame:
    """Read a COSMIC SBS96 reference file; return 96 x signatures in SBS96_CHANNELS order."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"signature reference not found: {path} "
            "(run `uv sync --extra signatures` then `uv run python scripts/download_signatures.py`)"
        )
    ref = pd.read_csv(path, sep="\t", index_col=0)
    return _reindex_by_label(ref, f"reference {path}")


def _reindex_by_label(frame: pd.DataFrame, what: str) -> pd.DataFrame:
    labels = list(frame.index)
    if len(set(labels)) != len(labels) or set(labels) != set(SBS96_CHANNELS):
        raise FitError(f"{what}: row labels are not exactly the 96 SBS channels")
    out = frame.reindex(SBS96_CHANNELS)
    out.index.name = frame.index.name
    return out


def subset_reference(
    reference: pd.DataFrame, signatures: list[str], exclude_artefacts: list[str]
) -> pd.DataFrame:
    """Restrict the reference to the cohort signatures (in the order given)."""
    if len(set(signatures)) != len(signatures):
        raise FitError("duplicate signatures requested")
    artefacts = sorted(set(signatures) & set(exclude_artefacts))
    if artefacts:
        raise FitError(f"requested signatures are listed as artefacts: {', '.join(artefacts)}")
    missing = [s for s in signatures if s not in reference.columns]
    if missing:
        raise FitError(f"signatures not in the reference: {', '.join(missing)}")
    return reference[list(signatures)].copy()


def _validate_counts(matrix: pd.DataFrame) -> None:
    """Raise FitError for duplicate tumour names, or negative / non-finite counts."""
    dup = matrix.columns[matrix.columns.duplicated()].unique()
    if len(dup):
        raise FitError(f"duplicate tumour column names: {', '.join(map(str, dup[:5]))}")
    try:
        values = matrix.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise FitError("matrix contains non-numeric counts") from exc
    if not np.isfinite(values).all():
        raise FitError("matrix contains non-finite counts (NaN or inf)")
    if (values < 0).any():
        raise FitError("matrix contains negative counts")


def select_tumours(matrix: pd.DataFrame, min_snvs: int) -> tuple[pd.DataFrame, list[str]]:
    """Keep tumours with at least ``min_snvs`` SNVs; return (kept matrix, excluded names)."""
    _validate_counts(matrix)
    totals = matrix.sum(axis=0)
    keep = totals >= min_snvs
    return matrix.loc[:, keep], [str(c) for c in matrix.columns[~keep.to_numpy()]]


def fit_exposures(
    matrix: pd.DataFrame, reference_subset: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """NNLS-fit each tumour; return (exposures, fractions, cosine).

    ``matrix`` is 96 x tumours (counts), ``reference_subset`` 96 x signatures; both are
    matched by label. Exposures and fractions are tumours x signatures. Exposures are
    NNLS-attributed mutation counts, not observed counts (reference columns are normalised
    to sum 1 first; the sum can differ from the observed total when the fit is poor).
    Fractions are exposure / sum of
    exposures (0 when a tumour has no exposure). Cosine is between the reconstruction and
    the observed profile (NaN for an all-zero tumour).
    """
    _validate_counts(matrix)
    matrix = _reindex_by_label(matrix, "matrix")
    ref = _reindex_by_label(reference_subset, "reference subset")
    sig_sums = ref.sum(axis=0)
    if (sig_sums <= 0).any():
        raise FitError("a reference signature sums to zero")
    a = (ref / sig_sums).to_numpy(dtype=float)
    expo: dict[str, np.ndarray] = {}
    cos: dict[str, float] = {}
    for tumour in matrix.columns:
        b = matrix[tumour].to_numpy(dtype=float)
        x, _ = nnls(a, b)
        expo[tumour] = x
        recon = a @ x
        denom = np.linalg.norm(recon) * np.linalg.norm(b)
        cos[tumour] = float(recon @ b / denom) if denom > 0 else float("nan")
    exposures = pd.DataFrame(expo, index=ref.columns).T
    exposures.index.name = "tumour"
    totals = exposures.sum(axis=1)
    fractions = exposures.div(totals.where(totals > 0), axis=0).fillna(0.0)
    return exposures, fractions, pd.Series(cos, name="reconstruction_cosine", dtype=float)


def low_cosine_flag(qc: pd.DataFrame, threshold: float = 0.8) -> pd.Series:
    """True where reconstruction cosine is below ``threshold`` (NaN cosine -> False).

    A reporting flag for a sensitivity analysis only. It must never use smoking or MSI status.
    """
    return (qc["cosine"] < threshold).rename("low_cosine")


def _qc_table(
    kept: pd.DataFrame, subset: pd.DataFrame, exposures: pd.DataFrame, cosine: pd.Series
) -> pd.DataFrame:
    """Per-tumour QC: observed SNVs, summed exposure, L1 residual, cosine."""
    a = (subset / subset.sum(axis=0)).reindex(SBS96_CHANNELS).to_numpy(dtype=float)
    obs = kept.reindex(SBS96_CHANNELS).to_numpy(dtype=float)
    recon = a @ exposures.to_numpy(dtype=float).T  # 96 x tumours
    qc = pd.DataFrame(
        {
            "observed_snvs": obs.sum(axis=0),
            "sum_exposures": exposures.sum(axis=1).to_numpy(),
            "residual_l1": np.abs(obs - recon).sum(axis=0),
            "cosine": cosine.to_numpy(),
        },
        index=exposures.index,
    )
    qc.index.name = "tumour"
    return qc


def _fitting_config(config: dict, project: str) -> tuple[list[str], list[str]]:
    fitting = config.get("fitting")
    if not isinstance(fitting, dict):
        raise FitError("config has no 'fitting' section")
    subsets = fitting.get("signature_subsets") or {}
    if project not in subsets:
        raise FitError(f"no signature subset for project {project} in config fitting section")
    return list(subsets[project]), list(fitting.get("exclude_artefacts") or [])


def run_cohort_fit(
    project: str,
    config: dict,
    matrix: pd.DataFrame,
    reference: pd.DataFrame,
    min_snvs: int | None = None,
) -> FitResult:
    """Fit one project's cohort and report exclusions at the main and sensitivity thresholds.

    ``min_snvs`` overrides ``fitting.min_snvs``. Sensitivity thresholds come from
    ``fitting.min_snvs_sensitivity``; only the number excluded is reported for them.
    """
    signatures, artefacts = _fitting_config(config, project)
    fitting = config["fitting"]
    threshold = int(min_snvs if min_snvs is not None else fitting.get("min_snvs", DEFAULT_MIN_SNVS))
    sens = [int(t) for t in fitting.get("min_snvs_sensitivity", DEFAULT_SENSITIVITY)]
    thresholds = sorted({threshold, *sens})
    subset = subset_reference(reference, signatures, artefacts)
    _validate_counts(matrix)
    matrix = _reindex_by_label(matrix, "matrix")
    totals = matrix.sum(axis=0)
    kept, excluded = select_tumours(matrix, threshold)
    exposures, fractions, cosine = fit_exposures(kept, subset)
    qc = _qc_table(kept, subset, exposures, cosine)
    rows = [{"min_snvs": t, "n_excluded": int((totals < t).sum())} for t in thresholds]
    below = totals[totals < max(thresholds)]
    excl = pd.DataFrame({"tumour": below.index.astype(str), "total_snvs": below.to_numpy()})
    for t in thresholds:
        excl[f"excluded_at_{t}"] = excl["total_snvs"] < t
    excl["excluded_main"] = excl["total_snvs"] < threshold
    return FitResult(
        project=project,
        signatures=signatures,
        exposures=exposures,
        fractions=fractions,
        cosine=cosine,
        qc=qc,
        excluded_table=excl,
        n_input=matrix.shape[1],
        min_snvs=threshold,
        excluded=excluded,
        sensitivity=pd.DataFrame(rows),
    )


def save_results(result: FitResult, output_dir: str | Path) -> list[Path]:
    """Write exposures, fractions and summary TSVs; return the paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = result.project
    paths = [
        out / f"exposures_{p}.tsv",
        out / f"exposures_{p}_fractions.tsv",
        out / f"fit_summary_{p}.tsv",
        out / f"fit_qc_{p}.tsv",
        out / f"excluded_{p}.tsv",
    ]
    result.exposures.to_csv(paths[0], sep="\t", float_format="%.6g")
    result.fractions.to_csv(paths[1], sep="\t", float_format="%.6g")
    summary = result.summary()
    summary.to_csv(paths[2], sep="\t", index=False, float_format="%.6g")
    result.qc.to_csv(paths[3], sep="\t", float_format="%.6g")
    result.excluded_table.to_csv(paths[4], sep="\t", index=False)
    return paths


def cross_check_with_sigprofiler(
    matrix: pd.DataFrame, signatures: list[str], reference_path: str | Path, workdir: str | Path
) -> pd.DataFrame:
    """Run SigProfilerAssignment ``cosmic_fit`` on the same subset (for cross-checks only).

    Returns exposures (tumours x signatures). Needs ``uv sync --extra signatures``.
    Selection options are switched off as far as the package allows, plots are off.
    """
    try:
        from SigProfilerAssignment import Analyzer
    except ImportError as exc:
        raise FitError(
            "SigProfilerAssignment is not installed (run `uv sync --extra signatures`)"
        ) from exc
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ref = load_reference(reference_path)[list(signatures)]
    db = workdir / "subset_signatures.txt"
    ref.to_csv(db, sep="\t")
    samples = workdir / "samples.txt"
    _reindex_by_label(matrix, "matrix").to_csv(samples, sep="\t")
    Analyzer.cosmic_fit(
        samples=str(samples),
        output=str(workdir / "spa_out"),
        signature_database=str(db),
        genome_build="GRCh38",
        cosmic_version=3.4,
        exome=True,
        make_plots=False,
        add_background_signatures=False,
        connected_sigs=False,
        export_probabilities=False,
        verbose=False,
        cpu=1,
    )
    found = list(workdir.glob("spa_out/**/Activities/Assignment_Solution_Activities.txt"))
    if not found:
        raise FitError("SigProfilerAssignment produced no activities table")
    act = pd.read_csv(found[0], sep="\t", index_col=0)
    return act.reindex(columns=list(signatures)).fillna(0.0)
