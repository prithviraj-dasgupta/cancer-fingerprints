"""Logic behind the Streamlit app (kept out of ``app/streamlit_app.py`` so it is testable).

Flow: uploaded MAF -> SBS96 matrix (:mod:`fingerprints.profile`) -> NNLS refit to the cancer
type's COSMIC subset, artefact signatures excluded (:mod:`fingerprints.fit`, D7) -> plain-English
report (:mod:`fingerprints.explain`). Nothing here needs streamlit or plotly.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from fingerprints import fit, profile

if TYPE_CHECKING:
    from fingerprints.explain import Report

__all__ = [
    "CANCER_TYPES",
    "EXAMPLE_MAF",
    "REFERENCE_ENV",
    "SIGNATURE_SETUP_COMMAND",
    "VCF_MESSAGE",
    "Analysis",
    "analyse_sample",
    "classify_upload",
    "load_config",
    "profile_warnings",
    "read_upload",
    "reference_path",
]

CANCER_TYPES = ("lung", "melanoma", "colorectal")
REFERENCE_ENV = "FINGERPRINTS_SIGNATURE_FILE"  # optional override of the COSMIC file path
SIGNATURE_SETUP_COMMAND = (
    "uv sync --extra signatures && uv run python scripts/download_signatures.py"
)
VCF_MESSAGE = (
    "VCF input is not supported yet (it needs a GRCh38 trinucleotide-context lookup). "
    "Please upload a MAF with a CONTEXT column."
)
MAF_SUFFIXES = (".maf", ".maf.gz", ".tsv", ".tsv.gz", ".txt")
VCF_SUFFIXES = (".vcf", ".vcf.gz")


def repo_root() -> Path:
    """Repository root (``config/`` lives there); falls back to the working directory."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "config" / "config.yaml").is_file() else Path.cwd()


EXAMPLE_MAF = repo_root() / "app" / "example" / "synthetic_melanoma.maf"


def load_config(path: str | Path | None = None) -> dict:
    """Load ``config/config.yaml`` (or ``path``)."""
    import yaml

    path = Path(path) if path else repo_root() / "config" / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config {path} is not a YAML mapping")
    return config


def reference_path(config: dict) -> Path:
    """Where the COSMIC reference file should be (env override, else per the config)."""
    override = os.environ.get(REFERENCE_ENV)
    if override:
        return Path(override)
    raw = (config.get("paths") or {}).get("raw_signatures", "data/raw/signatures")
    return repo_root() / raw / config["signatures"]["file"]


def classify_upload(filename: str) -> str:
    """Return ``"maf"``, ``"vcf"`` or ``"unsupported"`` from the file name."""
    name = filename.lower()
    if name.endswith(VCF_SUFFIXES):
        return "vcf"
    if name.endswith(MAF_SUFFIXES):
        return "maf"
    return "unsupported"


def read_upload(data: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded MAF bytes (plain or gzipped; gzip is detected from content)."""
    suffix = ".gz" if data[:2] == b"\x1f\x8b" else ".maf"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        return profile.read_maf(path)


def profile_warnings(row: pd.Series) -> list[str]:
    """Plain-language warnings from one sample's :func:`profile.drop_summary` row (D5)."""
    out: list[str] = []
    total, kept = int(row["total"]), int(row["kept"])
    dropped = {r: int(row[r]) for r in profile.DROP_REASONS if int(row[r]) > 0}
    if dropped:
        detail = ", ".join(f"{n} {r.replace('_', ' ')}" for r, n in dropped.items())
        out.append(
            f"{total - kept} of {total} mutation rows were not used in the profile ({detail})."
        )
    dbs, mnv = int(row["dbs_events"]), int(row["mnv_events"])
    if dbs or mnv:
        out.append(
            f"Adjacent SNVs were found: {dbs} likely doublet (DBS) and {mnv} likely multi-base "
            "(MNV) events. These are excluded from the SBS profile (decision D5) because they "
            "probably arose as one event; they are not counted in any signature."
        )
    return out


@dataclass
class Analysis:
    """Everything the app shows for one tumour."""

    sample: str
    n_snvs: int
    counts: pd.Series  # 96 channels
    exposures: pd.Series  # NNLS-attributed counts per signature
    cosine: float
    report: Report
    warnings: list[str]

    def body_markdown(self) -> str:
        """Report text without the warnings block (the app shows warnings separately)."""
        return replace(self.report, warnings=[]).to_markdown()

    def full_markdown(self) -> str:
        """Report text with every warning, including the profiling ones (for download)."""
        return replace(self.report, warnings=self.warnings).to_markdown()

    def exposures_tsv(self) -> str:
        total = float(self.exposures.sum())
        table = pd.DataFrame(
            {
                "signature": self.exposures.index,
                "exposure": self.exposures.to_numpy(),
                "fraction": (self.exposures / total).to_numpy() if total > 0 else 0.0,
            }
        )
        return table.to_csv(sep="\t", index=False, float_format="%.6g")


def analyse_sample(
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    sample: str,
    cancer_type: str,
    config: dict,
    reference: pd.DataFrame,
) -> Analysis:
    """Fit one tumour of ``matrix`` to the cancer type's D7 subset and build its report."""
    from fingerprints import explain  # lazy: the example-file script does not need it

    if cancer_type not in CANCER_TYPES:
        raise ValueError(f"cancer_type must be one of {list(CANCER_TYPES)}, got {cancer_type!r}")
    if sample not in matrix.columns:
        raise ValueError(f"sample {sample!r} is not in the matrix")
    signatures = explain.signature_subset_for(cancer_type, config)
    artefacts = list((config.get("fitting") or {}).get("exclude_artefacts") or [])
    subset = fit.subset_reference(reference, signatures, artefacts)
    one = matrix[[sample]]
    exposures, _fractions, cosine = fit.fit_exposures(one, subset)
    expo = exposures.iloc[0].rename(sample)
    n_snvs = int(one[sample].sum())
    min_snvs = int((config.get("fitting") or {}).get("min_snvs", fit.DEFAULT_MIN_SNVS))
    cos = float(cosine.iloc[0])
    report = explain.build_report(
        expo, n_snvs, cancer_type, sample_id=sample, cosine=cos, min_snvs=min_snvs
    )
    warnings = profile_warnings(summary.loc[sample])
    warnings += [w for w in report.warnings if w not in warnings]
    return Analysis(sample, n_snvs, one[sample], expo, cos, report, warnings)
