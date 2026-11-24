"""Plain-English explanation of a tumour's signature exposures.

Takes NNLS-attributed exposures (from :mod:`fingerprints.fit`), groups them by aetiology using
the single mapping file ``config/aetiology.yaml``, and writes a hedged report. Wording rules:
"consistent with" / "most likely reflects", never "caused by"; unknown and proposed
aetiologies are labelled as such; nothing is said about the person's history (a tobacco
signature is "consistent with tobacco exposure", not "this patient smoked").

Research and education only; not a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

__all__ = [
    "CANCER_TYPES",
    "DISCLAIMER",
    "LOW_COSINE_THRESHOLD",
    "GROUP_ORDER",
    "AetiologyError",
    "Report",
    "build_report",
    "load_aetiology",
    "normalise_cancer_type",
    "signature_subset_for",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AETIOLOGY_PATH = REPO_ROOT / "config" / "aetiology.yaml"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

CANCER_TYPES = ("lung", "melanoma", "colorectal")
_PROJECT_TO_TYPE = {
    "TCGA-LUAD": "lung",
    "TCGA-LUSC": "lung",
    "TCGA-SKCM": "melanoma",
    "TCGA-COAD": "colorectal",
    "TCGA-READ": "colorectal",
}
_TYPE_TO_SUBSET_PROJECT = {"lung": "TCGA-LUAD", "melanoma": "TCGA-SKCM", "colorectal": "TCGA-COAD"}

# Default display order (overridden by `group_order` in the mapping file).
GROUP_ORDER = ["tobacco", "uv", "apobec", "mmr", "pole", "ros", "background", "unknown_other"]
STATUSES = ("known", "proposed", "unknown")

HEADLINE_MIN_GROUP = 0.05  # groups below this go into "other small contributions"
NOTE_MIN_SIGNATURE = 0.02  # proposed/unknown signatures above this get a status note
# Report/app fit-quality warning: reconstruction cosine below this means the selected cancer
# type's signature subset fits the tumour poorly. NOTE: this is deliberately different from
# fit.low_cosine_flag's 0.8, which is the pre-specified D7 validation sensitivity cut (a quality
# filter for the smoker/MSI analyses); do not unify them.
LOW_COSINE_THRESHOLD = 0.90
DEFAULT_MIN_SNVS = 50

DISCLAIMER = (
    "Research and education only; this is not a diagnosis and must not be used for clinical "
    "decisions. The result comes from non-negative least squares (NNLS) refitting of the "
    "tumour's 96-channel SBS profile against COSMIC v3.4 reference signatures (exome-adjusted, "
    "GRCh38); it is not a trained machine-learning model. Signatures describe mutation "
    "patterns and say nothing certain about any individual's history. References: COSMIC "
    "Mutational Signatures v3.4 (https://cancer.sanger.ac.uk/signatures/); Alexandrov LB et al. "
    "Nature 578, 94-101 (2020), doi:10.1038/s41586-020-1943-3."
)


class AetiologyError(ValueError):
    """The aetiology mapping file is missing or malformed."""


def load_aetiology(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the aetiology mapping (default ``config/aetiology.yaml``)."""
    p = Path(path) if path is not None else DEFAULT_AETIOLOGY_PATH
    if not p.is_absolute() and path is not None:
        p = Path.cwd() / p
    if not p.is_file():
        raise AetiologyError(f"Aetiology mapping file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise AetiologyError(f"Aetiology mapping file is not valid YAML ({p}): {exc}") from exc
    if not isinstance(data, dict):
        raise AetiologyError(f"Aetiology mapping must be a YAML mapping: {p}")
    for key in ("groups", "signatures"):
        if not isinstance(data.get(key), dict) or not data[key]:
            raise AetiologyError(f"Aetiology mapping {p} needs a non-empty '{key}' section")
    for gname, g in data["groups"].items():
        if not isinstance(g, dict) or not {"label", "sentence"} <= set(g):
            raise AetiologyError(f"Group '{gname}' needs 'label' and 'sentence' ({p})")
    for sig, entry in data["signatures"].items():
        need = {"group", "plain_english", "status", "cosmic_aetiology", "citation", "cosmic_page"}
        if not isinstance(entry, dict) or not need <= set(entry):
            raise AetiologyError(f"Signature '{sig}' needs keys {sorted(need)} ({p})")
        if entry["status"] not in STATUSES:
            raise AetiologyError(f"Signature '{sig}' has invalid status {entry['status']!r}")
        if entry["group"] not in data["groups"]:
            raise AetiologyError(f"Signature '{sig}' uses undefined group {entry['group']!r}")
    return data


def normalise_cancer_type(cancer_type: str) -> str:
    """Return ``lung``, ``melanoma`` or ``colorectal``; accepts TCGA project ids."""
    key = str(cancer_type).strip()
    if key.upper() in _PROJECT_TO_TYPE:
        return _PROJECT_TO_TYPE[key.upper()]
    if key.lower() in CANCER_TYPES:
        return key.lower()
    raise ValueError(
        f"Unknown cancer type {cancer_type!r}; expected one of {list(CANCER_TYPES)} "
        f"or a TCGA project id {sorted(_PROJECT_TO_TYPE)}"
    )


def signature_subset_for(cancer_type: str, config: dict[str, Any] | None = None) -> list[str]:
    """Signatures fitted for a cancer type (``fitting.signature_subsets`` in config.yaml).

    lung -> TCGA-LUAD list, melanoma -> TCGA-SKCM, colorectal -> TCGA-COAD (the lists are
    identical within a cancer type).
    """
    ctype = normalise_cancer_type(cancer_type)
    if config is None:
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    subsets = config.get("fitting", {}).get("signature_subsets", {})
    project = _TYPE_TO_SUBSET_PROJECT[ctype]
    if project not in subsets:
        raise ValueError(f"config has no fitting.signature_subsets entry for {project}")
    return list(subsets[project])


@dataclass
class Report:
    sample_id: str | None
    cancer_type: str
    n_snvs: int
    warnings: list[str] = field(default_factory=list)
    group_fractions: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    signature_fractions: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    headline: str = ""
    lines: list[str] = field(default_factory=list)
    status_notes: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER
    cosine: float | None = None
    low_confidence: bool = False
    unknown_fraction: float = 0.0

    @property
    def title(self) -> str:
        who = f" for {self.sample_id}" if self.sample_id else ""
        return f"Mutational signature summary{who} ({self.cancer_type}, {self.n_snvs} SNVs)"

    def to_text(self) -> str:
        out = [self.title, "", f"Headline: {self.headline}"]
        if self.warnings:
            out += ["", "Warnings:"] + [f"  - {w}" for w in self.warnings]
        if self.lines:
            out += [""] + [f"  - {x}" for x in self.lines]
        if self.status_notes:
            out += ["", "Certainty of the aetiology:"] + [f"  - {x}" for x in self.status_notes]
        out += ["", self.disclaimer]
        return "\n".join(out)

    def to_markdown(self) -> str:
        out = [f"### {self.title}", "", f"**{self.headline}**"]
        if self.warnings:
            out += ["", "**Warnings**", ""] + [f"- {w}" for w in self.warnings]
        if self.lines:
            out += [""] + [f"- {x}" for x in self.lines]
        if self.status_notes:
            out += ["", "**Certainty of the aetiology**", ""] + [
                f"- {x}" for x in self.status_notes
            ]
        out += ["", f"*{self.disclaimer}*"]
        return "\n".join(out)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _round5(x: float) -> int:
    return int(5 * round(100 * x / 5))


def build_report(
    exposures: pd.Series,
    n_snvs: int,
    cancer_type: str,
    sample_id: str | None = None,
    cosine: float | None = None,
    aetiology: dict[str, Any] | None = None,
    min_snvs: int = DEFAULT_MIN_SNVS,
    low_cosine_threshold: float = LOW_COSINE_THRESHOLD,
) -> Report:
    """Build a hedged plain-English report from signature exposures.

    ``exposures`` maps signature name -> NNLS-attributed count (or fraction); it is
    normalised to fractions. Negative or missing values are treated as zero.
    """
    ctype = normalise_cancer_type(cancer_type)
    aet = aetiology if aetiology is not None else load_aetiology()
    sigs = aet["signatures"]
    groups = aet["groups"]
    order = list(aet.get("group_order", GROUP_ORDER))

    exp = pd.Series(exposures, dtype=float).fillna(0.0).clip(lower=0.0)
    n_snvs = int(n_snvs)
    warnings: list[str] = []
    low_conf = False
    if n_snvs < min_snvs:
        low_conf = True
        warnings.append(
            f"Low confidence: only {n_snvs} SNVs (fewer than {min_snvs}); with so few mutations "
            "the signature split is unreliable."
        )
    has_cosine = cosine is not None and not np.isnan(cosine)
    if has_cosine and cosine < low_cosine_threshold:
        low_conf = True
        warnings.append(
            f"The selected cancer type's signatures fit this tumour poorly (cosine {cosine:.2f}, "
            f"below {low_cosine_threshold:.2f}). Check that the cancer type is right."
        )

    total = float(exp.sum())
    report = Report(
        sample_id=sample_id,
        cancer_type=ctype,
        n_snvs=n_snvs,
        warnings=warnings,
        cosine=cosine,
        low_confidence=low_conf,
    )
    if total <= 0:
        warnings.append("All signature exposures are zero.")
        report.headline = "No confident attribution is possible for this tumour."
        report.lines = [
            "No signature received a non-zero exposure, so nothing can be said about the "
            "likely causes of these mutations."
        ]
        if has_cosine:
            report.lines.append(f"Fit quality: cosine similarity is {cosine:.2f}.")
        return report

    sig_frac = (exp / total)[lambda s: s > 0].sort_values(ascending=False)
    report.signature_fractions = sig_frac

    unmapped = [s for s in sig_frac.index if s not in sigs]
    if unmapped:
        warnings.append(
            f"No aetiology entry for {', '.join(unmapped)}; shown as 'signature without "
            "aetiology entry'."
        )
    group_of = {s: sigs[s]["group"] if s in sigs else "unmapped" for s in sig_frac.index}
    gf = sig_frac.groupby(group_of).sum()
    names = [g for g in order if g in gf.index] + sorted(set(gf.index) - set(order))
    gf = gf.reindex(sorted(names, key=lambda g: -gf[g]))
    gf = gf / gf.sum()
    report.group_fractions = gf

    def label(g: str) -> str:
        return groups[g]["label"] if g in groups else "signature without aetiology entry"

    big = [g for g in gf.index if gf[g] >= HEADLINE_MIN_GROUP]
    small = [g for g in gf.index if gf[g] < HEADLINE_MIN_GROUP]
    parts = [f"~{_round5(gf[g])}% {label(g)}" for g in big]
    if small:
        parts.append("other small contributions (each under 5%)")
    report.headline = "Most consistent with: " + ", ".join(parts)

    lines = []
    if has_cosine:
        lines.append(
            f"Fit quality: cosine similarity between the observed and reconstructed profile is "
            f"{cosine:.2f} (1.00 is a perfect reconstruction)."
        )
    lines += [
        "These percentages are the share of attributed mutations, not a measure of risk, and "
        "they may not add to exactly 100% because of rounding in the headline."
    ]
    for g in gf.index:
        members = [s for s in sig_frac.index if group_of[s] == g]
        member_txt = ", ".join(f"{s} {_pct(sig_frac[s])}" for s in members)
        sentence = groups[g]["sentence"].strip() if g in groups else ""
        lines.append(f"{_pct(gf[g])} {label(g)} ({member_txt}). {sentence}".strip())
    report.lines = lines

    unknown = [s for s in sig_frac.index if s in sigs and sigs[s]["status"] == "unknown"]
    report.unknown_fraction = float(sig_frac[unknown].sum()) if unknown else 0.0
    notes: list[str] = []
    if unknown:
        notes.append(
            f"{_pct(report.unknown_fraction)} of mutations are attributed to signatures with "
            f"unknown cause ({', '.join(unknown)}); this is reported as unknown cause, not as "
            "a confident explanation. SBS5 and SBS40a are clock-like or flat patterns whose "
            "cause is unknown."
        )
    for s in sig_frac.index:
        if s in sigs and sig_frac[s] >= NOTE_MIN_SIGNATURE:
            e = sigs[s]
            if e["status"] == "proposed":
                notes.append(
                    f"{s} ({_pct(sig_frac[s])}): aetiology is proposed, not established; "
                    f"{e['cosmic_aetiology'].strip()}"
                )
            elif e["status"] == "unknown":
                notes.append(
                    f"{s} ({_pct(sig_frac[s])}): cause unknown; {e['cosmic_aetiology'].strip()}"
                )
    report.status_notes = notes
    return report
