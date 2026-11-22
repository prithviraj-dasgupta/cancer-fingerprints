"""Cohort sample selection and clinical groups (decisions D2, D3, D4).

Pure pandas functions, no network. They turn the downloaded manifests, clinical tables and
MSI tables into one row per patient with the labels the validation step needs.

D2  One tumour sample per patient (:func:`select_samples`)
    TCGA barcodes look like ``TCGA-XX-XXXX-SSV-...``; characters 14-15 (the ``SS`` of the
    sample part) are the sample type code. Rule: prefer ``01`` (primary solid tumour); if the
    patient has none, use ``06`` (metastatic); otherwise drop the patient. Among several
    aliquots of the chosen type, take the lowest barcode (plain string order), so the result
    never depends on file order. Every candidate gets a row with ``chosen`` and a ``reason``:
    that table is the selection log.

D3  Smoking groups (:func:`add_smoking_groups`)
    GDC ``tobacco_smoking_status`` strings map to the GDC codes 1 lifelong non-smoker,
    2 current smoker, 3 reformed >15 y, 4 reformed <=15 y, 5 reformed, duration not specified
    (matched case-insensitively). ``smoking_primary_group`` is "never" (1) vs "smoker"
    (2 and 4); codes 3 and 5 are NA there. ``smoking_ever_group`` (sensitivity) is "never" (1)
    vs "ever" (2-5). "Not Reported", "Unknown" and blank are excluded.
    CONFLICT RULE: a case flagged ``tobacco_status_conflict`` (its exposure records hold more
    than one distinct informative status) is EXCLUDED from all smoking analyses (both groups
    NA, ``smoking_exclusion`` = "conflict") and counted. We do not guess which record is right.

D4  MSI per case (:func:`choose_msi`)
    One call per case. Prefer the WXS BAM (the MAFs are WXS); if there is none, use the WGS
    call. Several rows of the preferred strategy: take the lowest (aliquot barcode, file id)
    and record ``n_msi_rows``. ``msi_borderline`` is true when the chosen ``msi_score`` is in
    [0.15, 0.25] (both ends included). ``msi_wxs_wgs_disagree`` is true when the set of WXS
    statuses differs from the set of WGS statuses (both must exist). Cases with no MSI row
    are not in the output: they are excluded from the MMR comparison by the caller.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------
# D2: sample selection
# --------------------------------------------------------------------------------------

#: TCGA sample type codes (https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables).
SAMPLE_TYPE_LABELS = {
    "01": "Primary Solid Tumor",
    "02": "Recurrent Solid Tumor",
    "03": "Primary Blood Derived Cancer - Peripheral Blood",
    "05": "Additional - New Primary",
    "06": "Metastatic",
    "07": "Additional Metastatic",
    "10": "Blood Derived Normal",
    "11": "Solid Tissue Normal",
}

PREFERRED_TYPES = ("01", "06")  # in order of preference

SAMPLES_COLUMNS = [
    "project",
    "case_id",
    "case_submitter_id",
    "barcode",
    "file_id",
    "sample_type_code",
    "sample_type_label",
    "chosen",
    "reason",
]


def sample_type_code(barcode: str) -> str:
    """Two-digit sample type of a TCGA barcode (characters 14-15, e.g. 'TCGA-XX-XXXX-01A-...')."""
    m = re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(\d{2})", str(barcode))
    if not m:
        raise ValueError(f"Not a TCGA sample barcode: {barcode!r}")
    return m.group(1)


def sample_type_label(code: str) -> str:
    return SAMPLE_TYPE_LABELS.get(code, f"Other ({code})")


def read_maf_barcodes(path: str | Path) -> list[str]:
    """Distinct ``Tumor_Sample_Barcode`` values in one MAF (plain or .gz), sorted.

    GDC masked MAFs normally hold one tumour aliquot each, but this does not assume it.
    Only that column is read.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        n_comment = 0
        for line in fh:
            if not line.startswith("#"):
                break
            n_comment += 1
    df = pd.read_csv(
        path, sep="\t", skiprows=n_comment, usecols=["Tumor_Sample_Barcode"], dtype=str
    )
    return sorted(df["Tumor_Sample_Barcode"].dropna().unique())


CANDIDATE_COLUMNS = ["project", "case_id", "case_submitter_id", "barcode", "file_id"]


def collect_candidates(
    manifest: pd.DataFrame,
    barcodes_by_file: dict[str, list[str]],
    log_duplicates: bool = False,
) -> pd.DataFrame:
    """One row per (case, tumour barcode) found in the MAFs listed in ``manifest``.

    ``manifest`` needs file_id, project, case_id, case_submitter_id. ``barcodes_by_file`` maps
    file_id to the barcodes in that MAF. The barcode's patient part must match
    ``case_submitter_id`` or ValueError.

    A barcode seen in several files is kept once, under the lowest file_id. With
    ``log_duplicates=True`` the other files' rows are also returned, with
    ``duplicate_of_file_id`` set (blank on normal rows); :func:`select_samples` gives them the
    reason ``duplicate_barcode_other_file``. Without it the duplicates are dropped silently
    (the older behaviour, kept so existing tables stay byte-identical).
    """
    rows = []
    for r in manifest.sort_values("file_id").itertuples(index=False):
        found = barcodes_by_file.get(r.file_id, [])
        # A MAF with no mutation rows is kept in the log with a blank barcode.
        for bc in found or [""]:
            if bc and bc[:12] != r.case_submitter_id:
                raise ValueError(f"Barcode {bc} does not belong to case {r.case_submitter_id}")
            rows.append(
                {
                    "project": r.project,
                    "case_id": r.case_id,
                    "case_submitter_id": r.case_submitter_id,
                    "barcode": bc,
                    "file_id": r.file_id,
                    "duplicate_of_file_id": "",
                }
            )
    df = pd.DataFrame(rows, columns=[*CANDIDATE_COLUMNS, "duplicate_of_file_id"])
    real = df["barcode"] != ""
    dup = real & df.duplicated(["case_id", "barcode"], keep="first")
    first = df[real & ~dup].set_index(["case_id", "barcode"])["file_id"]
    df.loc[dup, "duplicate_of_file_id"] = [
        first[(c, b)] for c, b in zip(df.loc[dup, "case_id"], df.loc[dup, "barcode"], strict=True)
    ]
    if not log_duplicates:
        df = df[~dup]
        return df[CANDIDATE_COLUMNS].reset_index(drop=True)
    return df.reset_index(drop=True)


def maf_content_hash(path: str | Path, barcode: str) -> str:
    """sha256 of the sorted mutation rows of one tumour barcode in a MAF (comments skipped)."""
    import hashlib

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    header = lines[0].rstrip("\n").split("\t")
    col = header.index("Tumor_Sample_Barcode")
    rows = sorted(ln for ln in lines[1:] if ln.rstrip("\n").split("\t")[col] == barcode)
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def check_duplicate_contents(
    candidates: pd.DataFrame, hash_fn, strict: bool = True
) -> pd.DataFrame:
    """Compare duplicate-barcode rows with the file they duplicate.

    ``candidates`` comes from ``collect_candidates(..., log_duplicates=True)``;
    ``hash_fn(file_id, barcode)`` returns a content hash. Returns one row per duplicate with
    ``content_identical``. If any differ: ValueError when ``strict``, else they are only flagged
    (the caller must report them).
    """
    dups = candidates[candidates["duplicate_of_file_id"] != ""].copy()
    dups["content_identical"] = [
        hash_fn(r.file_id, r.barcode) == hash_fn(r.duplicate_of_file_id, r.barcode)
        for r in dups.itertuples(index=False)
    ]
    bad = dups[~dups["content_identical"]]
    if strict and len(bad):
        raise ValueError(
            f"{len(bad)} duplicate barcode(s) differ in content: {bad['barcode'].tolist()}"
        )
    return dups


def select_samples(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply D2 to a candidates table; return the samples table (one row per candidate).

    ``reason`` is one of: chosen; not_chosen_same_type_higher_barcode (same type as the chosen
    sample, higher barcode); not_preferred_type (a different type from the chosen one);
    dropped_no_01_or_06 (the patient has neither 01 nor 06, so nothing is chosen);
    empty_maf_no_barcode (the MAF has no mutation rows, so no barcode is known; never chosen);
    duplicate_barcode_other_file (same barcode as a row from a lower file_id; only present when
    the candidates were built with ``log_duplicates=True``; never chosen).
    """
    df = candidates.copy()
    dup_mask = (
        df["duplicate_of_file_id"] != ""
        if "duplicate_of_file_id" in df
        else pd.Series(False, index=df.index)
    )
    df["sample_type_code"] = df["barcode"].map(lambda b: sample_type_code(b) if b else "")
    df["sample_type_label"] = df["sample_type_code"].map(
        lambda c: sample_type_label(c) if c else ""
    )
    df["chosen"] = False
    df["reason"] = ""
    df.loc[dup_mask, "reason"] = "duplicate_barcode_other_file"
    for _, idx in df[~dup_mask].groupby("case_id", sort=True).groups.items():
        empty = df.loc[idx, "barcode"] == ""
        df.loc[empty[empty].index, "reason"] = "empty_maf_no_barcode"
        idx = empty[~empty].index
        if len(idx) == 0:
            continue
        sub = df.loc[idx]
        chosen_type = next(
            (t for t in PREFERRED_TYPES if (sub["sample_type_code"] == t).any()), None
        )
        if chosen_type is None:
            df.loc[idx, "reason"] = "dropped_no_01_or_06"
            continue
        of_type = sub[sub["sample_type_code"] == chosen_type]
        best = of_type["barcode"].min()
        for i, r in sub.iterrows():
            if r["barcode"] == best:
                df.loc[i, ["chosen", "reason"]] = [True, "chosen"]
            elif r["sample_type_code"] == chosen_type:
                df.loc[i, "reason"] = "not_chosen_same_type_higher_barcode"
            else:
                df.loc[i, "reason"] = "not_preferred_type"
    df = df.sort_values(["case_id", "barcode"]).reset_index(drop=True)
    return df[SAMPLES_COLUMNS]


# --------------------------------------------------------------------------------------
# D3: smoking groups
# --------------------------------------------------------------------------------------

#: Normalised (lower case, single spaces) GDC status string -> GDC code.
SMOKING_CODES = {
    "lifelong non-smoker": 1,
    "current smoker": 2,
    "current reformed smoker for > 15 yrs": 3,
    "current reformed smoker for < or = 15 yrs": 4,
    "current reformed smoker, duration not specified": 5,
}
SMOKING_CODE_LABELS = {
    1: "lifelong non-smoker",
    2: "current smoker",
    3: "reformed >15 y",
    4: "reformed <=15 y",
    5: "reformed, duration not specified",
}
UNINFORMATIVE = {"", "not reported", "unknown", "not allowed to collect", "none"}


def normalise_status(value: object) -> str:
    """Lower-case, whitespace-collapsed status; None, NaN and blank all become ""."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def smoking_code(value: object) -> int | None:
    """GDC tobacco_smoking_status string -> code 1-5, or None if not a known informative value."""
    return SMOKING_CODES.get(normalise_status(value))


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def add_smoking_groups(clinical: pd.DataFrame) -> pd.DataFrame:
    """Add D3 columns to a clinical table (one row per case). Rule in the module docstring.

    New columns: smoking_code (Int64), smoking_label, smoking_primary_group,
    smoking_ever_group, smoking_exclusion ("" if usable, else conflict | uninformative |
    unrecognised) and numeric pack_years_smoked.
    """
    df = clinical.copy()
    norm = df["tobacco_smoking_status"].map(normalise_status)
    code = norm.map(SMOKING_CODES)
    conflict = (
        df["tobacco_status_conflict"].map(_as_bool)
        if "tobacco_status_conflict" in df
        else pd.Series(False, index=df.index)
    )
    exclusion = pd.Series("", index=df.index, dtype=object)
    exclusion[norm.isin(UNINFORMATIVE)] = "uninformative"
    exclusion[~norm.isin(UNINFORMATIVE) & code.isna()] = "unrecognised"
    exclusion[conflict] = "conflict"  # takes priority
    code = code.where(exclusion == "")
    df["smoking_code"] = code.astype("Int64")
    df["smoking_label"] = df["smoking_code"].map(
        lambda c: SMOKING_CODE_LABELS.get(int(c)) if pd.notna(c) else pd.NA
    )
    df["smoking_primary_group"] = pd.Series(pd.NA, index=df.index, dtype=object)
    df.loc[df["smoking_code"] == 1, "smoking_primary_group"] = "never"
    df.loc[df["smoking_code"].isin([2, 4]), "smoking_primary_group"] = "smoker"
    df["smoking_ever_group"] = pd.Series(pd.NA, index=df.index, dtype=object)
    df.loc[df["smoking_code"] == 1, "smoking_ever_group"] = "never"
    df.loc[df["smoking_code"].isin([2, 3, 4, 5]), "smoking_ever_group"] = "ever"
    df["smoking_exclusion"] = exclusion
    df["pack_years_smoked"] = (
        pd.to_numeric(df["pack_years_smoked"], errors="coerce")
        if "pack_years_smoked" in df
        else float("nan")
    )
    return df


def smoking_counts(clinical: pd.DataFrame) -> dict[str, object]:
    """Counts for reporting, from a table that has had :func:`add_smoking_groups` applied."""
    return {
        "cases": len(clinical),
        "by_code": clinical["smoking_code"].value_counts(dropna=True).sort_index().to_dict(),
        "primary_group": clinical["smoking_primary_group"].value_counts().to_dict(),
        "ever_group": clinical["smoking_ever_group"].value_counts().to_dict(),
        "excluded": clinical["smoking_exclusion"].replace("", pd.NA).value_counts().to_dict(),
    }


# --------------------------------------------------------------------------------------
# D4: MSI per case
# --------------------------------------------------------------------------------------

BORDERLINE_LOW = 0.15
BORDERLINE_HIGH = 0.25
_EPS = 1e-12

MSI_OUT_COLUMNS = [
    "case_id",
    "msi_status",
    "msi_score",
    "msi_strategy",
    "msi_file_id",
    "msi_borderline",
    "msi_wxs_wgs_disagree",
    "n_msi_rows",
]


def is_borderline(score: float | None) -> bool:
    """True if score is within [0.15, 0.25], ends included."""
    if score is None or pd.isna(score):
        return False
    return BORDERLINE_LOW - _EPS <= float(score) <= BORDERLINE_HIGH + _EPS


def choose_msi(msi: pd.DataFrame) -> pd.DataFrame:
    """One MSI call per case from the per-BAM table (msi.tsv). Rules in the module docstring."""
    df = msi.copy()
    df["msi_score"] = pd.to_numeric(df["msi_score"], errors="coerce")
    df["experimental_strategy"] = df["experimental_strategy"].astype(str).str.upper().str.strip()
    if "aliquot_submitter_id" not in df:
        df["aliquot_submitter_id"] = ""
    df["aliquot_submitter_id"] = df["aliquot_submitter_id"].fillna("").astype(str)
    out = []
    for case_id, g in df.groupby("case_id", sort=True):
        pool = g[g["experimental_strategy"] == "WXS"]
        strategy = "WXS"
        if pool.empty:
            pool = g[g["experimental_strategy"] == "WGS"]
            strategy = "WGS"
        if pool.empty:
            continue
        pick = pool.sort_values(["aliquot_submitter_id", "file_id"]).iloc[0]
        wxs = set(g.loc[g["experimental_strategy"] == "WXS", "msi_status"])
        wgs = set(g.loc[g["experimental_strategy"] == "WGS", "msi_status"])
        out.append(
            {
                "case_id": case_id,
                "msi_status": pick["msi_status"],
                "msi_score": pick["msi_score"],
                "msi_strategy": strategy,
                "msi_file_id": pick["file_id"],
                "msi_borderline": is_borderline(pick["msi_score"]),
                "msi_wxs_wgs_disagree": bool(wxs and wgs and wxs != wgs),
                "n_msi_rows": len(g),
            }
        )
    return pd.DataFrame(out, columns=MSI_OUT_COLUMNS)


def chosen_only(samples: pd.DataFrame) -> pd.DataFrame:
    return samples[samples["chosen"]].reset_index(drop=True)


def count_by(samples: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    return samples.groupby(list(by)).size().rename("n").reset_index()
