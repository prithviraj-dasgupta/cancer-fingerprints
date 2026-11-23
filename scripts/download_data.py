#!/usr/bin/env python
"""Download TCGA open-access masked somatic MAFs and clinical tables from the GDC.

For each configured project this script:

1. queries the GDC ``/files`` endpoint for open-access "Masked Somatic Mutation" MAFs
   (GRCh38, Aliquot Ensemble Somatic Variant Merging and Masking workflow);
2. writes a manifest TSV (file_id, file_name, project, case_id, md5, size, ...) to
   ``data/raw/maf/<project>/manifest.tsv``. With ``--limit`` it writes
   ``manifest.limited.tsv`` instead, so a quick test never overwrites the full manifest;
3. downloads each MAF into ``data/raw/maf/<project>/`` unless it is already present with
   the expected size and md5 (so re-running is safe and cheap);
4. fetches clinical data from the ``/cases`` endpoint with demographic, diagnoses and
   exposures expanded. It saves the raw JSON and a flat TSV with one row per case;
5. for projects with ``fetch_msi: true`` (colorectal), saves the GDC MSI status/score
   metadata to ``data/raw/clinical/<project>/msi.tsv`` (skip with ``--skip-msi``;
   ``--skip-clinical`` skips only the cases table).

Later pipeline steps should list MAFs from ``manifest.tsv``, not by globbing the
directory: the folder can hold leftovers from earlier runs or ``--limit`` tests, and the
manifest carries the case IDs and md5s.

Examples::

    uv run python scripts/download_data.py --dry-run --limit 1 --projects TCGA-LUAD
    uv run python scripts/download_data.py --projects TCGA-SKCM --limit 5
    uv run python scripts/download_data.py            # everything in config/config.yaml

The HTTP-free logic (filter building, response parsing, skip decisions, flattening) lives in
small pure functions so it can be tested offline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"

log = logging.getLogger("download_data")

# Fields requested from /files for each MAF.
FILE_FIELDS = [
    "file_id",
    "file_name",
    "md5sum",
    "file_size",
    "access",
    "data_format",
    "analysis.workflow_type",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
]

# Fields requested from /files for MSI metadata (on tumour Aligned Reads files).
MSI_FIELDS = [
    "file_id",
    "experimental_strategy",
    "msi_status",
    "msi_score",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "associated_entities.entity_submitter_id",
    "associated_entities.entity_type",
]

# Stable sort orders for pagination.
FILES_SORT = "file_id:asc"
CASES_SORT = "case_id:asc"

# Per-file download attempts (on top of the session's HTTP-level retries).
DOWNLOAD_ATTEMPTS = 3

CLINICAL_EXPAND = "demographic,diagnoses,exposures,project"


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestRow:
    file_id: str
    file_name: str
    project: str
    case_id: str
    case_submitter_id: str
    md5: str
    size: int
    workflow_type: str


MANIFEST_COLUMNS = [f.name for f in fields(ManifestRow)]

CLINICAL_COLUMNS = [
    "case_id",
    "case_submitter_id",
    "project",
    "primary_site",
    "disease_type",
    # demographic
    "gender",
    "race",
    "ethnicity",
    "vital_status",
    "age_at_index",
    "days_to_death",
    # primary diagnosis
    "primary_diagnosis",
    "age_at_diagnosis_days",
    "ajcc_pathologic_stage",
    "tissue_or_organ_of_origin",
    "morphology",
    "icd_10_code",
    "site_of_resection_or_biopsy",
    "tumor_of_origin",
    # tobacco exposure
    "tobacco_smoking_status",
    "pack_years_smoked",
    "cigarettes_per_day",
    "years_smoked",
    "tobacco_smoking_onset_year",
    "tobacco_smoking_quit_year",
    "tobacco_status_conflict",
    "n_exposure_records",
    "n_diagnosis_records",
]

MSI_COLUMNS = [
    "case_id",
    "case_submitter_id",
    "project",
    "file_id",
    "experimental_strategy",
    "sample_submitter_id",
    "sample_type",
    "aliquot_submitter_id",
    "msi_status",
    "msi_score",
]


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def project_ids(config: dict[str, Any]) -> list[str]:
    return [p["id"] for p in config["projects"]]


def project_settings(config: dict[str, Any], project: str) -> dict[str, Any]:
    for p in config["projects"]:
        if p["id"] == project:
            return p
    raise KeyError(f"Project {project!r} is not in the config")


# --------------------------------------------------------------------------------------
# Pure functions: filters and parsing
# --------------------------------------------------------------------------------------


def _in(field: str, values: str | list[str]) -> dict[str, Any]:
    if isinstance(values, str):
        values = [values]
    return {"op": "in", "content": {"field": field, "value": list(values)}}


def build_maf_filters(project: str, maf_filters: dict[str, str]) -> dict[str, Any]:
    """Build the GDC /files filter for open masked somatic MAFs in one project.

    ``maf_filters`` maps short keys (data_category, data_type, data_format, access,
    workflow_type) to values, as in ``config.yaml``. Missing keys are left out.
    """
    field_map = {
        "data_category": "data_category",
        "data_type": "data_type",
        "data_format": "data_format",
        "access": "access",
        "workflow_type": "analysis.workflow_type",
        "experimental_strategy": "experimental_strategy",
    }
    content = [_in("cases.project.project_id", project)]
    for key, field in field_map.items():
        if maf_filters.get(key):
            content.append(_in(field, maf_filters[key]))
    unknown = set(maf_filters) - set(field_map)
    if unknown:
        raise ValueError(f"Unknown maf_filters keys: {sorted(unknown)}")
    return {"op": "and", "content": content}


def build_case_filters(project: str) -> dict[str, Any]:
    """GDC /cases filter for all cases in one project."""
    return _in("project.project_id", project)


def build_msi_filters(project: str) -> dict[str, Any]:
    """GDC /files filter for files in a project that carry an msi_status value."""
    return {
        "op": "and",
        "content": [
            _in("cases.project.project_id", project),
            {"op": "not", "content": {"field": "msi_status"}},
        ],
    }


def build_query(
    filters: dict[str, Any],
    fields_: list[str] | None = None,
    size: int = 500,
    from_: int = 0,
    expand: str | None = None,
    sort: str | None = None,
) -> dict[str, Any]:
    """Body for a POST to a GDC search endpoint.

    ``sort`` (e.g. ``"file_id:asc"``) should always be set when paginating: without a stable
    sort order, from/size pages can overlap or skip hits.
    """
    body: dict[str, Any] = {"filters": filters, "size": size, "from": from_, "format": "JSON"}
    if fields_:
        body["fields"] = ",".join(fields_)
    if expand:
        body["expand"] = expand
    if sort:
        body["sort"] = sort
    return body


class IncompleteResultError(RuntimeError):
    """Pagination returned a different number of unique records than the API total."""


def dedupe_and_check(
    items: list[Any], key, total: int, expect_complete: bool, what: str
) -> list[Any]:
    """Drop duplicate records (by ``key(item)``), keeping the first one.

    If ``expect_complete`` (no --limit), raise IncompleteResultError unless the number of
    unique records equals the API's reported ``total``.
    """
    seen: set[str] = set()
    unique = []
    for item in items:
        k = key(item)
        if k in seen:
            continue
        seen.add(k)
        unique.append(item)
    if len(unique) != len(items):
        log.warning("%s: dropped %d duplicate records", what, len(items) - len(unique))
    if expect_complete and len(unique) != total:
        raise IncompleteResultError(f"{what}: got {len(unique)} unique records, API total {total}")
    return unique


def parse_file_hits(response: dict[str, Any], project: str) -> list[ManifestRow]:
    """Turn a GDC /files JSON response into manifest rows.

    Masked somatic MAFs are per tumour aliquot, so each hit should have exactly one case.
    Hits with several cases are kept, with case IDs joined by ';' and a warning logged.
    """
    rows = []
    for hit in response.get("data", {}).get("hits", []):
        cases = hit.get("cases") or []
        if len(cases) != 1:
            log.warning("File %s has %d cases", hit.get("file_id"), len(cases))
        rows.append(
            ManifestRow(
                file_id=hit["file_id"],
                file_name=hit["file_name"],
                project=project,
                case_id=";".join(c.get("case_id", "") for c in cases),
                case_submitter_id=";".join(c.get("submitter_id", "") for c in cases),
                md5=hit["md5sum"],
                size=int(hit["file_size"]),
                workflow_type=(hit.get("analysis") or {}).get("workflow_type", ""),
            )
        )
    return rows


def pagination_total(response: dict[str, Any]) -> int:
    return int(response.get("data", {}).get("pagination", {}).get("total", 0))


def _primary_diagnosis(diagnoses: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the diagnosis record for the TCGA primary tumour.

    Newer GDC releases attach several diagnosis records to a case (e.g. prior malignancies
    or placeholders). Preference order: diagnosis_is_primary_disease is True, then
    classification_of_tumor == "primary", then the first record.
    """
    if not diagnoses:
        return {}
    for d in diagnoses:
        if d.get("diagnosis_is_primary_disease") is True:
            return d
    for d in diagnoses:
        if str(d.get("classification_of_tumor", "")).lower() == "primary":
            return d
    return diagnoses[0]


#: tobacco_smoking_status values that carry no information (compared case-insensitively).
UNINFORMATIVE_STATUS = {"", "not reported", "unknown", "not allowed to collect", "none"}


def _status_key(value: Any) -> str:
    return str(value or "").strip().lower()


def is_informative_status(value: Any) -> bool:
    return _status_key(value) not in UNINFORMATIVE_STATUS


def _tobacco_exposure(exposures: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the exposure record with tobacco data.

    Preference: first record with an informative tobacco_smoking_status (not "Not Reported" or
    "Unknown"), then first record with any status, then first with exposure_type 'Tobacco',
    else an empty dict.
    """
    for e in exposures:
        if is_informative_status(e.get("tobacco_smoking_status")):
            return e
    for e in exposures:
        if e.get("tobacco_smoking_status"):
            return e
    for e in exposures:
        if _status_key(e.get("exposure_type")) == "tobacco":
            return e
    return {}


def tobacco_status_conflict(exposures: list[dict[str, Any]]) -> bool:
    """True if a case's exposure records hold more than one distinct informative status."""
    statuses = {
        _status_key(e.get("tobacco_smoking_status"))
        for e in exposures
        if is_informative_status(e.get("tobacco_smoking_status"))
    }
    return len(statuses) > 1


def flatten_case(case: dict[str, Any]) -> dict[str, Any]:
    """Flatten one GDC /cases hit (with expand) into a single clinical row."""
    demo = case.get("demographic") or {}
    diag = _primary_diagnosis(case.get("diagnoses") or [])
    exposures = case.get("exposures") or []
    tob = _tobacco_exposure(exposures)
    project = (case.get("project") or {}).get("project_id", "")
    return {
        "case_id": case.get("case_id", case.get("id", "")),
        "case_submitter_id": case.get("submitter_id", ""),
        "project": project,
        "primary_site": case.get("primary_site", ""),
        "disease_type": case.get("disease_type", ""),
        # GDC renamed demographic.gender to sex_at_birth; accept either.
        "gender": demo.get("gender") or demo.get("sex_at_birth"),
        "race": demo.get("race"),
        "ethnicity": demo.get("ethnicity"),
        "vital_status": demo.get("vital_status"),
        "age_at_index": demo.get("age_at_index"),
        "days_to_death": demo.get("days_to_death"),
        "primary_diagnosis": diag.get("primary_diagnosis"),
        "age_at_diagnosis_days": diag.get("age_at_diagnosis"),
        "ajcc_pathologic_stage": diag.get("ajcc_pathologic_stage"),
        "tissue_or_organ_of_origin": diag.get("tissue_or_organ_of_origin"),
        "morphology": diag.get("morphology"),
        "icd_10_code": diag.get("icd_10_code"),
        # Both come from the same diagnosis record chosen by _primary_diagnosis() above.
        "site_of_resection_or_biopsy": diag.get("site_of_resection_or_biopsy"),
        "tumor_of_origin": diag.get("tumor_of_origin"),
        "tobacco_smoking_status": tob.get("tobacco_smoking_status"),
        "pack_years_smoked": tob.get("pack_years_smoked"),
        "cigarettes_per_day": tob.get("cigarettes_per_day"),
        "years_smoked": tob.get("years_smoked"),
        "tobacco_smoking_onset_year": tob.get("tobacco_smoking_onset_year"),
        "tobacco_smoking_quit_year": tob.get("tobacco_smoking_quit_year"),
        "tobacco_status_conflict": tobacco_status_conflict(exposures),
        "n_exposure_records": len(exposures),
        "n_diagnosis_records": len(case.get("diagnoses") or []),
    }


def parse_msi_hits(response: dict[str, Any], project: str) -> list[dict[str, Any]]:
    """Turn a GDC /files response (MSI query) into one row per file.

    GDC labels files "MSI" or "MSS" using MSIsensor2 (tumour BAM only) with a documented MSI
    score cutoff of 20% (GDC DNA-Seq pipeline docs, "Microsatellite Instability Detection";
    see docs/decisions.md D4). The data agree (2026-11-22: lowest MSI score 0.209, highest MSS
    score 0.1995 in COAD/READ). A case can have several rows (WXS and WGS BAMs), and these
    can disagree. All rows are kept; choosing between them is a Phase 2 decision.

    On the /files endpoint, ``cases.samples`` and ``associated_entities`` hold only the
    sample/aliquot the BAM was made from (checked live: exactly one each per file), so
    sample_type is the type of that specific sample.
    """
    rows = []
    for hit in response.get("data", {}).get("hits", []):
        cases = hit.get("cases") or [{}]
        case = cases[0]
        samples = case.get("samples") or []
        if len(samples) != 1:
            log.warning("MSI file %s has %d samples", hit.get("file_id"), len(samples))
        aliquots = [
            a.get("entity_submitter_id", "")
            for a in hit.get("associated_entities") or []
            if a.get("entity_type", "aliquot") == "aliquot"
        ]
        rows.append(
            {
                "case_id": case.get("case_id", ""),
                "case_submitter_id": case.get("submitter_id", ""),
                "project": project,
                "file_id": hit.get("file_id", ""),
                "experimental_strategy": hit.get("experimental_strategy", ""),
                "sample_submitter_id": ";".join(s.get("submitter_id", "") for s in samples),
                "sample_type": ";".join(s.get("sample_type", "") for s in samples),
                "aliquot_submitter_id": ";".join(aliquots),
                "msi_status": hit.get("msi_status", ""),
                "msi_score": hit.get("msi_score"),
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# Pure functions: local file decisions
# --------------------------------------------------------------------------------------


def md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def needs_download(
    path: Path, expected_size: int, expected_md5: str | None, verify_md5: bool = True
) -> tuple[bool, str]:
    """Decide whether a file must be (re)downloaded.

    Returns (should_download, reason). Reasons: 'missing', 'size mismatch', 'md5 mismatch',
    'ok'.
    """
    if not path.exists():
        return True, "missing"
    if path.stat().st_size != expected_size:
        return True, "size mismatch"
    if verify_md5 and expected_md5 and md5_of(path) != expected_md5:
        return True, "md5 mismatch"
    return False, "ok"


def maf_destination(raw_maf_dir: Path, row: ManifestRow) -> Path:
    # file_name comes from the API. Use only its basename so it cannot escape the directory.
    return raw_maf_dir / row.project / Path(row.file_name).name


def write_tsv(rows: Iterable[dict[str, Any]], path: Path, columns: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=columns, delimiter="\t", extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
            n += 1
    tmp.replace(path)
    return n


def read_manifest(path: Path) -> list[ManifestRow]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [
            ManifestRow(**{**r, "size": int(r["size"])})  # type: ignore[arg-type]
            for r in reader
        ]


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


def make_session(retries: int = 5, backoff_factor: float = 1.0):
    """requests.Session that retries on connection errors, 429 and 5xx responses."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "cancer-fingerprints-downloader"})
    return session


class GDCClient:
    def __init__(self, api_url: str, session, timeout: tuple[float, float], page_size: int):
        self.api_url = api_url.rstrip("/")
        self.session = session
        self.timeout = timeout
        self.page_size = page_size

    def search(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        r = self.session.post(f"{self.api_url}/{endpoint}", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def search_all(
        self,
        endpoint: str,
        filters: dict[str, Any],
        fields_: list[str] | None = None,
        expand: str | None = None,
        limit: int | None = None,
        sort: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield response pages until all hits (or ``limit`` hits) are returned."""
        fetched = 0
        while True:
            size = self.page_size if limit is None else min(self.page_size, limit - fetched)
            if size <= 0:
                return
            resp = self.search(endpoint, build_query(filters, fields_, size, fetched, expand, sort))
            hits = resp.get("data", {}).get("hits", [])
            yield resp
            fetched += len(hits)
            if not hits or fetched >= pagination_total(resp):
                return

    def download(self, file_id: str, dest: Path, expected_md5: str | None) -> None:
        """Stream /data/<file_id> to dest via a .part file, checking the md5."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        h = hashlib.md5()
        try:
            with self.session.get(
                f"{self.api_url}/data/{file_id}", stream=True, timeout=self.timeout
            ) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
                        h.update(chunk)
            if expected_md5 and h.hexdigest() != expected_md5:
                raise OSError(
                    f"md5 mismatch for {file_id}: got {h.hexdigest()}, want {expected_md5}"
                )
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(dest)


def download_with_retries(
    client: GDCClient, row: ManifestRow, dest: Path, attempts: int = DOWNLOAD_ATTEMPTS
) -> str | None:
    """Try to download one file up to ``attempts`` times. Return None or the last error.

    requests exceptions subclass OSError, so network, stream and md5 failures are all caught.
    """
    error = None
    for attempt in range(1, attempts + 1):
        try:
            client.download(row.file_id, dest, row.md5)
            return None
        except OSError as exc:
            error = str(exc)
            log.warning("  attempt %d/%d failed for %s: %s", attempt, attempts, dest.name, exc)
    return error


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def fetch_maf_manifest(
    client: GDCClient, project: str, maf_filters: dict[str, str], limit: int | None
) -> tuple[list[ManifestRow], int]:
    filters = build_maf_filters(project, maf_filters)
    rows: list[ManifestRow] = []
    total = 0
    for page in client.search_all("files", filters, FILE_FIELDS, limit=limit, sort=FILES_SORT):
        total = pagination_total(page)
        rows.extend(parse_file_hits(page, project))
    rows = dedupe_and_check(rows, lambda r: r.file_id, total, limit is None, f"{project} MAF files")
    return rows, total


def fetch_cases(client: GDCClient, project: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    total = 0
    pages = client.search_all(
        "cases", build_case_filters(project), expand=CLINICAL_EXPAND, sort=CASES_SORT
    )
    for page in pages:
        total = pagination_total(page)
        cases.extend(page.get("data", {}).get("hits", []))
    return dedupe_and_check(
        cases, lambda c: c.get("case_id", c.get("id")), total, True, f"{project} cases"
    )


def fetch_msi(client: GDCClient, project: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    pages = client.search_all("files", build_msi_filters(project), MSI_FIELDS, sort=FILES_SORT)
    for page in pages:
        total = pagination_total(page)
        rows.extend(parse_msi_hits(page, project))
    return dedupe_and_check(rows, lambda r: r["file_id"], total, True, f"{project} MSI files")


def process_project(
    client: GDCClient,
    config: dict[str, Any],
    project: str,
    out_root: Path,
    limit: int | None,
    dry_run: bool,
    skip_clinical: bool,
    verify_md5: bool,
    skip_msi: bool = False,
) -> dict[str, Any]:
    paths = config["paths"]
    raw_maf = out_root / paths["raw_maf"]
    raw_clin = out_root / paths["raw_clinical"]
    summary: dict[str, Any] = {"project": project}

    rows, total = fetch_maf_manifest(client, project, config["gdc"]["maf_filters"], limit)
    summary["maf_files_available"] = total
    summary["maf_files_selected"] = len(rows)
    log.info("%s: %d MAF files match (%d selected)", project, total, len(rows))

    if dry_run:
        for r in rows[:5]:
            log.info("  would fetch %s  %s  %d bytes", r.file_id, r.file_name, r.size)
    else:
        name = "manifest.tsv" if limit is None else "manifest.limited.tsv"
        manifest_path = raw_maf / project / name
        write_tsv((asdict(r) for r in rows), manifest_path, MANIFEST_COLUMNS)
        n_down = n_skip = 0
        failed: list[dict[str, str]] = []
        for r in rows:
            dest = maf_destination(raw_maf, r)
            should, reason = needs_download(dest, r.size, r.md5, verify_md5=verify_md5)
            if not should:
                n_skip += 1
                continue
            log.info("  downloading %s (%s)", dest.name, reason)
            error = download_with_retries(client, r, dest)
            if error is None:
                n_down += 1
            else:
                failed.append({"file_id": r.file_id, "error": error})
        summary.update(
            downloaded=n_down,
            skipped=n_skip,
            failed=len(failed),
            failed_files=failed,
            manifest=str(manifest_path),
        )
        log.info(
            "%s: downloaded %d, already present %d, failed %d",
            project,
            n_down,
            n_skip,
            len(failed),
        )

    if not skip_clinical:
        if dry_run:
            first = client.search(
                "cases",
                build_query(build_case_filters(project), size=1, expand=CLINICAL_EXPAND),
            )
            summary["cases_available"] = pagination_total(first)
            hits = first.get("data", {}).get("hits", [])
            if hits:
                flat = flatten_case(hits[0])
                log.info(
                    "%s: %d cases; first case %s smoking status=%r",
                    project,
                    summary["cases_available"],
                    flat["case_submitter_id"],
                    flat["tobacco_smoking_status"],
                )
        else:
            cases = fetch_cases(client, project)
            clin_dir = raw_clin / project
            clin_dir.mkdir(parents=True, exist_ok=True)
            with open(clin_dir / "cases.json", "w") as fh:
                json.dump(cases, fh, indent=1)
            n = write_tsv(
                (flatten_case(c) for c in cases), clin_dir / "clinical.tsv", CLINICAL_COLUMNS
            )
            summary["cases"] = n
            log.info("%s: wrote clinical data for %d cases", project, n)

    if not skip_msi and project_settings(config, project).get("fetch_msi"):
        msi_rows = fetch_msi(client, project)
        summary["msi_records"] = len(msi_rows)
        summary["msi_cases"] = len({r["case_id"] for r in msi_rows})
        summary["msi_status_counts"] = {
            s: sum(r["msi_status"] == s for r in msi_rows)
            for s in sorted({r["msi_status"] for r in msi_rows})
        }
        if dry_run:
            log.info("%s: %d files with MSI status", project, len(msi_rows))
        else:
            write_tsv(msi_rows, raw_clin / project / "msi.tsv", MSI_COLUMNS)
            log.info("%s: wrote %d MSI records", project, len(msi_rows))

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    p.add_argument(
        "--projects", nargs="+", help="GDC project IDs (default: all projects in the config)"
    )
    p.add_argument("--limit", type=int, help="At most N MAF files per project (quick tests)")
    p.add_argument("--dry-run", action="store_true", help="Query the API and report, write nothing")
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT,
        help="Root that config paths are resolved against (default: repo root)",
    )
    p.add_argument(
        "--skip-clinical", action="store_true", help="Do not fetch the clinical cases table"
    )
    p.add_argument(
        "--rebuild-clinical",
        action="store_true",
        help="Offline: rewrite clinical.tsv from the saved cases.json files; no network, "
        "no MAF or manifest changes",
    )
    p.add_argument("--skip-msi", action="store_true", help="Do not fetch MSI metadata (colorectal)")
    p.add_argument(
        "--no-verify-md5",
        action="store_true",
        help="Skip existing files on size alone (faster, less safe)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    config = load_config(args.config)
    known = project_ids(config)
    projects = args.projects or known
    unknown = [p for p in projects if p not in known]
    if unknown:
        log.error("Projects not in config: %s (known: %s)", unknown, known)
        return 2
    if args.limit is not None and args.limit < 1:
        log.error("--limit must be >= 1")
        return 2

    if args.rebuild_clinical:
        raw_clin = args.out / config["paths"]["raw_clinical"]
        for project in projects:
            with open(raw_clin / project / "cases.json") as fh:
                cases = json.load(fh)
            n = write_tsv(
                (flatten_case(c) for c in cases),
                raw_clin / project / "clinical.tsv",
                CLINICAL_COLUMNS,
            )
            log.info("%s: rebuilt clinical.tsv for %d cases", project, n)
        return 0

    http = config["gdc"]["http"]
    client = GDCClient(
        config["gdc"]["api_url"],
        make_session(http["retries"], http["backoff_factor"]),
        timeout=(http["connect_timeout"], http["read_timeout"]),
        page_size=http["page_size"],
    )
    summaries = []
    for project in projects:
        summaries.append(
            process_project(
                client,
                config,
                project,
                args.out,
                args.limit,
                args.dry_run,
                args.skip_clinical,
                verify_md5=not args.no_verify_md5,
                skip_msi=args.skip_msi,
            )
        )
    print(json.dumps(summaries, indent=2))
    n_failed = sum(s.get("failed", 0) for s in summaries)
    if n_failed:
        log.error("%d file(s) failed to download; re-run to retry them", n_failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
