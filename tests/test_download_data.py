"""Offline tests for scripts/download_data.py (no network unless marked)."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "data_engineer"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


dd = _load_script("download_data")

MAF_FILTERS = {
    "data_category": "Simple Nucleotide Variation",
    "data_type": "Masked Somatic Mutation",
    "data_format": "MAF",
    "access": "open",
    "workflow_type": "Aliquot Ensemble Somatic Variant Merging and Masking",
}


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def _cases() -> list[dict]:
    return _load("gdc_cases_response.json")["data"]["hits"]


# ---- config -------------------------------------------------------------------------


def test_config_lists_all_v1_projects():
    cfg = dd.load_config()
    assert dd.project_ids(cfg) == ["TCGA-LUAD", "TCGA-LUSC", "TCGA-SKCM", "TCGA-COAD", "TCGA-READ"]
    assert cfg["genome_build"] == "GRCh38"
    assert dd.build_maf_filters("TCGA-LUAD", cfg["gdc"]["maf_filters"])  # keys are all known
    assert dd.project_settings(cfg, "TCGA-COAD")["fetch_msi"] is True


# ---- filters ------------------------------------------------------------------------


def test_build_maf_filters_structure():
    f = dd.build_maf_filters("TCGA-LUAD", MAF_FILTERS)
    assert f["op"] == "and"
    by_field = {c["content"]["field"]: c["content"]["value"] for c in f["content"]}
    assert by_field == {
        "cases.project.project_id": ["TCGA-LUAD"],
        "data_category": ["Simple Nucleotide Variation"],
        "data_type": ["Masked Somatic Mutation"],
        "data_format": ["MAF"],
        "access": ["open"],
        "analysis.workflow_type": ["Aliquot Ensemble Somatic Variant Merging and Masking"],
    }
    assert all(c["op"] == "in" for c in f["content"])
    json.dumps(f)  # must be serialisable


def test_build_maf_filters_rejects_unknown_key():
    with pytest.raises(ValueError):
        dd.build_maf_filters("TCGA-LUAD", {"data_typo": "x"})


def test_build_msi_filters_uses_not_missing():
    f = dd.build_msi_filters("TCGA-COAD")
    assert {"op": "not", "content": {"field": "msi_status"}} in f["content"]


def test_build_query_body():
    body = dd.build_query(
        {"op": "in"}, ["file_id", "md5sum"], size=10, from_=20, expand="x", sort="file_id:asc"
    )
    assert body == {
        "filters": {"op": "in"},
        "size": 10,
        "from": 20,
        "format": "JSON",
        "fields": "file_id,md5sum",
        "expand": "x",
        "sort": "file_id:asc",
    }


# ---- parsing ------------------------------------------------------------------------


def test_parse_file_hits_to_manifest_rows():
    resp = _load("gdc_files_response.json")
    rows = dd.parse_file_hits(resp, "TCGA-LUAD")
    assert len(rows) == 2
    r = rows[0]
    assert r.file_id == "00000000-0000-0000-0000-000000000001"
    assert r.file_name == "aaaa.wxs.aliquot_ensemble_masked.maf.gz"
    assert r.project == "TCGA-LUAD"
    assert r.case_id == "case-0001"
    assert r.case_submitter_id == "TCGA-XX-0001"
    assert r.md5 == "64f7830d48cf22f8849e962b7f518ee5"
    assert r.size == 245502
    assert r.workflow_type == "Aliquot Ensemble Somatic Variant Merging and Masking"
    assert dd.pagination_total(resp) == 618


def test_parse_file_hits_joins_multiple_cases():
    resp = _load("gdc_files_response.json")
    hit = resp["data"]["hits"][0]
    hit["cases"].append({"case_id": "case-9999", "submitter_id": "TCGA-XX-9999"})
    row = dd.parse_file_hits(resp, "TCGA-LUAD")[0]
    assert row.case_id == "case-0001;case-9999"
    assert row.case_submitter_id == "TCGA-XX-0001;TCGA-XX-9999"


def test_manifest_roundtrip(tmp_path):
    rows = dd.parse_file_hits(_load("gdc_files_response.json"), "TCGA-LUAD")
    path = tmp_path / "m" / "manifest.tsv"
    assert dd.write_tsv((asdict(r) for r in rows), path, dd.MANIFEST_COLUMNS) == 2
    header = path.read_text().splitlines()[0].split("\t")
    assert header[:6] == ["file_id", "file_name", "project", "case_id", "case_submitter_id", "md5"]
    assert dd.read_manifest(path) == rows


def test_flatten_case_picks_primary_diagnosis_and_tobacco_exposure():
    row = dd.flatten_case(_cases()[0])
    assert row["case_submitter_id"] == "TCGA-XX-0001"
    assert row["project"] == "TCGA-LUAD"
    assert row["primary_diagnosis"] == "Adenocarcinoma, NOS"
    assert row["age_at_diagnosis_days"] == 26538
    assert row["tobacco_smoking_status"] == "Current Reformed Smoker for < or = 15 yrs"
    assert row["pack_years_smoked"] == 40.0
    assert row["n_exposure_records"] == 2
    assert row["n_diagnosis_records"] == 2
    assert row["tobacco_status_conflict"] is False
    assert set(row) == set(dd.CLINICAL_COLUMNS)


def test_flatten_case_without_exposures():
    row = dd.flatten_case(_cases()[1])
    assert row["tobacco_smoking_status"] is None
    assert row["n_exposure_records"] == 0
    assert row["tobacco_status_conflict"] is False
    assert row["primary_diagnosis"] == "Malignant melanoma, NOS"  # falls back to first record


def test_flatten_case_skips_not_reported_and_flags_conflict():
    row = dd.flatten_case(_cases()[2])
    # "Not Reported" comes first but is uninformative; the next informative record wins.
    assert row["tobacco_smoking_status"] == "Current Smoker"
    assert row["pack_years_smoked"] == 50
    # "Current Smoker" vs "Lifelong Non-Smoker" disagree.
    assert row["tobacco_status_conflict"] is True
    assert row["n_exposure_records"] == 3


def test_primary_diagnosis_classification_fallback():
    # No diagnosis_is_primary_disease flag: classification_of_tumor "Primary" (any case) wins.
    assert dd.flatten_case(_cases()[2])["primary_diagnosis"] == "Squamous cell carcinoma, NOS"
    assert dd._primary_diagnosis([]) == {}


def test_tobacco_exposure_fallbacks():
    only_not_reported = [{"exposure_type": "Tobacco", "tobacco_smoking_status": "Not Reported"}]
    assert dd._tobacco_exposure(only_not_reported) is only_not_reported[0]
    by_type = [{"exposure_type": "Alcohol"}, {"exposure_type": "TOBACCO", "pack_years_smoked": 3}]
    assert dd._tobacco_exposure(by_type) is by_type[1]
    assert dd._tobacco_exposure([{"exposure_type": "Alcohol"}]) == {}
    # Case-insensitive comparison: same status in two spellings is not a conflict.
    same = [
        {"tobacco_smoking_status": "Current Smoker"},
        {"tobacco_smoking_status": "current smoker"},
    ]
    assert dd.tobacco_status_conflict(same) is False
    assert dd.tobacco_status_conflict(only_not_reported + same) is False


def test_parse_msi_hits():
    rows = dd.parse_msi_hits(_load("gdc_msi_response.json"), "TCGA-COAD")
    assert [r["msi_status"] for r in rows] == ["MSI", "MSS"]
    assert rows[0]["case_submitter_id"] == "TCGA-YY-0101"
    assert rows[0]["sample_submitter_id"] == "TCGA-YY-0101-01A"
    assert rows[0]["sample_type"] == "Primary Tumor"
    assert rows[0]["aliquot_submitter_id"] == "TCGA-YY-0101-01A-11D-A000-10"
    assert rows[1]["sample_type"] == "Metastatic"
    assert rows[1]["experimental_strategy"] == "WGS"
    assert rows[1]["msi_score"] == pytest.approx(0.03)
    assert set(rows[0]) == set(dd.MSI_COLUMNS)


def test_dedupe_and_check():
    items = [{"k": "a"}, {"k": "b"}, {"k": "a"}]
    assert dd.dedupe_and_check(items, lambda x: x["k"], 2, True, "t") == items[:2]
    with pytest.raises(dd.IncompleteResultError):
        dd.dedupe_and_check(items, lambda x: x["k"], 3, True, "t")
    assert len(dd.dedupe_and_check(items, lambda x: x["k"], 3, False, "t")) == 2


# ---- skip logic ---------------------------------------------------------------------


def test_needs_download_missing(tmp_path):
    assert dd.needs_download(tmp_path / "nope", 10, "x") == (True, "missing")


def test_needs_download_size_and_md5(tmp_path):
    p = tmp_path / "f.maf.gz"
    data = b"hello maf"
    p.write_bytes(data)
    good_md5 = hashlib.md5(data).hexdigest()
    assert dd.needs_download(p, len(data), good_md5) == (False, "ok")
    assert dd.needs_download(p, len(data) + 1, good_md5) == (True, "size mismatch")
    assert dd.needs_download(p, len(data), "0" * 32) == (True, "md5 mismatch")
    assert dd.needs_download(p, len(data), "0" * 32, verify_md5=False) == (False, "ok")


def test_maf_destination_strips_directories(tmp_path):
    row = dd.ManifestRow("id", "../../evil.maf.gz", "TCGA-LUAD", "c", "s", "m", 1, "w")
    assert dd.maf_destination(tmp_path, row) == tmp_path / "TCGA-LUAD" / "evil.maf.gz"


# ---- pagination ---------------------------------------------------------------------


def _file_hit(i: int) -> dict:
    return {
        "file_id": f"f{i:04d}",
        "file_name": f"f{i:04d}.maf.gz",
        "md5sum": "0" * 32,
        "file_size": 1,
        "analysis": {"workflow_type": "w"},
        "cases": [{"case_id": f"c{i:04d}", "submitter_id": f"TCGA-XX-{i:04d}"}],
    }


class PagingClient(dd.GDCClient):
    """Fake /files endpoint that slices a fixed list by from/size and records requests."""

    def __init__(self, hits: list[dict], total: int | None = None, page_size: int = 500):
        super().__init__("https://example.invalid", None, (1, 1), page_size)
        self.hits = hits
        self.total = len(hits) if total is None else total
        self.bodies: list[dict] = []

    def search(self, endpoint, body):
        self.bodies.append(body)
        start, size = body["from"], body["size"]
        return {
            "data": {"hits": self.hits[start : start + size], "pagination": {"total": self.total}}
        }


def test_pagination_fetches_all_unique_rows():
    client = PagingClient([_file_hit(i) for i in range(618)])
    rows, total = dd.fetch_maf_manifest(client, "TCGA-LUAD", MAF_FILTERS, limit=None)
    assert total == 618
    assert len(rows) == 618 and len({r.file_id for r in rows}) == 618
    assert [(b["from"], b["size"]) for b in client.bodies] == [(0, 500), (500, 500)]
    assert all(b["sort"] == "file_id:asc" for b in client.bodies)


def test_pagination_respects_limit():
    client = PagingClient([_file_hit(i) for i in range(618)])
    rows, total = dd.fetch_maf_manifest(client, "TCGA-LUAD", MAF_FILTERS, limit=3)
    assert total == 618 and [r.file_id for r in rows] == ["f0000", "f0001", "f0002"]
    assert len(client.bodies) == 1 and client.bodies[0]["size"] == 3


def test_pagination_detects_duplicates_and_short_results():
    hits = [_file_hit(i) for i in range(500)] + [_file_hit(i) for i in range(118)]  # overlap
    with pytest.raises(dd.IncompleteResultError):
        dd.fetch_maf_manifest(PagingClient(hits), "TCGA-LUAD", MAF_FILTERS, limit=None)
    short = PagingClient([_file_hit(i) for i in range(600)], total=618)
    with pytest.raises(dd.IncompleteResultError):
        dd.fetch_maf_manifest(short, "TCGA-LUAD", MAF_FILTERS, limit=None)


# ---- GDCClient.download with a fake session -----------------------------------------


class FakeResponse:
    def __init__(self, chunks: list[bytes], fail_midway: bool = False):
        self.chunks = chunks
        self.fail_midway = fail_midway

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i, c in enumerate(self.chunks):
            if self.fail_midway and i == 1:
                raise ConnectionError("stream broke")
            yield c


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.calls = 0

    def get(self, url, stream, timeout):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


def _client(session) -> dd.GDCClient:
    return dd.GDCClient("https://example.invalid", session, (1, 1), 500)


def test_download_success_renames_part(tmp_path):
    data = [b"abc", b"def"]
    dest = tmp_path / "x" / "f.maf.gz"
    md5 = hashlib.md5(b"abcdef").hexdigest()
    _client(FakeSession([FakeResponse(data)])).download("id", dest, md5)
    assert dest.read_bytes() == b"abcdef"
    assert not (dest.parent / "f.maf.gz.part").exists()


def test_download_md5_mismatch_deletes_part(tmp_path):
    dest = tmp_path / "f.maf.gz"
    with pytest.raises(OSError, match="md5 mismatch"):
        _client(FakeSession([FakeResponse([b"abc"])])).download("id", dest, "0" * 32)
    assert not dest.exists()
    assert not (tmp_path / "f.maf.gz.part").exists()


def test_download_stream_error_deletes_part(tmp_path):
    dest = tmp_path / "f.maf.gz"
    with pytest.raises(ConnectionError):
        _client(FakeSession([FakeResponse([b"a", b"b"], fail_midway=True)])).download(
            "id", dest, None
        )
    assert not (tmp_path / "f.maf.gz.part").exists()


def test_download_with_retries_recovers_then_reports_failure(tmp_path):
    good = [b"abc"]
    md5 = hashlib.md5(b"abc").hexdigest()
    row = dd.ManifestRow("id", "f.maf.gz", "P", "c", "s", md5, 3, "w")
    # First attempt breaks mid-stream, second succeeds.
    session = FakeSession([FakeResponse([b"a", b"b"], fail_midway=True), FakeResponse(good)])
    assert dd.download_with_retries(_client(session), row, tmp_path / "f.maf.gz") is None
    assert session.calls == 2
    # Always wrong content: all attempts fail and the error is returned.
    bad = FakeSession([FakeResponse([b"zzz"])])
    err = dd.download_with_retries(_client(bad), row, tmp_path / "g.maf.gz", attempts=3)
    assert "md5 mismatch" in err and bad.calls == 3


# ---- orchestration with a fake client -----------------------------------------------


class FakeClient(dd.GDCClient):
    def __init__(self, files_resp: dict, fail_ids: set[str] | None = None):
        super().__init__("https://example.invalid", session=None, timeout=(1, 1), page_size=500)
        self.files_resp = files_resp
        self.fail_ids = fail_ids or set()
        self.downloads: list[str] = []

    def search(self, endpoint, body):
        resp = copy.deepcopy(self.files_resp)
        resp["data"]["hits"] = resp["data"]["hits"][: body["size"]]
        return resp

    def download(self, file_id, dest, expected_md5):
        self.downloads.append(file_id)
        if file_id in self.fail_ids:
            raise OSError("boom")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"")


def test_dry_run_writes_nothing(tmp_path):
    cfg = dd.load_config()
    client = FakeClient(_load("gdc_files_response.json"))
    summary = dd.process_project(
        client, cfg, "TCGA-LUAD", tmp_path, 1, True, skip_clinical=True, verify_md5=True
    )
    assert summary["maf_files_available"] == 618
    assert summary["maf_files_selected"] == 1
    assert client.downloads == []
    assert not any(tmp_path.iterdir())


def _single_file_resp(data: bytes) -> dict:
    resp = _load("gdc_files_response.json")
    resp["data"]["hits"] = resp["data"]["hits"][:1]
    resp["data"]["hits"][0].update(file_size=len(data), md5sum=hashlib.md5(data).hexdigest())
    resp["data"]["pagination"]["total"] = 1
    return resp


def test_existing_valid_file_is_skipped(tmp_path):
    cfg = dd.load_config()
    data = b"x" * 10
    resp = _single_file_resp(data)
    dest = tmp_path / cfg["paths"]["raw_maf"] / "TCGA-LUAD" / resp["data"]["hits"][0]["file_name"]
    dest.parent.mkdir(parents=True)
    dest.write_bytes(data)

    client = FakeClient(resp)
    summary = dd.process_project(
        client, cfg, "TCGA-LUAD", tmp_path, None, False, skip_clinical=True, verify_md5=True
    )
    assert summary["skipped"] == 1 and summary["downloaded"] == 0 and summary["failed"] == 0
    assert client.downloads == []
    assert (dest.parent / "manifest.tsv").is_file()


def test_limit_writes_limited_manifest_only(tmp_path):
    cfg = dd.load_config()
    full = tmp_path / cfg["paths"]["raw_maf"] / "TCGA-LUAD" / "manifest.tsv"
    full.parent.mkdir(parents=True)
    full.write_text("sentinel\n")
    dd.process_project(
        FakeClient(_load("gdc_files_response.json")),
        cfg,
        "TCGA-LUAD",
        tmp_path,
        1,
        False,
        skip_clinical=True,
        verify_md5=True,
    )
    assert full.read_text() == "sentinel\n"  # untouched
    limited = dd.read_manifest(full.parent / "manifest.limited.tsv")
    assert len(limited) == 1


def test_failed_download_is_counted_and_main_exits_nonzero(tmp_path, monkeypatch):
    cfg = dd.load_config()
    resp = _single_file_resp(b"abc")
    fid = resp["data"]["hits"][0]["file_id"]
    client = FakeClient(resp, fail_ids={fid})
    summary = dd.process_project(
        client, cfg, "TCGA-LUAD", tmp_path, None, False, skip_clinical=True, verify_md5=True
    )
    assert summary["failed"] == 1 and summary["failed_files"][0]["file_id"] == fid
    assert client.downloads == [fid] * dd.DOWNLOAD_ATTEMPTS

    monkeypatch.setattr(dd, "GDCClient", lambda *a, **k: FakeClient(resp, fail_ids={fid}))
    monkeypatch.setattr(dd, "make_session", lambda *a, **k: None)
    rc = dd.main(["--projects", "TCGA-LUAD", "--skip-clinical", "--out", str(tmp_path / "o")])
    assert rc == 1


def test_main_rejects_unknown_project_and_bad_limit():
    assert dd.main(["--projects", "TCGA-NOPE", "--dry-run"]) == 2
    assert dd.main(["--limit", "0", "--dry-run"]) == 2


# ---- live API (opt-in) --------------------------------------------------------------


@pytest.mark.network
def test_live_gdc_filters_return_hits():
    cfg = dd.load_config()
    client = dd.GDCClient(cfg["gdc"]["api_url"], dd.make_session(2, 0.5), (10, 60), 500)
    rows, total = dd.fetch_maf_manifest(client, "TCGA-LUAD", cfg["gdc"]["maf_filters"], limit=1)
    assert total > 500 and len(rows) == 1
    assert rows[0].file_name.endswith(".maf.gz") and rows[0].case_submitter_id.startswith("TCGA-")
    msi = dd.fetch_msi(client, "TCGA-READ")
    assert {r["msi_status"] for r in msi} <= {"MSI", "MSS"} and len(msi) > 100
