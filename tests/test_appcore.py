"""Tests for the app logic and the bundled synthetic example MAF."""

from __future__ import annotations

import gzip

import pandas as pd
import pytest
import yaml

from fingerprints import appcore, fit, profile

CONFIG = appcore.load_config()
COSMIC = appcore.reference_path(CONFIG)
needs_cosmic = pytest.mark.skipif(
    not COSMIC.is_file(), reason="COSMIC reference not downloaded (data/raw/signatures)"
)


def test_classify_upload():
    assert appcore.classify_upload("x.MAF") == "maf"
    assert appcore.classify_upload("x.maf.gz") == "maf"
    assert appcore.classify_upload("x.tsv") == "maf"
    assert appcore.classify_upload("x.vcf") == "vcf"
    assert appcore.classify_upload("X.VCF.GZ") == "vcf"
    assert appcore.classify_upload("x.xlsx") == "unsupported"
    assert "not supported yet" in appcore.VCF_MESSAGE and "CONTEXT" in appcore.VCF_MESSAGE


def test_read_upload_plain_and_gzip():
    data = appcore.EXAMPLE_MAF.read_bytes()
    plain = appcore.read_upload(data, "a.maf")
    zipped = appcore.read_upload(gzip.compress(data), "a.maf.gz")
    assert len(plain) == len(zipped) == 400
    assert "CONTEXT" in plain.columns


def test_maf_without_context_gives_clear_error():
    maf = appcore.read_upload(appcore.EXAMPLE_MAF.read_bytes(), "a.maf").drop(columns="CONTEXT")
    with pytest.raises(ValueError, match="CONTEXT"):
        profile.build_sbs96_matrix(maf)


def test_profile_warnings_reports_drops_and_adjacent_events():
    row = pd.Series({r: 0 for r in profile.DROP_REASONS} | {"kept": 8, "total": 12})
    row["not_snp"], row["adjacent_snv_dbs"] = 2, 2
    row["dbs_events"], row["mnv_events"] = 1, 0
    row["dbs_events"] = 1
    msgs = appcore.profile_warnings(row)
    assert "4 of 12" in msgs[0] and "not snp" in msgs[0]
    assert "1 likely doublet" in msgs[1]
    clean = pd.Series({r: 0 for r in profile.DROP_REASONS} | {"kept": 5, "total": 5})
    clean["dbs_events"] = clean["mnv_events"] = 0
    assert appcore.profile_warnings(clean) == []


def test_reference_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(appcore.REFERENCE_ENV, str(tmp_path / "x.txt"))
    assert appcore.reference_path(CONFIG) == tmp_path / "x.txt"
    monkeypatch.delenv(appcore.REFERENCE_ENV)
    assert appcore.reference_path(CONFIG).name == CONFIG["signatures"]["file"]


# --- the bundled synthetic example ------------------------------------------------------------


def test_example_maf_is_synthetic_and_parses():
    text = appcore.EXAMPLE_MAF.read_text()
    assert text.startswith("#SYNTHETIC")
    assert appcore.EXAMPLE_MAF.stat().st_size < 100_000
    maf = profile.read_maf(appcore.EXAMPLE_MAF)
    assert "CONTEXT" in maf.columns
    assert set(maf["Tumor_Sample_Barcode"]) == {"SYNTHETIC-0001"}
    assert set(maf["Hugo_Symbol"]) == {"Unknown"}
    matrix, summary = profile.build_sbs96_matrix(maf, return_dropped=True)
    assert int(matrix.sum().sum()) == 400
    assert int(summary.loc["SYNTHETIC-0001", "total"]) == 400  # nothing dropped, none adjacent
    assert summary.loc["SYNTHETIC-0001", ["dbs_events", "mnv_events"]].sum() == 0


@needs_cosmic
def test_example_round_trip_recovers_planted_mix():
    matrix = profile.build_sbs96_matrix(appcore.EXAMPLE_MAF)
    reference = fit.load_reference(COSMIC)
    sigs = CONFIG["fitting"]["signature_subsets"]["TCGA-SKCM"]
    subset = fit.subset_reference(reference, sigs, CONFIG["fitting"]["exclude_artefacts"])
    _, fractions, cosine = fit.fit_exposures(matrix, subset)
    f = fractions.iloc[0]
    uv = f[["SBS7a", "SBS7b", "SBS7c", "SBS7d"]].sum()  # planted 0.75
    clock = f[["SBS1", "SBS5"]].sum()  # planted 0.25
    assert uv == pytest.approx(0.75, abs=0.12)
    assert clock == pytest.approx(0.25, abs=0.12)
    assert f["SBS7a"] > f["SBS7b"] > 0
    assert cosine.iloc[0] > 0.9


def test_analyse_sample_with_fake_reference(fake_reference_file):
    matrix, summary = profile.build_sbs96_matrix(appcore.EXAMPLE_MAF, return_dropped=True)
    reference = fit.load_reference(fake_reference_file)
    result = appcore.analyse_sample(
        matrix, summary, "SYNTHETIC-0001", "melanoma", CONFIG, reference
    )
    assert result.n_snvs == 400
    assert set(result.exposures.index) == set(CONFIG["fitting"]["signature_subsets"]["TCGA-SKCM"])
    assert result.report.group_fractions.sum() == pytest.approx(1.0, abs=1e-6)
    assert "SYNTHETIC-0001" in result.exposures_tsv() or "signature" in result.exposures_tsv()
    result.warnings.append("extra profile warning")
    assert "extra profile warning" in result.full_markdown()
    assert "extra profile warning" not in result.body_markdown()
    with pytest.raises(ValueError, match="cancer_type"):
        appcore.analyse_sample(matrix, summary, "SYNTHETIC-0001", "brain", CONFIG, reference)
    with pytest.raises(ValueError, match="not in the matrix"):
        appcore.analyse_sample(matrix, summary, "nope", "lung", CONFIG, reference)


def test_config_is_a_mapping(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump([1, 2]))
    with pytest.raises(ValueError, match="mapping"):
        appcore.load_config(p)
