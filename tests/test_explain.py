"""Tests for fingerprints.explain on hand-made exposures."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from fingerprints.explain import (
    DEFAULT_CONFIG_PATH,
    DISCLAIMER,
    AetiologyError,
    build_report,
    load_aetiology,
    normalise_cancer_type,
    signature_subset_for,
)


@pytest.fixture(scope="module")
def aet():
    return load_aetiology()


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())


def test_headline_numbers_and_grouping(aet):
    exp = pd.Series({"SBS4": 40, "SBS1": 10, "SBS5": 10, "SBS2": 10, "SBS13": 5, "SBS18": 25})
    r = build_report(exp, 500, "lung", aetiology=aet)
    assert r.group_fractions.sum() == pytest.approx(1.0)
    assert r.group_fractions["tobacco"] == pytest.approx(0.40)
    assert r.group_fractions["background"] == pytest.approx(0.20)
    assert r.group_fractions["apobec"] == pytest.approx(0.15)
    assert r.group_fractions["ros"] == pytest.approx(0.25)
    assert list(r.group_fractions.index)[0] == "tobacco"
    assert "~40% tobacco smoking" in r.headline
    assert "~20% background/ageing" in r.headline
    assert "~15% APOBEC" in r.headline


def test_fractions_input_equals_counts(aet):
    a = build_report(pd.Series({"SBS4": 3, "SBS5": 1}), 300, "lung", aetiology=aet)
    b = build_report(pd.Series({"SBS4": 0.75, "SBS5": 0.25}), 300, "lung", aetiology=aet)
    assert a.group_fractions.round(9).equals(b.group_fractions.round(9))


def test_small_groups_suppressed(aet):
    exp = pd.Series({"SBS4": 96, "SBS2": 2, "SBS18": 2})
    r = build_report(exp, 500, "lung", aetiology=aet)
    assert "other small contributions" in r.headline
    assert "APOBEC" not in r.headline
    assert "oxidative" not in r.headline


def test_low_snv_warning_still_reports(aet):
    r = build_report(pd.Series({"SBS7a": 10}), 30, "melanoma", aetiology=aet)
    assert r.low_confidence
    assert any("30 SNVs" in w and "Low confidence" in w for w in r.warnings)
    assert r.headline
    ok = build_report(pd.Series({"SBS7a": 10}), 50, "melanoma", aetiology=aet)
    assert not ok.warnings
    custom = build_report(pd.Series({"SBS7a": 10}), 80, "melanoma", aetiology=aet, min_snvs=100)
    assert custom.warnings


def test_unknown_cause_reported_honestly(aet):
    exp = pd.Series({"SBS5": 60, "SBS40a": 20, "SBS1": 20})
    r = build_report(exp, 400, "colorectal", aetiology=aet)
    assert r.unknown_fraction == pytest.approx(0.80)
    text = r.to_text()
    assert "unknown cause" in text
    assert "SBS5" in text and "SBS40a" in text
    assert any("80%" in n and "unknown cause" in n for n in r.status_notes)


def test_sbs28_is_unknown_not_pole(aet):
    r = build_report(pd.Series({"SBS28": 1}), 400, "colorectal", aetiology=aet)
    assert list(r.group_fractions.index) == ["unknown_other"]
    assert r.unknown_fraction == pytest.approx(1.0)
    assert "unknown cause" in r.headline


def test_proposed_labelled(aet):
    r = build_report(pd.Series({"SBS18": 30, "SBS38": 10}), 400, "melanoma", aetiology=aet)
    joined = " ".join(r.status_notes)
    assert "SBS18" in joined and "SBS38" in joined
    assert joined.count("proposed") >= 2
    assert "proposed" in r.headline


def test_hedging_and_no_caused_by(aet):
    exp = pd.Series({s: 1 for s in aet["signatures"]})
    r = build_report(exp, 1000, "colorectal", sample_id="X1", aetiology=aet)
    text = r.to_text().lower()
    assert "consistent with" in text
    assert "most likely reflects" in text
    assert "caused by" not in text
    assert "this patient smoked" not in text
    assert "caused by" not in r.to_markdown().lower()
    assert "caused by" not in str(aet).lower()


def test_disclaimer_last(aet):
    r = build_report(pd.Series({"SBS4": 5}), 100, "lung", aetiology=aet)
    assert r.to_text().endswith(DISCLAIMER)
    assert "not a diagnosis" in DISCLAIMER
    assert "NNLS" in DISCLAIMER and "v3.4" in DISCLAIMER
    assert "not a trained" in DISCLAIMER
    assert "Alexandrov" in DISCLAIMER and "COSMIC" in DISCLAIMER
    assert r.to_markdown().rstrip().endswith(f"{DISCLAIMER}*")


def test_zero_exposure(aet):
    r = build_report(pd.Series({"SBS4": 0, "SBS5": 0}), 100, "lung", aetiology=aet)
    assert "No confident attribution" in r.headline
    assert r.group_fractions.empty
    assert r.to_text().endswith(DISCLAIMER)
    empty = build_report(pd.Series(dtype=float), 0, "lung", aetiology=aet)
    assert "No confident attribution" in empty.headline


def test_low_cosine_warning(aet):
    exp = pd.Series({"SBS4": 5})
    low = build_report(exp, 100, "lung", cosine=0.55, aetiology=aet)
    assert any("0.55" in w for w in low.warnings)
    fine = build_report(exp, 100, "lung", cosine=0.95, aetiology=aet)
    assert not fine.warnings
    assert not build_report(exp, 100, "lung", aetiology=aet).warnings


def test_unmapped_signature_does_not_crash(aet):
    r = build_report(pd.Series({"SBS99": 5, "SBS4": 5}), 100, "lung", aetiology=aet)
    assert any("SBS99" in w for w in r.warnings)


def test_every_subset_signature_in_mapping(aet, config):
    for project, sigs in config["fitting"]["signature_subsets"].items():
        for s in sigs:
            assert s in aet["signatures"], f"{s} ({project}) missing from aetiology.yaml"


def test_mapping_status_and_groups_valid(aet):
    for s, e in aet["signatures"].items():
        assert e["status"] in {"known", "proposed", "unknown"}, s
        assert e["group"] in aet["groups"], s
        assert e["citation"] and e["cosmic_aetiology"] and e["plain_english"], s
    for s in ("SBS5", "SBS28", "SBS40a"):
        assert aet["signatures"][s]["status"] == "unknown"
    for s in ("SBS18", "SBS38"):
        assert aet["signatures"][s]["status"] == "proposed"
    assert {aet["signatures"][s]["group"] for s in ("SBS14", "SBS20")} == {"mmr"}


def test_markdown_output(aet):
    r = build_report(pd.Series({"SBS7a": 8, "SBS5": 2}), 20, "melanoma", "S1", aetiology=aet)
    md = r.to_markdown()
    assert md.startswith("### ")
    assert "S1" in md and "**Warnings**" in md and "- " in md


def test_signature_subset_for(config):
    assert (
        signature_subset_for("lung", config) == config["fitting"]["signature_subsets"]["TCGA-LUAD"]
    )
    assert "SBS7a" in signature_subset_for("melanoma", config)
    assert "SBS44" in signature_subset_for("colorectal", config)
    assert signature_subset_for("TCGA-READ", config) == signature_subset_for("colorectal")


def test_cancer_type_validation(aet):
    assert normalise_cancer_type("TCGA-LUSC") == "lung"
    assert normalise_cancer_type("Melanoma") == "melanoma"
    with pytest.raises(ValueError):
        build_report(pd.Series({"SBS1": 1}), 100, "breast", aetiology=aet)


def test_load_aetiology_errors(tmp_path):
    with pytest.raises(AetiologyError, match="not found"):
        load_aetiology(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("groups: {}\nsignatures: {}\n")
    with pytest.raises(AetiologyError):
        load_aetiology(bad)
    bad.write_text("[unclosed")
    with pytest.raises(AetiologyError):
        load_aetiology(bad)
