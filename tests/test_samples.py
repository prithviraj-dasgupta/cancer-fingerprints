"""Offline tests for fingerprints.samples (fake barcodes, tiny fixtures)."""

import gzip
from pathlib import Path

import pandas as pd
import pytest

from fingerprints import samples as S

FIX = Path(__file__).parent / "fixtures" / "data_engineer"


def cand(rows):
    return pd.DataFrame(
        [
            {
                "project": "TCGA-X",
                "case_id": c,
                "case_submitter_id": b[:12],
                "barcode": b,
                "file_id": f"file-{i}",
            }
            for i, (c, b) in enumerate(rows)
        ]
    )


def reasons(df):
    return dict(zip(df["barcode"], df["reason"], strict=True))


# ---- D2 ----------------------------------------------------------------------------


def test_sample_type_code():
    assert S.sample_type_code("TCGA-AB-1234-01A-11D-0001-01") == "01"
    assert S.sample_type_code("TCGA-AB-1234-06B-11D-0001-01") == "06"
    with pytest.raises(ValueError):
        S.sample_type_code("not-a-barcode")


def test_prefers_01_over_06():
    out = S.select_samples(
        cand([("c1", "TCGA-AA-0001-06A-01D-0001-01"), ("c1", "TCGA-AA-0001-01A-01D-0001-01")])
    )
    r = reasons(out)
    assert r["TCGA-AA-0001-01A-01D-0001-01"] == "chosen"
    assert r["TCGA-AA-0001-06A-01D-0001-01"] == "not_preferred_type"


def test_falls_back_to_06_and_drops_other_types():
    out = S.select_samples(
        cand(
            [
                ("c1", "TCGA-AA-0001-06A-01D-0001-01"),
                ("c1", "TCGA-AA-0001-02A-01D-0001-01"),
                ("c2", "TCGA-AA-0002-02A-01D-0001-01"),
                ("c2", "TCGA-AA-0002-05A-01D-0001-01"),
            ]
        )
    )
    r = reasons(out)
    assert r["TCGA-AA-0001-06A-01D-0001-01"] == "chosen"
    assert r["TCGA-AA-0001-02A-01D-0001-01"] == "not_preferred_type"
    assert r["TCGA-AA-0002-02A-01D-0001-01"] == "dropped_no_01_or_06"
    assert r["TCGA-AA-0002-05A-01D-0001-01"] == "dropped_no_01_or_06"
    assert out.groupby("case_id")["chosen"].sum().to_dict() == {"c1": 1, "c2": 0}


def test_tie_break_lowest_barcode_independent_of_order():
    a = "TCGA-AA-0001-01A-11D-0100-01"
    b = "TCGA-AA-0001-01B-11D-0050-01"
    c = "TCGA-AA-0001-01A-11D-0099-01"  # lowest string
    for order in ([a, b, c], [c, b, a], [b, a, c]):
        out = S.select_samples(cand([("c1", x) for x in order]))
        r = reasons(out)
        assert r[c] == "chosen"
        assert r[a] == r[b] == "not_chosen_same_type_higher_barcode"


def test_one_row_per_candidate_and_columns():
    out = S.select_samples(cand([("c1", "TCGA-AA-0001-01A-11D-0001-01")]))
    assert list(out.columns) == S.SAMPLES_COLUMNS
    assert out.loc[0, "sample_type_label"] == "Primary Solid Tumor"


def test_read_maf_barcodes_multi_sample_and_comments(tmp_path):
    p = tmp_path / "x.maf.gz"
    text = (
        "#version gdc-1.0.0\nHugo_Symbol\tTumor_Sample_Barcode\nA\tTCGA-AA-0001-01A-11D-0001-01\n"
        "B\tTCGA-AA-0001-06A-11D-0001-01\nC\tTCGA-AA-0001-01A-11D-0001-01\n"
    )
    with gzip.open(p, "wt") as fh:
        fh.write(text)
    assert S.read_maf_barcodes(p) == [
        "TCGA-AA-0001-01A-11D-0001-01",
        "TCGA-AA-0001-06A-11D-0001-01",
    ]


def test_collect_candidates_dedupes_and_checks_case():
    manifest = pd.DataFrame(
        {
            "file_id": ["f2", "f1"],
            "project": ["TCGA-X"] * 2,
            "case_id": ["c1", "c1"],
            "case_submitter_id": ["TCGA-AA-0001"] * 2,
        }
    )
    bc = "TCGA-AA-0001-01A-11D-0001-01"
    out = S.collect_candidates(manifest, {"f1": [bc], "f2": [bc]})
    assert len(out) == 1 and out.loc[0, "file_id"] == "f1"
    with pytest.raises(ValueError):
        S.collect_candidates(manifest, {"f1": ["TCGA-ZZ-9999-01A-11D-0001-01"]})


# ---- D3 ----------------------------------------------------------------------------


@pytest.fixture
def smoking():
    return S.add_smoking_groups(pd.read_csv(FIX / "clinical_small.tsv", sep="\t", dtype=str))


def test_smoking_codes_case_insensitive(smoking):
    s = smoking.set_index("case_id")
    assert s.loc["c1", "smoking_code"] == 1
    assert s.loc["c2", "smoking_code"] == 2
    assert s.loc["c3", "smoking_code"] == 3  # upper case in the fixture
    assert s.loc["c4", "smoking_code"] == 4  # lower case in the fixture
    assert s.loc["c5", "smoking_code"] == 5


def test_smoking_groups(smoking):
    s = smoking.set_index("case_id")
    g = s["smoking_primary_group"]
    assert g["c1"] == "never"
    assert g["c2"] == g["c4"] == "smoker"
    assert g[["c3", "c5", "c6", "c7", "c8", "c9", "c10"]].isna().all()
    assert s.loc["c1", "smoking_ever_group"] == "never"
    assert all(s.loc[c, "smoking_ever_group"] == "ever" for c in ["c2", "c3", "c4", "c5"])


def test_smoking_exclusions_and_conflict(smoking):
    s = smoking.set_index("case_id")
    assert s.loc["c6", "smoking_exclusion"] == "uninformative"
    assert s.loc["c7", "smoking_exclusion"] == "uninformative"
    assert s.loc["c8", "smoking_exclusion"] == "uninformative"
    assert s.loc["c10", "smoking_exclusion"] == "unrecognised"
    # conflict: informative status but excluded from both groups
    assert s.loc["c9", "smoking_exclusion"] == "conflict"
    assert pd.isna(s.loc["c9", "smoking_code"])
    assert pd.isna(s.loc["c9", "smoking_ever_group"])


def test_pack_years_numeric_and_counts(smoking):
    assert smoking["pack_years_smoked"].dtype.kind == "f"
    assert smoking.set_index("case_id").loc["c3", "pack_years_smoked"] == 20.5
    counts = S.smoking_counts(smoking)
    assert counts["primary_group"] == {"smoker": 2, "never": 1}
    assert counts["ever_group"] == {"ever": 4, "never": 1}
    assert counts["excluded"]["conflict"] == 1


# ---- D4 ----------------------------------------------------------------------------


@pytest.fixture
def msi():
    out = S.choose_msi(pd.read_csv(FIX / "msi_small.tsv", sep="\t", dtype=str))
    return out.set_index("case_id")


def test_msi_prefers_wxs_and_flags_disagreement(msi):
    assert msi.loc["m1", "msi_strategy"] == "WXS"
    assert msi.loc["m1", "msi_status"] == "MSI"
    assert msi.loc["m1", "n_msi_rows"] == 2
    assert bool(msi.loc["m1", "msi_wxs_wgs_disagree"])


def test_msi_wgs_fallback(msi):
    assert msi.loc["m2", "msi_strategy"] == "WGS"
    assert msi.loc["m2", "msi_status"] == "MSI"
    assert not bool(msi.loc["m2", "msi_wxs_wgs_disagree"])


def test_msi_several_wxs_lowest_aliquot(msi):
    assert msi.loc["m3", "msi_file_id"] == "f8"
    assert msi.loc["m3", "msi_score"] == pytest.approx(0.10)
    assert msi.loc["m3", "n_msi_rows"] == 2


def test_msi_borderline_boundaries(msi):
    assert bool(msi.loc["m3", "msi_borderline"]) is False  # chose 0.10
    assert bool(msi.loc["m4", "msi_borderline"])  # 0.25 inclusive
    assert not bool(msi.loc["m5", "msi_borderline"])  # 0.1499
    assert not bool(msi.loc["m6", "msi_borderline"])  # 0.2501
    assert S.is_borderline(0.15) and S.is_borderline(0.25)
    assert not S.is_borderline(None) and not S.is_borderline(float("nan"))


def test_msi_no_rows_no_output():
    empty = pd.DataFrame(
        columns=["case_id", "file_id", "experimental_strategy", "msi_status", "msi_score"]
    )
    assert S.choose_msi(empty).empty


def test_empty_maf_is_logged_and_never_chosen():
    manifest = pd.DataFrame(
        {
            "file_id": ["f1", "f2"],
            "project": ["TCGA-X"] * 2,
            "case_id": ["c1", "c2"],
            "case_submitter_id": ["TCGA-AA-0001", "TCGA-AA-0002"],
        }
    )
    cands = S.collect_candidates(manifest, {"f1": ["TCGA-AA-0001-01A-11D-0001-01"], "f2": []})
    out = S.select_samples(cands)
    assert len(out) == 2
    row = out[out["case_id"] == "c2"].iloc[0]
    assert row["reason"] == "empty_maf_no_barcode" and not row["chosen"]
    assert out["chosen"].sum() == 1


def test_duplicate_barcode_logged_and_content_checked():
    manifest = pd.DataFrame(
        {
            "file_id": ["f2", "f1"],
            "project": ["TCGA-X"] * 2,
            "case_id": ["c1", "c1"],
            "case_submitter_id": ["TCGA-AA-0001"] * 2,
        }
    )
    bc = "TCGA-AA-0001-01A-11D-0001-01"
    quiet = S.collect_candidates(manifest, {"f1": [bc], "f2": [bc]})
    assert len(quiet) == 1  # default keeps the old behaviour
    cands = S.collect_candidates(manifest, {"f1": [bc], "f2": [bc]}, log_duplicates=True)
    out = S.select_samples(cands)
    assert out["chosen"].sum() == 1
    assert dict(zip(out["file_id"], out["reason"], strict=True)) == {
        "f1": "chosen",
        "f2": "duplicate_barcode_other_file",
    }
    same = S.check_duplicate_contents(cands, lambda f, b: "h")
    assert same["content_identical"].all()
    with pytest.raises(ValueError, match="differ"):
        S.check_duplicate_contents(cands, lambda f, b: f)
    flagged = S.check_duplicate_contents(cands, lambda f, b: f, strict=False)
    assert not flagged["content_identical"].any()


def test_maf_content_hash(tmp_path):
    def write(name, rows):
        p = tmp_path / name
        with gzip.open(p, "wt") as fh:
            fh.write("#v\nHugo_Symbol\tTumor_Sample_Barcode\n" + rows)
        return p

    a = write("a.maf.gz", "A\tb1\nB\tb1\nC\tb2\n")
    b = write("b.maf.gz", "B\tb1\nA\tb1\nZ\tb2\n")
    assert S.maf_content_hash(a, "b1") == S.maf_content_hash(b, "b1")  # row order ignored
    assert S.maf_content_hash(a, "b2") != S.maf_content_hash(b, "b2")


def test_normalise_status_blank_and_nan():
    for v in [None, float("nan"), pd.NA, "", "   "]:
        assert S.normalise_status(v) == ""
    assert S.normalise_status("  Current   Smoker ") == "current smoker"
    df = pd.DataFrame(
        {"tobacco_smoking_status": [None, float("nan"), "nan"], "pack_years_smoked": [None] * 3}
    )
    out = S.add_smoking_groups(df)
    assert out["smoking_exclusion"].tolist() == ["uninformative", "uninformative", "unrecognised"]
