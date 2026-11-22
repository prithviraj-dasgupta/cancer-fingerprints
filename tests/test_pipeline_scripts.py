"""Offline tests for scripts/build_profiles.py and scripts/run_fits.py (tiny fixtures only)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fingerprints.profile import SBS96_CHANNELS

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MAF = Path(__file__).parent / "fixtures" / "profile_small.maf"
B1 = "TCGA-FP-0001-01A-11D-A000-08"  # kept 7 (exclude) / 9 (count_as_sbs)
B2 = "TCGA-FP-0002-01A-11D-A000-08"  # kept 9 / 12
B3 = "TCGA-FP-0003-01A-11D-A000-08"  # kept 0 (two indel rows only)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bp = _load("build_profiles")
rf = _load("run_fits")


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A fake repo root: config, manifest, samples table, one MAF holding three barcodes."""
    maf_dir = tmp_path / "data" / "raw" / "maf" / "TCGA-FAKE"
    maf_dir.mkdir(parents=True)
    shutil.copy(FIXTURE_MAF, maf_dir / "fileA.maf")
    pd.DataFrame({"file_id": ["id-A", "id-B"], "file_name": ["sub/fileA.maf", "fileB.maf"]}).to_csv(
        maf_dir / "manifest.tsv", sep="\t", index=False
    )
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        {
            "project": "TCGA-FAKE",
            "case_submitter_id": ["C1", "C2", "C3", "C4"],
            "barcode": [B2, B1, B3, "NOT-CHOSEN"],
            "file_id": ["id-A", "id-A", "id-A", "id-B"],
            "chosen": [True, True, True, False],
        }
    ).to_csv(tables / "samples_TCGA-FAKE.tsv", sep="\t", index=False)
    return tmp_path


@pytest.fixture
def config() -> dict:
    return {
        "paths": {"raw_maf": "data/raw/maf", "processed": "data/processed"},
        "projects": [{"id": "TCGA-FAKE"}],
        "profile": {"adjacent_snvs": "exclude"},
        "fitting": {
            "min_snvs": 5,
            "min_snvs_sensitivity": [2, 8],
            "exclude_artefacts": ["SBS27"],
            "signature_subsets": {"TCGA-FAKE": ["SBS1", "SBS5", "SBS18"]},
        },
    }


# ---- build_profiles ---------------------------------------------------------------------


def test_mode_suffix():
    assert bp.mode_suffix("exclude") == ""
    assert bp.mode_suffix("count_as_sbs") == "_count_as_sbs"


def test_chosen_samples_accepts_bool_and_strings():
    df = pd.DataFrame({"barcode": ["a", "b", "c"], "chosen": ["True", "False", "true"]})
    assert list(bp.chosen_samples(df)["barcode"]) == ["a", "c"]
    df2 = pd.DataFrame({"barcode": ["a", "b"], "chosen": [True, False]})
    assert list(bp.chosen_samples(df2)["barcode"]) == ["a"]


def test_chosen_file_path_uses_manifest_basename(tmp_path):
    manifest = pd.DataFrame({"file_id": ["x", "y"], "file_name": ["sub/a.maf.gz", "b.maf.gz"]})
    assert bp.chosen_file_path("x", manifest, tmp_path) == tmp_path / "a.maf.gz"
    with pytest.raises(ValueError, match="found 0 times"):
        bp.chosen_file_path("missing", manifest, tmp_path)


def test_restrict_to_barcode():
    maf = pd.read_csv(FIXTURE_MAF, sep="\t", comment="#", dtype=str)
    sub = bp.restrict_to_barcode(maf, B2)
    assert len(sub) == 19
    assert set(sub["Tumor_Sample_Barcode"]) == {B2}
    with pytest.raises(ValueError, match="not present"):
        bp.restrict_to_barcode(maf, "TCGA-NOPE")


def test_build_project_counts_exact(fake_root, config):
    matrix, drops = bp.build_project("TCGA-FAKE", config, fake_root, "exclude")
    assert list(matrix.index) == SBS96_CHANNELS
    assert list(matrix.columns) == [B2, B1, B3]  # samples-table order, NOT-CHOSEN absent
    assert matrix.sum().to_dict() == {B2: 9, B1: 7, B3: 0}
    d = drops.set_index("tumour")
    assert d.loc[B1, ["kept", "not_snp", "tri_allelic", "adjacent_snv_dbs", "total"]].tolist() == [
        7,
        1,
        1,
        2,
        13,
    ]
    assert d.loc[B1, "dbs_events"] == 1 and d.loc[B2, "mnv_events"] == 1
    assert (d["project"] == "TCGA-FAKE").all()


def test_build_project_count_as_sbs(fake_root, config):
    matrix, drops = bp.build_project("TCGA-FAKE", config, fake_root, "count_as_sbs")
    assert matrix.sum().to_dict() == {B2: 12, B1: 9, B3: 0}
    assert drops["adjacent_snv_dbs"].sum() == 0
    assert drops["dbs_events"].sum() == 1  # events are reported in both modes


def test_aggregate_drops_percentages(fake_root, config):
    _, drops = bp.build_project("TCGA-FAKE", config, fake_root, "exclude")
    s = bp.aggregate_drops(drops, "TCGA-FAKE")
    assert s["n_tumours"] == 3 and s["total_rows"] == 34 and s["not_snp"] == 4
    assert s["snp_rows"] == 30 and s["kept_snvs"] == 16
    assert s["n_zero_snv_tumours"] == 1
    assert s["median_snvs"] == 7
    assert s["adjacent_snv_dbs"] == 5
    assert s["pct_adjacent_snv_dbs"] == pytest.approx(100 * 5 / 30)
    assert s["pct_not_snp_of_all_rows"] == pytest.approx(100 * 4 / 34)
    assert s["pct_kept_of_snp_rows"] == pytest.approx(100 * 16 / 30)
    assert s["dbs_events"] == 1 and s["mnv_events"] == 1


def test_read_adjacent_mode():
    assert bp.read_adjacent_mode({"profile": {"adjacent_snvs": "count_as_sbs"}}) == "count_as_sbs"
    assert bp.read_adjacent_mode({}) == "exclude"


def test_build_profiles_main_writes_files(fake_root, config, caplog):
    cfg = fake_root / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    assert bp.main(["--config", str(cfg), "--root", str(fake_root)]) == 0
    assert (fake_root / "data/processed/sbs96_TCGA-FAKE.tsv").is_file()
    assert (fake_root / "results/tables/profile_drops_TCGA-FAKE.tsv").is_file()
    assert (fake_root / "results/tables/profile_drops_summary.tsv").is_file()
    args = ["--config", str(cfg), "--root", str(fake_root), "--adjacent-snvs", "count_as_sbs"]
    assert bp.main(args) == 0
    assert (fake_root / "data/processed/sbs96_TCGA-FAKE_count_as_sbs.tsv").is_file()
    assert (fake_root / "results/tables/profile_drops_summary_count_as_sbs.tsv").is_file()


# ---- run_fits ---------------------------------------------------------------------------


def test_output_suffix_and_profile_path(tmp_path):
    assert rf.output_suffix("exclude", None) == ""
    assert rf.output_suffix("count_as_sbs", None) == "_count_as_sbs"
    assert rf.output_suffix("exclude", "SBS18") == "_noSBS18"
    assert rf.output_suffix("count_as_sbs", "SBS18") == "_count_as_sbs_noSBS18"
    assert rf.profile_path(tmp_path, "P", "exclude") == tmp_path / "sbs96_P.tsv"
    assert rf.profile_path(tmp_path, "P", "count_as_sbs") == tmp_path / "sbs96_P_count_as_sbs.tsv"


def test_config_without_signature(config):
    cfg = rf.config_without_signature(config, "TCGA-FAKE", "SBS18")
    assert cfg["fitting"]["signature_subsets"]["TCGA-FAKE"] == ["SBS1", "SBS5"]
    assert config["fitting"]["signature_subsets"]["TCGA-FAKE"] == ["SBS1", "SBS5", "SBS18"]
    with pytest.raises(ValueError, match="not in the"):
        rf.config_without_signature(config, "TCGA-FAKE", "SBS4")
    same = rf.config_without_signature(config, "TCGA-FAKE", None)
    assert same == config and same is not config


def _fake_reference_and_matrix():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame(rng.random((96, 3)), index=SBS96_CHANNELS, columns=["SBS1", "SBS5", "SBS18"])
    ref = ref / ref.sum()
    totals = {"t10": 10, "t30": 30, "t4": 4}  # t4 is below min_snvs 5
    mat = pd.DataFrame(
        {n: np.round(ref["SBS1"].to_numpy() * k) for n, k in totals.items()},
        index=SBS96_CHANNELS,
    )
    return ref, mat


def test_fit_project_writes_main_and_all_files(tmp_path, config, monkeypatch):
    ref, mat = _fake_reference_and_matrix()
    monkeypatch.setattr(rf, "ALL_FLOOR", 2)
    row = rf.fit_project("TCGA-FAKE", config, mat, ref, tmp_path, "_noX")
    totals = mat.sum()
    main_q = pd.read_csv(tmp_path / "fit_qc_TCGA-FAKE_noX.tsv", sep="\t", index_col=0)
    all_q = pd.read_csv(tmp_path / "fit_qc_TCGA-FAKE_noX_all2.tsv", sep="\t", index_col=0)
    assert set(main_q.index) == {n for n in mat if totals[n] >= 5}
    assert set(all_q.index) == {n for n in mat if totals[n] >= 2}
    assert (all_q["total_snvs"] == totals[all_q.index]).all()
    assert "total_snvs" in main_q.columns
    for name in [
        "exposures_TCGA-FAKE_noX.tsv",
        "exposures_TCGA-FAKE_noX_fractions.tsv",
        "fit_summary_TCGA-FAKE_noX.tsv",
        "excluded_TCGA-FAKE_noX.tsv",
        "exposures_TCGA-FAKE_noX_all2.tsv",
        "exposures_TCGA-FAKE_noX_all2_fractions.tsv",
    ]:
        assert (tmp_path / name).is_file(), name
    assert row["project"] == "TCGA-FAKE_noX"
    assert row["n_excluded_lt20"] == 2 and row["n_excluded_lt50"] == 3
    assert row["n_fitted_main"] == int((totals >= 5).sum())


def test_mean_fractions(tmp_path):
    pd.DataFrame({"SBS1": [1.0, 0.5], "SBS5": [0.0, 0.5]}, index=["a", "b"]).to_csv(
        tmp_path / "exposures_P1_fractions.tsv", sep="\t"
    )
    pd.DataFrame({"SBS1": [0.2], "SBS7a": [0.8]}, index=["c"]).to_csv(
        tmp_path / "exposures_P2_fractions.tsv", sep="\t"
    )
    mf = rf.mean_fractions(tmp_path, ["P1", "P2", "P3"])
    assert list(mf.columns) == ["P1", "P2"]
    assert mf.loc["SBS1", "P1"] == pytest.approx(0.75)
    assert mf.loc["SBS5", "P2"] == 0.0 and mf.loc["SBS7a", "P2"] == pytest.approx(0.8)
