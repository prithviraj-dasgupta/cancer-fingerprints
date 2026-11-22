"""Offline tests for fingerprints.fit using a tiny fake reference (never the COSMIC file)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fingerprints import fit
from fingerprints.profile import SBS96_CHANNELS

SIGS = ["SBS1", "SBS2", "SBS3", "SBS4", "SBS5"]


@pytest.fixture
def reference() -> pd.DataFrame:
    """96 x 5 synthetic signatures with near-disjoint support, columns sum to 1."""
    rng = np.random.default_rng(0)
    data = np.full((96, 5), 0.001)
    for j in range(5):
        data[j * 19 : j * 19 + 19, j] += rng.uniform(0.5, 1.5, 19)
    data /= data.sum(axis=0)
    return pd.DataFrame(data, index=SBS96_CHANNELS, columns=SIGS)


def mixture(reference: pd.DataFrame, counts: dict[str, int]) -> pd.Series:
    """Exact integer-ish tumour: sum of count * signature (kept float; NNLS is exact)."""
    profile = sum(reference[s] * n for s, n in counts.items())
    return profile.rename("T")


def test_nnls_recovers_exact_mixture(reference):
    truth = {"SBS1": 300, "SBS4": 100}
    matrix = pd.DataFrame({"T1": mixture(reference, truth)})
    expo, frac, cos = fit.fit_exposures(matrix, reference)
    for s in SIGS:
        assert expo.loc["T1", s] == pytest.approx(truth.get(s, 0), abs=1e-6)
    assert frac.loc["T1", "SBS1"] == pytest.approx(0.75)
    assert frac.loc["T1", "SBS4"] == pytest.approx(0.25)
    assert cos["T1"] == pytest.approx(1.0)
    assert (expo >= 0).all().all()


def test_integer_counts_with_sampling_noise_close(reference):
    rng = np.random.default_rng(1)
    p = (reference["SBS2"] * 0.6 + reference["SBS5"] * 0.4).to_numpy()
    counts = rng.multinomial(5000, p / p.sum())
    matrix = pd.DataFrame({"T1": counts}, index=SBS96_CHANNELS)
    expo, frac, cos = fit.fit_exposures(matrix, reference)
    assert frac.loc["T1", "SBS2"] == pytest.approx(0.6, abs=0.05)
    assert frac.loc["T1", "SBS5"] == pytest.approx(0.4, abs=0.05)
    assert expo.loc["T1"].sum() == pytest.approx(5000, rel=0.02)
    assert cos["T1"] > 0.99


def test_rows_matched_by_label_not_position(reference):
    matrix = pd.DataFrame({"T1": mixture(reference, {"SBS3": 200, "SBS5": 50})})
    expo, _, _ = fit.fit_exposures(matrix, reference)
    shuffled_matrix = matrix.sample(frac=1, random_state=3)
    shuffled_ref = reference.sample(frac=1, random_state=4)
    expo2, _, _ = fit.fit_exposures(shuffled_matrix, shuffled_ref)
    pd.testing.assert_frame_equal(expo, expo2)


def test_bad_row_labels_raise(reference):
    matrix = pd.DataFrame({"T1": mixture(reference, {"SBS1": 100})})
    bad = matrix.rename(index={SBS96_CHANNELS[0]: "X[C>A]X"})
    with pytest.raises(fit.FitError, match="96 SBS channels"):
        fit.fit_exposures(bad, reference)
    with pytest.raises(fit.FitError):
        fit.fit_exposures(matrix.iloc[:90], reference)


def test_load_reference_reorders_by_label(tmp_path, reference):
    path = tmp_path / "ref.txt"
    reference.sort_index().rename_axis("Type").to_csv(path, sep="\t")
    loaded = fit.load_reference(path)
    assert list(loaded.index) == SBS96_CHANNELS
    pd.testing.assert_frame_equal(loaded, reference, check_names=False)


def test_load_reference_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_signatures"):
        fit.load_reference(tmp_path / "none.txt")


def test_subset_reference(reference):
    sub = fit.subset_reference(reference, ["SBS4", "SBS1"], ["SBS3"])
    assert list(sub.columns) == ["SBS4", "SBS1"]


def test_subset_reference_missing_and_artefact(reference):
    with pytest.raises(fit.FitError, match="not in the reference: SBS99"):
        fit.subset_reference(reference, ["SBS1", "SBS99"], [])
    with pytest.raises(fit.FitError, match="artefacts: SBS3"):
        fit.subset_reference(reference, ["SBS1", "SBS3"], ["SBS3"])


def test_select_tumours_boundary():
    idx = pd.Index(SBS96_CHANNELS)
    m = pd.DataFrame(0, index=idx, columns=["t49", "t50", "t0"])
    m.iloc[0, 0] = 49
    m.iloc[0, 1] = 50
    kept, excluded = fit.select_tumours(m, 50)
    assert list(kept.columns) == ["t50"]
    assert excluded == ["t49", "t0"]


def test_restricted_subset_gets_no_exposure(reference):
    """A tumour that is pure SBS4 must not get SBS4 exposure if SBS4 is outside the subset."""
    matrix = pd.DataFrame({"T1": mixture(reference, {"SBS4": 400})})
    sub = fit.subset_reference(reference, ["SBS1", "SBS2"], [])
    expo, _, cos = fit.fit_exposures(matrix, sub)
    assert list(expo.columns) == ["SBS1", "SBS2"]
    assert "SBS4" not in expo.columns
    assert cos["T1"] < 0.5  # poor reconstruction is reported, not hidden


def test_zero_count_tumour(reference):
    matrix = pd.DataFrame({"T0": np.zeros(96), "T1": mixture(reference, {"SBS1": 100})})
    matrix.index = SBS96_CHANNELS
    expo, frac, cos = fit.fit_exposures(matrix, reference)
    assert (expo.loc["T0"] == 0).all()
    assert (frac.loc["T0"] == 0).all()
    assert np.isnan(cos["T0"])
    assert expo.loc["T1", "SBS1"] == pytest.approx(100, abs=1e-6)


def test_deterministic(reference):
    matrix = pd.DataFrame({"T1": mixture(reference, {"SBS1": 30, "SBS2": 70})})
    a = fit.fit_exposures(matrix, reference)[0]
    b = fit.fit_exposures(matrix, reference)[0]
    pd.testing.assert_frame_equal(a, b)


def _config():
    return {
        "fitting": {
            "min_snvs": 50,
            "min_snvs_sensitivity": [20, 100],
            "exclude_artefacts": ["SBS3"],
            "signature_subsets": {"P1": ["SBS1", "SBS2", "SBS5"], "BAD": ["SBS3"]},
        }
    }


def _cohort(reference):
    cols = {
        "n10": mixture(reference, {"SBS1": 10}),
        "n19": mixture(reference, {"SBS1": 19}),
        "n20": mixture(reference, {"SBS1": 20}),
        "n49": mixture(reference, {"SBS2": 49}),
        "n50": mixture(reference, {"SBS2": 50}),
        "n99": mixture(reference, {"SBS5": 99}),
        "n100": mixture(reference, {"SBS5": 100}),
        "n300": mixture(reference, {"SBS1": 150, "SBS5": 150}),
    }
    # tiny upward nudge so float rounding cannot put a total of 50 just below 50
    return pd.DataFrame(cols) * (1 + 1e-9)


def test_run_cohort_fit_and_sensitivity(reference):
    res = fit.run_cohort_fit("P1", _config(), _cohort(reference), reference)
    assert res.n_input == 8
    # totals < 50 are n10, n19, n20, n49
    assert sorted(res.excluded) == ["n10", "n19", "n20", "n49"]
    sens = dict(zip(res.sensitivity["min_snvs"], res.sensitivity["n_excluded"], strict=True))
    assert sens == {50: 4, 20: 2, 100: 6}  # <20: n10, n19; <100: all but n100, n300
    assert list(res.exposures.columns) == ["SBS1", "SBS2", "SBS5"]
    assert len(res.exposures) == 4
    assert res.exposures.loc["n300", "SBS1"] == pytest.approx(150, abs=1e-6)


def test_run_cohort_fit_min_snvs_override(reference):
    res = fit.run_cohort_fit("P1", _config(), _cohort(reference), reference, min_snvs=100)
    assert res.min_snvs == 100
    assert len(res.excluded) == 6


def test_run_cohort_fit_errors(reference):
    with pytest.raises(fit.FitError, match="no signature subset for project NOPE"):
        fit.run_cohort_fit("NOPE", _config(), _cohort(reference), reference)
    with pytest.raises(fit.FitError, match="artefacts"):
        fit.run_cohort_fit("BAD", _config(), _cohort(reference), reference)
    with pytest.raises(fit.FitError, match="no 'fitting' section"):
        fit.run_cohort_fit("P1", {}, _cohort(reference), reference)


def test_save_results(tmp_path, reference):
    res = fit.run_cohort_fit("P1", _config(), _cohort(reference), reference)
    paths = fit.save_results(res, tmp_path / "tables")
    names = [p.name for p in paths]
    assert names == [
        "exposures_P1.tsv",
        "exposures_P1_fractions.tsv",
        "fit_summary_P1.tsv",
        "fit_qc_P1.tsv",
        "excluded_P1.tsv",
    ]
    expo = pd.read_csv(paths[0], sep="\t", index_col=0)
    assert expo.shape == (4, 3)
    summary = pd.read_csv(paths[2], sep="\t").set_index("statistic")["value"]
    assert summary["n_tumours_input"] == "8"
    assert summary["n_excluded_main"] == "4"
    assert summary["n_excluded_min_snvs_20"] == "2"
    assert summary["n_excluded_min_snvs_100"] == "6"
    assert float(summary["mean_reconstruction_cosine"]) == pytest.approx(1.0)
    assert "NNLS" in summary["method"]


def test_qc_columns_and_values(reference):
    # pure SBS3 tumour fitted without SBS3: poor fit, exposures do not match observed total
    cohort = pd.DataFrame(
        {"good": mixture(reference, {"SBS1": 100}), "bad": mixture(reference, {"SBS3": 100})}
    )
    cfg = _config()
    cfg["fitting"]["exclude_artefacts"] = []
    res = fit.run_cohort_fit("P1", cfg, cohort, reference)
    qc = res.qc
    assert list(qc.columns) == ["observed_snvs", "sum_exposures", "residual_l1", "cosine"]
    assert qc.loc["good", "observed_snvs"] == pytest.approx(100)
    assert qc.loc["good", "sum_exposures"] == pytest.approx(100, abs=1e-6)
    assert qc.loc["good", "residual_l1"] == pytest.approx(0, abs=1e-6)
    assert qc.loc["good", "cosine"] == pytest.approx(1.0)
    assert qc.loc["bad", "residual_l1"] > 1.0
    assert qc.loc["bad", "cosine"] < 0.8
    flags = fit.low_cosine_flag(qc)
    assert flags.to_dict() == {"good": False, "bad": True}
    assert fit.low_cosine_flag(qc, threshold=0.0).sum() == 0


def test_low_cosine_flag_nan_is_false():
    qc = pd.DataFrame({"cosine": [np.nan, 0.5, 0.9]}, index=["a", "b", "c"])
    assert fit.low_cosine_flag(qc).tolist() == [False, True, False]


@pytest.mark.parametrize("main", [20, 100])
def test_threshold_dedupe(reference, main):
    res = fit.run_cohort_fit("P1", _config(), _cohort(reference), reference, min_snvs=main)
    assert list(res.sensitivity["min_snvs"]) == [20, 100]
    assert res.summary()["statistic"].is_unique


def test_threshold_dedupe_non_listed_main(reference):
    res = fit.run_cohort_fit("P1", _config(), _cohort(reference), reference, min_snvs=30)
    assert list(res.sensitivity["min_snvs"]) == [20, 30, 100]


def test_excluded_table_file(tmp_path, reference):
    res = fit.run_cohort_fit("P1", _config(), _cohort(reference), reference)
    paths = fit.save_results(res, tmp_path)
    excl = pd.read_csv(paths[4], sep="\t").set_index("tumour")
    # all tumours below the largest threshold (100): all but n100 and n300
    assert sorted(excl.index) == ["n10", "n19", "n20", "n49", "n50", "n99"]
    assert excl.loc["n19", "excluded_at_20"]
    assert not excl.loc["n20", "excluded_at_20"]
    assert excl.loc["n49", "excluded_main"]
    assert not excl.loc["n50", "excluded_main"]
    assert excl.loc["n99", "excluded_at_100"] and not excl.loc["n99", "excluded_at_50"]
    assert excl.loc["n49", "total_snvs"] == pytest.approx(49)


def test_duplicate_columns_raise(reference):
    m = pd.DataFrame({"T1": mixture(reference, {"SBS1": 60})})
    dup = pd.concat([m, m], axis=1)
    with pytest.raises(fit.FitError, match="duplicate tumour"):
        fit.fit_exposures(dup, reference)
    with pytest.raises(fit.FitError, match="duplicate tumour"):
        fit.select_tumours(dup, 50)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_bad_counts_raise(reference, bad):
    m = pd.DataFrame({"T1": mixture(reference, {"SBS1": 60})})
    m.iloc[3, 0] = bad
    with pytest.raises(fit.FitError, match="negative|non-finite"):
        fit.fit_exposures(m, reference)
    with pytest.raises(fit.FitError, match="negative|non-finite"):
        fit.select_tumours(m, 50)
