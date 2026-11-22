"""Tests for fingerprints.validate on tiny synthetic cohorts with known effects."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fingerprints import validate as v
from fingerprints.profile import SBS96_CHANNELS, SBS96_PLOT_ORDER


def lung_table(n_never=40, n_smoker=60, effect=0.3, seed=0, project="LUAD") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_never + n_smoker):
        smoker = i >= n_never
        sbs4 = np.clip(rng.normal(0.05 + (effect if smoker else 0), 0.05), 0, 1)
        rows.append(
            {
                "tumour": f"{project}-{i}",
                "SBS4": sbs4,
                "SBS1": 1 - sbs4,
                v.SNV_COL: 100 + i,
                v.COSINE_COL: 0.9,
                "smoking_code": "2" if smoker else "1",
                "smoking_label": "current" if smoker else "never",
                "smoking_primary_group": "smoker" if smoker else "never",
                "smoking_ever_group": "ever" if smoker else "never",
                "pack_years_smoked": float(10 + i) if smoker else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("tumour")


def test_signature_groups_are_exact():
    assert v.MMR_SIGS == ["SBS6", "SBS14", "SBS15", "SBS20", "SBS26", "SBS44"]
    assert v.UV_SIGS == ["SBS7a", "SBS7b", "SBS7c", "SBS7d"]
    t = pd.DataFrame(
        {s: [0.1] for s in [*v.MMR_SIGS, *v.UV_SIGS, "SBS38", "SBS1", "SBS21", "SBS10a"]}
    )
    assert v.sum_signatures(t, v.MMR_SIGS).iloc[0] == pytest.approx(0.6)
    assert v.sum_signatures(t, v.UV_SIGS).iloc[0] == pytest.approx(0.4)
    assert v.sum_signatures(t, v.UV_PLUS_38_SIGS).iloc[0] == pytest.approx(0.5)
    with pytest.raises(KeyError):
        v.sum_signatures(t.drop(columns="SBS44"), v.MMR_SIGS)


def test_mann_whitney_recovers_effect_and_sign():
    t = lung_table()
    a = t[t.smoking_code == "2"].SBS4
    b = t[t.smoking_code == "1"].SBS4
    res = v.mann_whitney(a, b)
    assert res["n1"] == 60 and res["n2"] == 40
    assert res["auc"] > 0.95 and res["p"] < 1e-6
    assert res["rank_biserial"] == pytest.approx(2 * res["auc"] - 1)
    flipped = v.mann_whitney(b, a)
    assert flipped["auc"] == pytest.approx(1 - res["auc"])
    assert flipped["rank_biserial"] == pytest.approx(-res["rank_biserial"])


def test_auc_exact_with_ties():
    # x=[1,2], y=[2,3]: pairs (1,2)<,(1,3)<,(2,2)=,(2,3)< -> AUC = 0.5/4 = 0.125
    assert v.auc_from_ranks(np.array([1.0, 2.0]), np.array([2.0, 3.0])) == pytest.approx(0.125)


def test_no_effect_gives_large_p_and_auc_near_half():
    t = lung_table(effect=0.0, seed=3)
    res = v.mann_whitney(t[t.smoking_code == "2"].SBS4, t[t.smoking_code == "1"].SBS4)
    assert res["p"] > 0.05
    assert abs(res["auc"] - 0.5) < 0.12


def test_bootstrap_ci_contains_estimate_and_is_deterministic():
    t = lung_table(effect=0.05, seed=4)
    a, b = t[t.smoking_code == "2"].SBS4, t[t.smoking_code == "1"].SBS4
    r1, r2 = v.mann_whitney(a, b), v.mann_whitney(a, b)
    assert (r1["ci_low"], r1["ci_high"]) == (r2["ci_low"], r2["ci_high"])
    assert r1["ci_low"] <= r1["auc"] <= r1["ci_high"]
    other = v.bootstrap_auc_ci(a, b, seed=1)
    assert other != (r1["ci_low"], r1["ci_high"])


def test_mann_whitney_empty_group_is_nan():
    res = v.mann_whitney([0.1, 0.2], [])
    assert np.isnan(res["auc"]) and np.isnan(res["p"]) and res["n2"] == 0


def test_spearman_known_rho():
    x = np.arange(30.0)
    res = v.spearman(x, x**2)
    assert res["rho"] == pytest.approx(1.0) and res["n"] == 30 and res["p"] < 1e-6
    assert res["ci_low"] <= res["rho"] <= res["ci_high"] + 1e-9
    neg = v.spearman(x, -x)
    assert neg["rho"] == pytest.approx(-1.0)
    nan = v.spearman([1, 2, np.nan, 4], [1, 2, 3, np.nan])
    assert nan["n"] == 2 and np.isnan(nan["rho"])


def test_van_elteren_single_stratum_matches_auc_and_detects_effect():
    t = lung_table()
    a = t[t.smoking_code == "2"].SBS4.to_numpy()
    b = t[t.smoking_code == "1"].SBS4.to_numpy()
    ve = v.van_elteren([(a, b)])
    assert ve["auc"] == pytest.approx(v.auc_from_ranks(a, b))
    assert ve["p"] < 1e-6
    assert ve["ci_low"] <= ve["auc"] <= ve["ci_high"]


def test_van_elteren_adjusts_for_stratum_baseline():
    """Strata differ in baseline; pooling naively gives a spurious effect, stratifying does not."""
    rng = np.random.default_rng(1)
    # stratum A: low baseline, mostly group 1; stratum B: high baseline, mostly group 2; no effect
    a1, a2 = rng.normal(0, 1, 90), rng.normal(0, 1, 10)
    b1, b2 = rng.normal(5, 1, 10), rng.normal(5, 1, 90)
    naive = v.mann_whitney(np.concatenate([a1, b1]), np.concatenate([a2, b2]))
    ve = v.van_elteren([(a1, a2), (b1, b2)])
    assert naive["auc"] < 0.3  # confounded
    assert abs(ve["auc"] - 0.5) < 0.1 and ve["p"] > 0.01
    assert ve["n_strata"] == 2


def test_van_elteren_skips_empty_strata():
    ve = v.van_elteren([(np.array([1.0, 2.0]), np.array([]))])
    assert ve["n_strata"] == 0 and np.isnan(ve["auc"])


def test_filters_by_snvs_and_cosine():
    t = pd.DataFrame(
        {v.SNV_COL: [19, 20, 49, 50, 100], v.COSINE_COL: [0.9, 0.79, 0.8, np.nan, 0.95]},
        index=list("abcde"),
    )
    assert list(v.apply_filters(t, 50).index) == ["d", "e"]
    assert list(v.apply_filters(t, 20).index) == ["b", "c", "d", "e"]
    assert list(v.apply_filters(t, 20, 0.8).index) == ["c", "e"]  # >= 0.8; NaN fails


def test_filters_never_use_group_labels():
    t = lung_table()
    shuffled = t.copy()
    shuffled["smoking_primary_group"] = (
        t["smoking_primary_group"].sample(frac=1, random_state=1).values
    )
    shuffled["msi_status"] = "MSI"
    kept1 = v.apply_filters(t, 100, 0.8).index
    kept2 = v.apply_filters(shuffled, 100, 0.8).index
    assert list(kept1) == list(kept2)
    # works on a table that has no group columns at all
    assert len(v.apply_filters(t[[v.SNV_COL, v.COSINE_COL]], 100, 0.8)) == len(kept1)


def test_borderline_boundaries_inclusive():
    s = pd.Series([0.1499, 0.15, 0.2, 0.25, 0.2501, np.nan])
    assert list(v.is_borderline(s)) == [False, True, True, True, False, False]


def test_assemble_prefers_total_snvs_and_joins_clinical():
    frac = pd.DataFrame({"SBS4": [0.5, 0.2]}, index=["t1", "t2"])
    qc = pd.DataFrame(
        {"observed_snvs": [99, 99], "total_snvs": [60, 70], "cosine": [0.9, 0.8]},
        index=["t1", "t2"],
    )
    clin = pd.DataFrame({"barcode": ["t1", "t2"], "smoking_code": ["1", "2"]})
    out = v.assemble_cohort(frac, qc, clin)
    assert list(out.index) == ["t1", "t2"] and out.loc["t1", v.SNV_COL] == 60


def _guard_inputs():
    frac = pd.DataFrame({"SBS4": [0.5, 0.2]}, index=["t1", "t2"])
    qc = pd.DataFrame({"total_snvs": [60, 70], "cosine": [0.9, 0.8]}, index=["t1", "t2"])
    clin = pd.DataFrame({"barcode": ["t1", "t2", "t3"], "smoking_code": ["1", "2", "1"]})
    return frac, qc, clin


def test_silent_drop_guard_allows_only_cases_below_fit_threshold():
    frac, qc, clin = _guard_inputs()
    totals = pd.Series({"t1": 60, "t2": 70, "t3": 19})
    out = v.assemble_cohort(frac, qc, clin, totals)
    assert list(out.index) == ["t1", "t2"]  # t3 (19 SNVs) is the known, explained drop
    un = v.unmatched_chosen_cases(pd.Index(clin["barcode"]), frac.index, totals)
    assert un["barcode"].tolist() == ["t3"] and un[v.SNV_COL].tolist() == [19]


def test_silent_drop_guard_fails_loudly_on_unexplained_cases():
    frac, qc, clin = _guard_inputs()
    with pytest.raises(v.CohortMismatchError, match="not explained"):
        v.assemble_cohort(frac, qc, clin, pd.Series({"t1": 60, "t2": 70, "t3": 20}))  # at 20
    with pytest.raises(v.CohortMismatchError, match="not explained"):
        v.assemble_cohort(frac, qc, clin, pd.Series({"t1": 60, "t2": 70}))  # absent from matrix
    with pytest.raises(v.CohortMismatchError, match="no matrix totals"):
        v.assemble_cohort(frac, qc, clin)  # cannot verify
    with pytest.raises(v.CohortMismatchError, match="no clinical row"):
        v.assemble_cohort(frac, qc, clin[clin.barcode != "t2"], pd.Series({"t3": 5}))


def test_cohort_flow_exact_counts():
    groups = pd.Series(["a", "a", "a", "b", "b"], index=["t1", "t2", "t3", "t4", "t5"])
    totals = pd.Series({"t1": 10, "t2": 25, "t3": 120, "t4": 55, "t5": 100})  # t1<20
    cos = pd.Series({"t2": 0.9, "t3": 0.7, "t4": 0.85, "t5": 0.8})
    fl = v.cohort_flow("X", groups, totals, pd.Index(["t2", "t3", "t4", "t5"]), cos)
    a = fl[fl.group == "a"].iloc[0]
    b = fl[fl.group == "b"].iloc[0]
    assert (a.chosen_cases, a.in_fit_table, a.snvs_ge_20, a.snvs_ge_50, a.snvs_ge_100) == (
        3, 2, 2, 1, 1,
    )  # fmt: skip
    assert a["snvs_ge_50_and_cosine_ge_0.8"] == 0  # t3 has cosine 0.7
    assert (b.chosen_cases, b.snvs_ge_50, b.snvs_ge_100, b["snvs_ge_50_and_cosine_ge_0.8"]) == (
        2, 2, 1, 2,
    )  # fmt: skip


def test_van_elteren_tied_two_strata_hand_computed():
    # stratum 1: x=[1], y=[1, 2]  -> mid-ranks 1.5,1.5,3; W=1.5; N=3; w=1/4
    #   E(W)=2; Var(W)=n1*n2/12*((N+1) - (2^3-2)/(N(N-1))) = 2/12*(4-1) = 0.5; AUC=0.25
    # stratum 2: x=[1], y=[2, 3]  -> ranks 1,2,3; W=1; E=2; Var=2/12*4=2/3; AUC=0
    # T-E = 0.25*(1.5-2) + 0.25*(1-2) = -0.375; Var(T) = (0.5 + 2/3)/16 = 7/96
    ve = v.van_elteren(
        [(np.array([1.0]), np.array([1.0, 2.0])), (np.array([1.0]), np.array([2.0, 3.0]))]
    )
    z = -0.375 / np.sqrt(7 / 96)
    assert ve["z"] == pytest.approx(z)
    assert ve["p"] == pytest.approx(2 * stats.norm.sf(abs(z)))
    # weights w*n1*n2 = 0.5 in both strata -> weighted AUC = (0.5*0.25 + 0.5*0)/1 = 0.125
    assert ve["auc"] == pytest.approx(0.125)
    assert ve["n1"] == 2 and ve["n2"] == 4 and ve["n_strata"] == 2


def test_pooled_msi_matches_per_project_split():
    t = crc_table()
    rows = v.to_table(v.colorectal_msi_tests(t))
    pick = lambda c: rows[(rows.cohort == c) & (rows.setting == "primary")].iloc[0]  # noqa: E731
    pooled, coad, read = pick("COAD+READ"), pick("COAD"), pick("READ")
    assert coad.n1 + read.n1 == pooled.n1 == 30
    assert coad.n2 + read.n2 == pooled.n2 == 60
    assert coad.n1 == ((t.project == "TCGA-COAD") & (t.msi_group == "MSI")).sum()
    assert read.n2 == ((t.project == "TCGA-READ") & (t.msi_group == "MSS")).sum()


def test_describe_row_has_min_max_columns_and_no_ci():
    t = lung_table()
    row = v.describe_row("x", "LUAD", "c", "descriptive", v.PRIMARY, t, "SBS4")
    assert row["min"] == pytest.approx(t.SBS4.min()) and row["max"] == pytest.approx(t.SBS4.max())
    assert np.isnan(row["ci_low"]) and np.isnan(row["ci_high"])


def test_pooled_median_cosine_uses_analysed_groups_only():
    a, b = lung_table(seed=1), lung_table(30, 50, 0.2, seed=2, project="LUSC")
    a.loc[a.smoking_code == "2", v.COSINE_COL] = 0.99
    a.loc[a.smoking_code == "1", v.COSINE_COL] = 0.81
    b.loc[:, v.COSINE_COL] = 0.9
    # an unlabelled tumour (neither group) with an extreme cosine must not enter the median
    extra = a.iloc[[0]].copy()
    extra.index = ["odd"]
    extra["smoking_primary_group"] = np.nan
    extra[v.COSINE_COL] = 0.0
    a = pd.concat([a, extra])
    row = v.lung_pooled({"LUAD": a, "LUSC": b}, settings=[v.PRIMARY])[0]
    used = pd.concat(
        [x[x.smoking_primary_group.isin(["smoker", "never"])][v.COSINE_COL] for x in (a, b)]
    )
    assert row["median_cosine1"] == pytest.approx(used.median())


def test_msi_fraction_read_row_has_no_coad_expectation():
    t = crc_table()
    rows = v.msi_fraction_by_case({"TCGA-COAD": t[t.project == "TCGA-COAD"],
                                   "TCGA-READ": t[t.project == "TCGA-READ"]})  # fmt: skip
    coad, read = rows
    assert "10-20%" in coad["notes"] and "does not cover" in read["notes"]
    assert "Nature 487" not in read["notes"]


def test_skcm_only_summed_fraction_is_primary():
    rows = v.to_table(v.melanoma_summary(mel_table()))
    assert list(rows[rows.label == "primary"].analysis) == ["melanoma_uv_fraction"]
    assert (
        rows[rows.analysis == "melanoma_uv_proportion"]
        .query("setting == 'primary'")
        .label.eq("secondary")
        .all()
    )
    cnt = v.to_table(v.melanoma_summary(mel_table(), "count_as_sbs", [v.PRIMARY], False))
    assert (cnt.label == "sensitivity").all()


def test_ct_control_runs_at_each_setting():
    m = pd.DataFrame(0, index=SBS96_CHANNELS, columns=["s1", "s2", "s3"])
    m.loc["A[C>T]A", ["s1", "s2", "s3"]] = [30, 60, 120]
    m.loc["A[C>A]A", ["s1", "s2", "s3"]] = [10, 10, 10]
    ct = v.ct_fractions(m)
    rows = v.melanoma_ct_control({"SKCM": ct, "LUAD": ct}, v.SETTINGS[:3])
    assert [r["setting"] for r in rows[::3]] == ["primary", "min_snvs_20", "min_snvs_100"]
    assert [r["n1"] for r in rows[::3]] == [2, 3, 1]
    assert [r["label"] for r in rows[::3]] == ["secondary", "sensitivity", "sensitivity"]


def test_borderline_counts_three_denominators():
    t = crc_table()
    t.loc[["t0", "t1", "t40"], "msi_score"] = [0.15, 0.25, 0.2]  # two MSI, one MSS
    t.loc["t40", v.SNV_COL] = 30  # in fit table (>=20) but not analysed (>=50)
    t = v.add_mmr_columns(t.drop(columns=["mmr_sum", "msi_group", "borderline", "total_snvs"]))
    clin = t.assign(barcode=t.index)
    rows = {r["group"]: r for r in v.borderline_counts(clin, t, "ALL")}
    assert rows["MSI"]["clinical_borderline"] == 2 and rows["MSI"]["analysed_ge50_borderline"] == 2
    assert rows["MSS"]["clinical_borderline"] == 1 and rows["MSS"]["in_fit_table_borderline"] == 1
    assert rows["MSS"]["analysed_ge50_borderline"] == 0


def _synthetic_tables():
    luad = lung_table(seed=1)
    lusc = lung_table(30, 50, 0.2, seed=2, project="LUSC")
    cohorts = {"LUAD": luad, "LUSC": lusc}
    lung = v.to_table(v.lung_two_group(cohorts, "with SBS18"))
    dose = v.lung_dose_response(cohorts, v.SETTINGS)
    for r in dose:
        r["comparison"] += "; fit with SBS18"
    mel = mel_table()
    mel2 = mel.copy()
    mel2["uv_sbs7a_d"] = mel["uv_sbs7a_d"] + 0.1
    crc = crc_table()
    crc["SBS6"] = np.where(crc.msi_group == "MSS", 0.02, 0.1)
    tables = {
        "lung_sbs4": lung,
        "lung_dose_response": v.to_table(dose),
        "melanoma_uv": v.to_table(v.melanoma_summary(mel)),
        "melanoma_adjacent_check": v.to_table([v.melanoma_adjacent_check(mel, mel2)]),
        "colorectal_mmr": v.to_table(v.colorectal_msi_tests(crc)),
        "colorectal_sbs6_leakage": v.to_table(v.sbs1_sbs6_overlap(crc)),
    }
    totals = luad[v.SNV_COL]
    flow = v.cohort_flow(
        "LUAD", v.lung_flow_groups(luad.assign(barcode=luad.index)), totals, luad.index,
        luad[v.COSINE_COL],
    )  # fmt: skip
    return tables, flow


def test_summary_contains_required_sections_and_caveats():
    tables, flow = _synthetic_tables()
    drops = pd.DataFrame({"tri_allelic": [0, 0], "multiallelic_position": [0, 0]})
    text = v.build_summary(tables, flow, drops)
    for needle in (
        "## Headline",
        "## Cohort flow and the SNV filter",
        "label-blind",
        "stays at >= 50 SNVs as pre-specified",
        "side by side",
        "consistent with",
        "three pre-specified primary analyses",
        "not for diagnosis",
        "Refitting, not machine learning",
        "Lower fit quality in never-smokers",
        "D6/D6b filters did nothing",
        "SBS1/SBS6 overlap",
        "Pack-years vs SBS4",
        "READ has only",
        "D5 check",
    ):
        assert needle in text, needle
    assert text.index("## Headline") < text.index("## Primary results")
    assert text.index("## Primary results") < text.index("## Limitations")
    lower = text.lower()
    assert "causes" not in lower and "caused" not in lower  # no causal wording


def test_two_group_row_applies_setting_before_groups():
    t = lung_table()
    t.loc[t.index[:5], v.SNV_COL] = 30  # five never-smokers below 50
    row = v.two_group_row(
        "x",
        "LUAD",
        "c",
        "primary",
        v.PRIMARY,
        "SBS4",
        t,
        "smoking_primary_group",
        "smoker",
        "never",
    )
    assert row["n2"] == 35 and row["n1"] == 60
    assert row["median_cosine1"] == pytest.approx(0.9)
    assert set(row) == set(v.TABLE_COLUMNS)
    row20 = v.two_group_row(
        "x", "LUAD", "c", "sensitivity", v.SETTINGS[1], "SBS4", t, "smoking_primary_group",
        "smoker", "never",
    )  # fmt: skip
    assert row20["n2"] == 40


def test_lung_two_group_labels_and_pooled():
    cohorts = {"LUAD": lung_table(seed=1), "LUSC": lung_table(30, 50, 0.2, seed=2, project="LUSC")}
    rows = v.to_table(v.lung_two_group(cohorts))
    prim = rows[rows.label == "primary"]
    assert len(prim) == 1 and prim.iloc[0].cohort == "LUAD" and prim.iloc[0].setting == "primary"
    assert (rows[rows.cohort == "LUSC"].label == "descriptive").all()
    pooled = rows[rows.cohort == "LUAD+LUSC"]
    assert len(pooled) == 8 and pooled.iloc[0].effect > 0.9
    assert set(pooled[pooled.setting == "primary"].label) == {"secondary"}


def test_dose_response_uses_ever_smokers_only():
    t = lung_table()
    t.loc[t.smoking_code == "2", "SBS4"] = np.linspace(0, 1, 60)
    rows = v.lung_dose_response({"LUAD": t})
    assert rows[0]["n1"] == 60 and rows[0]["effect"] == pytest.approx(1.0)


def mel_table() -> pd.DataFrame:
    t = pd.DataFrame(
        {
            "SBS1": [0.1, 0.5, 0.0],
            "SBS5": [0.1, 0.2, 0.9],
            "SBS7a": [0.4, 0.1, 0.0],
            "SBS7b": [0.3, 0.1, 0.0],
            "SBS7c": [0.05, 0.0, 0.0],
            "SBS7d": [0.05, 0.0, 0.0],
            "SBS38": [0.0, 0.1, 0.1],
            v.SNV_COL: [100, 100, 100],
            v.COSINE_COL: [0.95, 0.9, 0.9],
            "sample_type_code": ["06", "06", "01"],
        },
        index=["a", "b", "c"],
    )
    return v.add_uv_columns(t, ["SBS1", "SBS5", "SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS38"])


def test_uv_columns_and_summary():
    t = mel_table()
    assert t.uv_sbs7a_d.tolist() == pytest.approx([0.8, 0.2, 0.0])
    assert t.uv_sbs7a_b.tolist() == pytest.approx([0.7, 0.2, 0.0])
    assert t.uv_sbs7a_d_plus_38.tolist() == pytest.approx([0.8, 0.3, 0.1])
    assert t.uv_is_largest.tolist() == [1.0, 0.0, 0.0]  # b: SBS1 0.5 > 0.2
    assert t.uv_gt_50.tolist() == [1.0, 0.0, 0.0]
    rows = v.to_table(v.melanoma_summary(t, settings=[v.PRIMARY]))
    prim = rows[rows.label == "primary"]
    assert set(prim.analysis) == {"melanoma_uv_fraction"}
    prop = rows[rows.comparison.str.contains("single largest")].iloc[0]
    assert prop.effect == pytest.approx(1 / 3) and prop.n1 == 3


def test_adjacent_check_and_sample_type():
    a = mel_table()
    b = a.copy()
    b["uv_sbs7a_d"] = a["uv_sbs7a_d"] + 0.1
    row = v.melanoma_adjacent_check(a, b)
    assert row["effect"] == pytest.approx(0.1) and row["n1"] == 3
    assert "material change (|median diff| > 0.05): yes" in row["notes"]
    types = v.melanoma_sample_type(a)
    assert sorted(r["group1"] for r in types) == ["sample type 01", "sample type 06"]


def test_ct_fractions_exact():
    m = pd.DataFrame(0, index=SBS96_CHANNELS, columns=["s1", "s2"])
    m.loc["A[C>T]A", "s1"] = 1  # C>T, no pyrimidine neighbour
    m.loc["T[C>T]A", "s1"] = 1  # dipyrimidine (5' T), not CC
    m.loc["C[C>T]A", "s1"] = 1  # dipyrimidine and CC
    m.loc["A[C>T]C", "s1"] = 1  # CC (3' C)
    m.loc["A[C>A]A", "s1"] = 4
    m.loc["A[T>C]A", "s2"] = 5
    out = v.ct_fractions(m)
    assert out.loc["s1", "ct_all"] == pytest.approx(0.5)
    assert out.loc["s1", "ct_dipyrimidine"] == pytest.approx(0.375)
    assert out.loc["s1", "ct_cc"] == pytest.approx(0.25)
    assert out.loc["s2", "ct_all"] == 0 and out.loc["s1", v.SNV_COL] == 8
    ctl = v.melanoma_ct_control({"SKCM": out, "LUAD": out}, [v.Setting("p", 5)])
    assert len(ctl) == 3 and ctl[0]["n1"] == 2


def crc_table(n_msi=30, n_mss=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_msi + n_mss):
        msi = i < n_msi
        r = {s: 0.0 for s in [*v.MMR_SIGS, "SBS1", "SBS5", "SBS10a", "SBS10b", "SBS28"]}
        r["SBS44"] = rng.uniform(0.3, 0.6) if msi else rng.uniform(0, 0.1)
        r["SBS1"] = 1 - r["SBS44"]
        r.update(
            {
                v.SNV_COL: 2000 if msi else 100,
                v.COSINE_COL: 0.9,
                "msi_status": "MSI" if msi else "MSS",
                "msi_score": 0.3 if msi else 0.1,
                "project": "TCGA-COAD" if i % 2 == 0 else "TCGA-READ",
                "case_id": f"c{i}",
            }
        )
        rows.append(r)
    t = pd.DataFrame(rows, index=[f"t{i}" for i in range(n_msi + n_mss)])
    t.loc["t0", ["SBS10a", "SBS1", "SBS44"]] = [0.8, 0.2, 0.0]  # POLE-like MSI
    return v.add_mmr_columns(t)


def test_colorectal_msi_recovered_and_borderline_excluded():
    t = crc_table()
    rows = v.to_table(v.colorectal_msi_tests(t))
    prim = rows[rows.label == "primary"]
    assert len(prim) == 1 and prim.iloc[0].n1 == 30 and prim.iloc[0].n2 == 60
    assert prim.iloc[0].effect > 0.95
    t.loc[t.index[:10], "msi_score"] = 0.25  # borderline MSI, inclusive upper bound
    t = v.add_mmr_columns(t.drop(columns=["mmr_sum", "msi_group", "borderline", "total_snvs"]))
    bl = v.to_table(v.colorectal_msi_tests(t))
    ex = bl[(bl.cohort == "COAD+READ") & bl.setting.str.startswith("exclude_borderline")]
    assert ex.iloc[0].n1 == 20 and ex.iloc[0].label == "sensitivity"


def test_colorectal_other_tables():
    t = crc_table()
    hyper = v.colorectal_hypermutation(t)[0]
    assert hyper["median1"] == 2000 and hyper["median2"] == 100
    pole = v.to_table(v.pole_like_counts(t))
    r = pole[(pole.cohort == "COAD+READ") & (pole.group1 == "MSI")].iloc[0]
    assert r.n1 == 30 and r.effect == pytest.approx(1 / 30)
    per = v.colorectal_per_signature(t)
    assert [r["comparison"].split(": ")[1].split()[0] for r in per] == v.MMR_SIGS
    leak = v.sbs1_sbs6_overlap(t.assign(SBS6=0.0))
    assert len(leak) == 8  # per group: describe, SBS6>0, SBS6>5%, Spearman
    clin = {"COAD": t[t.project == "TCGA-COAD"]}
    frac = v.msi_fraction_by_case(clin)[0]
    assert frac["n1"] + frac["n2"] == 45


def test_plots_and_markdown_smoke(tmp_path):
    t = crc_table()
    assert v.plot_colorectal(t, "x", tmp_path / "c.png").stat().st_size > 1000
    luad = lung_table()
    luad["smoking_code"] = luad["smoking_code"].astype(str)
    p = v.plot_lung_groups({"LUAD": luad}, {"LUAD": "s"}, tmp_path / "l.png")
    assert p.exists()
    assert v.plot_pack_years({"LUAD": luad}, {}, tmp_path / "p.png").exists()
    m = pd.DataFrame(1.0, index=SBS96_CHANNELS, columns=["a", "b"])
    assert v.plot_mean_spectra({"X": m}, tmp_path / "s.png", SBS96_PLOT_ORDER).exists()
    ct = v.ct_fractions(m * 60)
    sigs = ["SBS1", "SBS5", "SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS38"]
    mel = mel_table().assign(**{v.SNV_COL: 100})
    assert v.plot_melanoma(mel, {"SKCM": ct}, sigs, tmp_path / "m.png").exists()
    md = v.table_to_markdown(v.to_table(v.lung_dose_response({"LUAD": luad})), ["cohort", "effect"])
    assert md.startswith("| cohort | effect |")


def test_run_validation_script_is_importable():
    spec = importlib.util.spec_from_file_location(
        "run_validation", Path(__file__).parents[1] / "scripts" / "run_validation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
