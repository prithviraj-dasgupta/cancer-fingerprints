"""Run the Milestone 4 validation analyses and write tables, figures and a summary.

Reads (all produced by earlier steps; never downloads anything):
  results/tables/exposures_<P><sfx>_all20_fractions.tsv   NNLS fractions, tumours with >=20 SNVs
  results/tables/fit_qc_<P><sfx>_all20.tsv                QC: total_snvs, cosine, ...
  results/tables/clinical_<P>.tsv                         chosen case per patient + groups
  data/processed/sbs96_<P>[_count_as_sbs].tsv             raw SBS96 matrices (C>T control)
with <sfx> '' (default fit), '_noSBS18' (lung, colorectal) or '_count_as_sbs' (SKCM).
Thresholds 20/50/100 are applied by filtering on total_snvs.

Writes results/tables/validation_<analysis>.tsv, results/figures/*.png and
results/validation_summary.md. The analysis plan is fixed in docs/decisions.md (D7).

Usage: uv run python scripts/run_validation.py [--results results] [--processed data/processed]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from fingerprints import validate as v
from fingerprints.profile import SBS96_PLOT_ORDER

LUNG = {"LUAD": "TCGA-LUAD", "LUSC": "TCGA-LUSC"}
CRC = {"COAD": "TCGA-COAD", "READ": "TCGA-READ"}
SKCM = "TCGA-SKCM"
SKCM_SIGS = ["SBS1", "SBS5", "SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS38"]
NUMERIC = ["pack_years_smoked", "msi_score"]
BOOLS = ["msi_borderline", "msi_wxs_wgs_disagree", "tobacco_status_conflict"]


def read_tsv(path: Path, **kw) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"error: expected input file is missing: {path}")
    return pd.read_csv(path, sep="\t", **kw)


def load_clinical(results: Path, project: str) -> pd.DataFrame:
    clin = read_tsv(results / "tables" / f"clinical_{project}.tsv", dtype=str)
    for col in NUMERIC:
        if col in clin:
            clin[col] = pd.to_numeric(clin[col], errors="coerce")
    return clin


def matrix_totals(processed: Path, project: str, sfx: str) -> pd.Series:
    """SBS96 matrix totals per tumour (the _noSBS18 fits use the default profile)."""
    msfx = "_count_as_sbs" if sfx == "_count_as_sbs" else ""
    return read_tsv(processed / f"sbs96_{project}{msfx}.tsv", index_col=0).sum(axis=0)


def load_cohort(
    results: Path,
    processed: Path,
    project: str,
    sfx: str,
    clin: pd.DataFrame,
    counts: bool = False,
) -> pd.DataFrame:
    """Fit fractions + QC + clinical; chosen cases missing from the fit table must have <20 SNVs."""
    tab = results / "tables"
    # counts=True reads the NNLS-attributed exposure counts instead of the fractions
    kind = "" if counts else "_fractions"
    frac = read_tsv(tab / f"exposures_{project}{sfx}_all20{kind}.tsv", index_col=0)
    qc = read_tsv(tab / f"fit_qc_{project}{sfx}_all20.tsv", index_col=0)
    totals = matrix_totals(processed, project, sfx)
    out = v.assemble_cohort(frac, qc, clin, totals)
    n = len(v.unmatched_chosen_cases(pd.Index(clin["barcode"]), frac.index, totals))
    print(
        f"{project}{sfx}: {len(out)} tumours joined; {n} chosen cases below 20 SNVs",
        file=sys.stderr,
    )
    return out


def load_matrix(processed: Path, project: str, sfx: str, barcodes: pd.Index) -> pd.DataFrame:
    m = read_tsv(processed / f"sbs96_{project}{sfx}.tsv", index_col=0)
    return m.loc[:, m.columns.isin(barcodes)]


def fmt_stats(row: pd.Series, what: str = "AUC") -> str:
    return (
        f"n={int(row['n1'])} vs {int(row['n2'])}; {what}={row['effect']:.2f} "
        f"({row['ci_low']:.2f}-{row['ci_high']:.2f}); p={row['p']:.2g}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument(
        "--raw-clinical",
        type=Path,
        default=Path("data/raw/clinical"),
        help="downloaded GDC clinical tables (<project>/clinical.tsv), for melanoma histology/site",
    )
    args = ap.parse_args(argv)
    res, proc = args.results, args.processed
    tabdir, figdir = res / "tables", res / "figures"
    tables: dict[str, pd.DataFrame] = {}

    clin = {p: load_clinical(res, p) for p in [*LUNG.values(), SKCM, *CRC.values()]}

    # ---------------- A. lung ----------------
    lung = {k: load_cohort(res, proc, p, "", clin[p]) for k, p in LUNG.items()}
    lung_no18 = {k: load_cohort(res, proc, p, "_noSBS18", clin[p]) for k, p in LUNG.items()}
    rows = v.lung_two_group(lung, "with SBS18") + v.lung_two_group(lung_no18, "without SBS18")
    tables["lung_sbs4"] = v.to_table(rows)
    dose_with = v.lung_dose_response(lung, v.SETTINGS)
    dose_without = v.lung_dose_response(lung_no18, [v.PRIMARY])
    for r in dose_with:
        r["comparison"] += "; fit with SBS18"
    for r in dose_without:
        r["comparison"] += "; fit without SBS18"
        r["label"] = "sensitivity"
    lung_counts = {k: load_cohort(res, proc, p, "", clin[p], counts=True) for k, p in LUNG.items()}
    dose_count = v.lung_dose_response(lung_counts, v.SETTINGS, measure="mutation count")
    for r in dose_count:
        r["comparison"] += "; fit with SBS18"
    tables["lung_dose_response"] = v.to_table(dose_with + dose_without + dose_count)
    tables["lung_never_smoker_sbs4"] = v.to_table(v.lung_never_smoker_sbs4(lung))
    tables["lung_by_smoking_group"] = v.to_table(v.lung_group_descriptives(lung))

    # ---------------- B. melanoma ----------------
    skcm = v.add_uv_columns(load_cohort(res, proc, SKCM, "", clin[SKCM]), SKCM_SIGS)
    skcm_c = v.add_uv_columns(load_cohort(res, proc, SKCM, "_count_as_sbs", clin[SKCM]), SKCM_SIGS)
    mel = v.melanoma_summary(skcm, "exclude (default)")
    mel += v.melanoma_summary(skcm_c, "count_as_sbs", [v.PRIMARY], primary_variant=False)
    mel += v.melanoma_sample_type(skcm)
    tables["melanoma_uv"] = v.to_table(mel)
    tables["melanoma_adjacent_check"] = v.to_table([v.melanoma_adjacent_check(skcm, skcm_c)])
    ct = {}
    all_projects = {"SKCM": SKCM, **LUNG, **CRC}
    for name, p in all_projects.items():
        m = load_matrix(proc, p, "", clin[p]["barcode"])
        ct[name] = v.ct_fractions(m)
    tables["melanoma_ct_control"] = v.to_table(
        v.melanoma_ct_control(ct, [v.PRIMARY, v.SETTINGS[1], v.SETTINGS[2]])
    )

    # ---------------- C. colorectal ----------------
    crc_parts = []
    for p in CRC.values():
        c = v.add_mmr_columns(load_cohort(res, proc, p, "", clin[p]))
        crc_parts.append(c)
    crc = pd.concat(crc_parts)
    tables["colorectal_mmr"] = v.to_table(v.colorectal_msi_tests(crc))
    tables["colorectal_mmr_per_signature"] = v.to_table(v.colorectal_per_signature(crc))
    tables["colorectal_hypermutation"] = v.to_table(
        v.colorectal_hypermutation(crc)
        + v.msi_fraction_by_case({k: clin[p] for k, p in CRC.items()})
    )
    tables["colorectal_sbs6_leakage"] = v.to_table(v.sbs1_sbs6_overlap(crc))
    tables["colorectal_pole_like"] = v.to_table(v.pole_like_counts(crc))
    crc_no18 = pd.concat(
        [v.add_mmr_columns(load_cohort(res, proc, p, "_noSBS18", clin[p])) for p in CRC.values()]
    )
    nb = v.colorectal_msi_tests(crc_no18, [v.PRIMARY], include_borderline_exclusion=False)
    for r in nb:
        r["comparison"] += "; fit without SBS18"
        r["label"] = "sensitivity"
    tables["colorectal_mmr_noSBS18"] = v.to_table(nb)

    raw_skcm = args.raw_clinical / SKCM / "clinical.tsv"
    extra_clin = None
    if raw_skcm.is_file():
        extra_clin = pd.read_csv(raw_skcm, sep="\t", dtype=str).drop_duplicates("case_id")
        extra_clin = extra_clin.set_index("case_id")
    non_uv = v.melanoma_non_uv_listing(skcm, SKCM_SIGS, extra_clin)
    analysed = v.apply_filters(skcm, v.PRIMARY.min_snvs)
    analysed_extra = pd.DataFrame(index=analysed.index)
    for col in v.SKCM_CLINICAL_EXTRA:
        if extra_clin is not None and col in extra_clin.columns:
            analysed_extra[col] = extra_clin[col].reindex(analysed["case_id"]).to_numpy()
    fields = [c for c in ("primary_diagnosis", "primary_site") if c in analysed_extra.columns]
    non_uv_fields = v.non_uv_field_counts(non_uv, analysed_extra, fields)
    n_origin = int(
        analysed["tumor_of_origin"].replace("", pd.NA).notna().sum()
        if "tumor_of_origin" in analysed.columns
        else 0
    )
    biopsy = analysed[["sample_type_code", "site_of_resection_or_biopsy"]].copy()
    biopsy["sample_type_code"] = biopsy["sample_type_code"].astype(str)
    biopsy_counts = v.non_uv_field_counts(
        non_uv, biopsy, ["site_of_resection_or_biopsy"], by="sample_type_code"
    )
    skcm_note = (
        "Subtype fields: biopsy site (`site_of_resection_or_biopsy`) is available. "
        f"`tumor_of_origin` is empty in the downloaded data ({n_origin} of {len(analysed)} "
        "analysed tumours have a value). Acral/mucosal/uveal subtype and genomic subtype "
        "(BRAF/NRAS/NF1) are not recorded in the downloaded clinical fields; histology "
        "(`primary_diagnosis`, `morphology`) is available. `primary_site` for SKCM largely "
        "reflects the sampled (often metastatic) site and should not be read as the site of the "
        "primary tumour."
        if extra_clin is not None
        else "Subtype not available in downloaded clinical fields (raw clinical table not found)."
    )
    low_mmr = v.msi_low_mmr_listing(crc)
    tables_listing = {"colorectal_msi_low_mmr": low_mmr, "melanoma_non_uv": non_uv}
    extras = {
        "msi_low_mmr": low_mmr,
        "melanoma_non_uv": non_uv,
        "non_uv_fields": non_uv_fields,
        "non_uv_biopsy_site": biopsy_counts,
        "skcm_fields_note": skcm_note,
    }

    flow_parts = []
    for k, p in LUNG.items():
        qc = read_tsv(tabdir / f"fit_qc_{p}_all20.tsv", index_col=0)
        g = v.lung_flow_groups(clin[p])
        flow_parts.append(v.cohort_flow(k, g, matrix_totals(proc, p, ""), qc.index, qc["cosine"]))
    qc = read_tsv(tabdir / f"fit_qc_{SKCM}_all20.tsv", index_col=0)
    flow_parts.append(
        v.cohort_flow(
            "SKCM",
            v.skcm_flow_groups(clin[SKCM]),
            matrix_totals(proc, SKCM, ""),
            qc.index,
            qc["cosine"],
        )
    )
    for k, p in CRC.items():
        qc = read_tsv(tabdir / f"fit_qc_{p}_all20.tsv", index_col=0)
        g = v.colorectal_flow_groups(clin[p])
        flow_parts.append(v.cohort_flow(k, g, matrix_totals(proc, p, ""), qc.index, qc["cosine"]))
    flow = pd.concat(flow_parts, ignore_index=True)
    tables_extra = {"cohort_flow": flow}
    borderline = pd.DataFrame(
        [
            r
            for part, (k, p) in zip(crc_parts, CRC.items(), strict=True)
            for r in v.borderline_counts(clin[p], part, k)
        ]
    )
    tables_extra["colorectal_borderline"] = borderline
    tables_extra.update(tables_listing)
    for name, df in tables_extra.items():
        tabdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(tabdir / f"validation_{name}.tsv", sep="\t", index=False)

    for name, df in tables.items():
        tabdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(tabdir / f"validation_{name}.tsv", sep="\t", index=False, float_format="%.6g")

    # ---------------- figures ----------------
    def pick(df: pd.DataFrame, **kw) -> pd.Series:
        sel = df
        for k, val in kw.items():
            sel = sel[sel[k] == val]
        return sel.iloc[0]

    spectra = {}
    for name, p in all_projects.items():
        m = load_matrix(proc, p, "", clin[p]["barcode"])
        m = m.loc[:, m.sum(axis=0) >= 50]
        spectra[name] = m.div(m.sum(axis=0), axis=1)
    v.plot_mean_spectra(spectra, figdir / "fig1_mean_spectra.png", SBS96_PLOT_ORDER)
    lt = tables["lung_sbs4"]
    lstats = {
        k: fmt_stats(
            pick(
                lt,
                cohort=k,
                setting="primary",
                analysis="lung_sbs4_primary_contrast",
                comparison="never vs smoker (current + reformed <=15y); fit with SBS18",
            )
        )
        for k in LUNG
    }
    v.plot_lung_groups(lung, lstats, figdir / "fig2_lung_sbs4_by_smoking.png")
    dr = tables["lung_dose_response"]
    dstats = {}
    for k in LUNG:
        r = pick(dr, cohort=k, setting="primary", label="secondary")
        dstats[k] = (
            f"n={int(r['n1'])}; Spearman rho={r['effect']:.2f} "
            f"({r['ci_low']:.2f}-{r['ci_high']:.2f}); p={r['p']:.2g}"
        )
    v.plot_pack_years(lung, dstats, figdir / "fig3_sbs4_vs_pack_years.png")
    v.plot_melanoma(skcm, ct, SKCM_SIGS, figdir / "fig4_melanoma_uv.png")
    cr = tables["colorectal_mmr"]
    crow = pick(cr, cohort="COAD+READ", setting="primary", label="primary")
    v.plot_colorectal(crc, fmt_stats(crow), figdir / "fig5_colorectal_mmr.png")

    drops_path = tabdir / "profile_drops_summary.tsv"
    drops = read_tsv(drops_path) if drops_path.is_file() else None
    text = v.build_summary(tables, flow, drops, borderline, extras)
    v.write_markdown(res / "validation_summary.md", text)
    print(f"wrote {len(tables)} tables, 5 figures and summary under {res}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
