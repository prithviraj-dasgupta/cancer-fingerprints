#!/usr/bin/env python
"""Run the NNLS cohort fits (D7) for each project and save exposure / QC tables.

Inputs
    data/processed/sbs96_<project>[_count_as_sbs].tsv    from scripts/build_profiles.py
    data/raw/signatures/COSMIC_v3.4_SBS_GRCh38_exome.txt   COSMIC reference (gitignored)
    config/config.yaml                                     fitting.* (approved subsets)

Outputs (results/tables/, suffix = '' | '_count_as_sbs' | '_noSBS18' | both, in that order)
    exposures_<P><sfx>.tsv, exposures_<P><sfx>_fractions.tsv, fit_qc_<P><sfx>.tsv,
    fit_summary_<P><sfx>.tsv, excluded_<P><sfx>.tsv   main threshold (fitting.min_snvs, 50)
    exposures_<P><sfx>_all20.tsv, ..._all20_fractions.tsv, fit_qc_<P><sfx>_all20.tsv
        every tumour with >= 20 SNVs (the lowest sensitivity threshold). NNLS is per tumour and
        independent, so these exposures are identical to a fit at 50 or 100: validation picks
        its min_snvs (20 / 50 / 100) by filtering fit_qc_*_all20.tsv on ``total_snvs``.
    fit_summary_all<sfx>.tsv   one row per project (exclusions, cosine statistics)
    mean_fractions.tsv         mean fraction per signature per project (main run only)

Run:  uv run python scripts/run_fits.py
      uv run python scripts/run_fits.py --adjacent-snvs count_as_sbs --projects TCGA-SKCM
      uv run python scripts/run_fits.py --drop-signature SBS18 --projects TCGA-LUAD TCGA-LUSC
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fingerprints import fit as F  # noqa: E402

logger = logging.getLogger("run_fits")

DEFAULT_MODE = "exclude"
LOW_COSINE = 0.8
ALL_FLOOR = 20  # lowest sensitivity threshold: all tumours at or above it are fitted once


def output_suffix(adjacent_snvs: str, drop_signature: str | None) -> str:
    """'' for the main run; '_count_as_sbs', '_noSBS18' or both (in that order)."""
    sfx = "" if adjacent_snvs == DEFAULT_MODE else f"_{adjacent_snvs}"
    if drop_signature:
        sfx += f"_no{drop_signature}"
    return sfx


def profile_path(processed: Path, project: str, adjacent_snvs: str) -> Path:
    sfx = "" if adjacent_snvs == DEFAULT_MODE else f"_{adjacent_snvs}"
    return processed / f"sbs96_{project}{sfx}.tsv"


def config_without_signature(config: dict, project: str, signature: str | None) -> dict:
    """Deep copy of the config with ``signature`` removed from the project's subset."""
    cfg = copy.deepcopy(config)
    if signature is None:
        return cfg
    subset = list(cfg["fitting"]["signature_subsets"][project])
    if signature not in subset:
        raise ValueError(f"{signature} is not in the {project} subset: {subset}")
    cfg["fitting"]["signature_subsets"][project] = [s for s in subset if s != signature]
    return cfg


def with_total_snvs(result: F.FitResult) -> F.FitResult:
    """Add ``total_snvs`` (= observed SNVs, SBS96 total) to the QC table."""
    qc = result.qc.copy()
    qc["total_snvs"] = qc["observed_snvs"].astype(int)
    return dataclasses.replace(result, qc=qc)


def overview_row(result: F.FitResult, matrix: pd.DataFrame) -> dict:
    """One summary row per project: exclusions at each threshold, cosine statistics."""
    totals = matrix.sum(axis=0)
    cos = result.cosine
    row = {
        "project": result.project,
        "n_tumours_input": result.n_input,
        "n_signatures": len(result.signatures),
        "n_excluded_lt20": int((totals < 20).sum()),
        "n_excluded_lt50": int((totals < 50).sum()),
        "n_excluded_lt100": int((totals < 100).sum()),
        "n_fitted_main": len(result.exposures),
        "mean_cosine": float(cos.mean()),
        "median_cosine": float(cos.median()),
        "frac_cosine_lt_0.8": float((cos < LOW_COSINE).mean()),
        "n_cosine_lt_0.8": int((cos < LOW_COSINE).sum()),
    }
    return row


def fit_project(
    project: str, config: dict, matrix: pd.DataFrame, reference: pd.DataFrame, out: Path, sfx: str
) -> dict:
    """Fit one project; write main and all20 outputs; return its overview row."""
    main = with_total_snvs(F.run_cohort_fit(project, config, matrix, reference))
    all20 = with_total_snvs(
        F.run_cohort_fit(project, config, matrix, reference, min_snvs=ALL_FLOOR)
    )
    F.save_results(dataclasses.replace(main, project=f"{project}{sfx}"), out)
    name = f"{project}{sfx}_all{ALL_FLOOR}"
    all20.exposures.to_csv(out / f"exposures_{name}.tsv", sep="\t", float_format="%.6g")
    all20.fractions.to_csv(out / f"exposures_{name}_fractions.tsv", sep="\t", float_format="%.6g")
    all20.qc.to_csv(out / f"fit_qc_{name}.tsv", sep="\t", float_format="%.6g")
    row = overview_row(main, matrix)
    row["project"] = f"{project}{sfx}"
    # consistency: tumours in main must be the >=50 subset of all20, with identical exposures
    common = main.exposures.index
    diff = (
        (main.exposures - all20.exposures.loc[common]).abs().to_numpy().max() if len(common) else 0
    )
    if diff > 1e-9:
        raise RuntimeError(f"{project}: main and all20 exposures differ (max {diff})")
    logger.info(
        "%s: input %d, fitted %d (>=50), %d (>=%d); excluded <20/<50/<100: %d/%d/%d; "
        "cosine mean %.3f median %.3f, <0.8: %d (%.1f%%)",
        row["project"],
        row["n_tumours_input"],
        row["n_fitted_main"],
        len(all20.exposures),
        ALL_FLOOR,
        row["n_excluded_lt20"],
        row["n_excluded_lt50"],
        row["n_excluded_lt100"],
        row["mean_cosine"],
        row["median_cosine"],
        row["n_cosine_lt_0.8"],
        100 * row["frac_cosine_lt_0.8"],
    )
    return row


def mean_fractions(tables: Path, projects: list[str]) -> pd.DataFrame:
    """Mean fraction per signature (rows) per project (columns), from the main fractions files."""
    cols = {}
    for p in projects:
        f = tables / f"exposures_{p}_fractions.tsv"
        if f.is_file():
            cols[p] = pd.read_csv(f, sep="\t", index_col=0).mean(axis=0)
    out = pd.DataFrame(cols).fillna(0.0)
    out.index.name = "signature"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "config.yaml")
    ap.add_argument("--adjacent-snvs", choices=["exclude", "count_as_sbs"], default=DEFAULT_MODE)
    ap.add_argument(
        "--drop-signature",
        default=None,
        metavar="SBSxx",
        help="refit with this signature removed from each project's subset",
    )
    ap.add_argument("--projects", nargs="+", default=None, help="default: all in the config")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = yaml.safe_load(args.config.read_text())
    root = args.root
    all_projects = [p["id"] for p in config["projects"]]
    projects = args.projects or all_projects
    sfx = output_suffix(args.adjacent_snvs, args.drop_signature)
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ref_path = root / config["paths"]["raw_signatures"] / config["signatures"]["file"]
    reference = F.load_reference(ref_path)
    logger.info("suffix %r, projects %s", sfx, projects)

    rows = []
    for project in projects:
        t0 = time.time()
        pf = profile_path(root / config["paths"]["processed"], project, args.adjacent_snvs)
        if not pf.is_file():
            raise FileNotFoundError(f"{pf} missing (run scripts/build_profiles.py first)")
        matrix = pd.read_csv(pf, sep="\t", index_col=0)
        cfg = config_without_signature(config, project, args.drop_signature)
        rows.append(fit_project(project, cfg, matrix, reference, tables, sfx))
        logger.info("%s done in %.1fs", project, time.time() - t0)

    # Overview and mean fractions are rebuilt from the project files on disk.
    ov = pd.DataFrame(rows)
    prev = tables / f"fit_summary_all{sfx}.tsv"
    if prev.is_file():
        old = pd.read_csv(prev, sep="\t")
        ov = pd.concat([old[~old["project"].isin(ov["project"])], ov])
    order = {f"{p}{sfx}": i for i, p in enumerate(all_projects)}
    ov = ov.assign(_o=ov["project"].map(order)).sort_values("_o").drop(columns="_o")
    ov.to_csv(prev, sep="\t", index=False, float_format="%.4g")
    if sfx == "":
        mf = mean_fractions(tables, all_projects)
        mf.to_csv(tables / "mean_fractions.tsv", sep="\t", float_format="%.4f")
        logger.info("wrote mean_fractions.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
