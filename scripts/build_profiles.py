#!/usr/bin/env python
"""Build the SBS96 profile of each chosen tumour (one per case, D2) for every project.

Inputs
    results/tables/samples_<project>.tsv   chosen sample per case (file_id + barcode)
    data/raw/maf/<project>/manifest.tsv    file_id -> file_name
    data/raw/maf/<project>/<file_name>     only the chosen files are read
    config/config.yaml                     profile.adjacent_snvs (default), projects

Outputs
    data/processed/sbs96_<project>[_count_as_sbs].tsv            96 x tumours (gitignored)
    results/tables/profile_drops_<project>[_count_as_sbs].tsv    per-tumour kept / drops / events
    results/tables/profile_drops_summary[_count_as_sbs].tsv      per-project totals, % SNP rows

Each MAF is restricted to the chosen barcode BEFORE profiling, so other aliquots that share
a file never leak into a tumour's counts. Run:  uv run python scripts/build_profiles.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fingerprints import profile as P  # noqa: E402

logger = logging.getLogger("build_profiles")

SAMPLE_COL = "Tumor_Sample_Barcode"
DEFAULT_MODE = "exclude"


def mode_suffix(adjacent_snvs: str) -> str:
    """File-name suffix for an adjacent-SNV mode: '' for the default run."""
    return "" if adjacent_snvs == DEFAULT_MODE else f"_{adjacent_snvs}"


def chosen_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """Rows of the samples table that are chosen (handles bool or 'True' strings)."""
    flag = samples["chosen"].astype(str).str.lower() == "true"
    return samples[flag].reset_index(drop=True)


def chosen_file_path(file_id: str, manifest: pd.DataFrame, maf_dir: Path) -> Path:
    """Path of the MAF for ``file_id`` via the manifest's file_name."""
    hit = manifest.loc[manifest["file_id"] == file_id, "file_name"]
    if len(hit) != 1:
        raise ValueError(f"file_id {file_id} found {len(hit)} times in manifest (need exactly 1)")
    return maf_dir / Path(hit.iloc[0]).name


def restrict_to_barcode(maf: pd.DataFrame, barcode: str) -> pd.DataFrame:
    """Keep only rows of the chosen tumour barcode; error if it is absent."""
    sub = maf[maf[SAMPLE_COL].astype(str) == barcode]
    if sub.empty:
        raise ValueError(
            f"barcode {barcode} not present in MAF (found: {sorted(set(maf[SAMPLE_COL]))[:5]})"
        )
    return sub.reset_index(drop=True)


def read_adjacent_mode(config: dict) -> str:
    return (config.get("profile") or {}).get("adjacent_snvs", DEFAULT_MODE)


def build_project(
    project: str, config: dict, root: Path, adjacent_snvs: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (96 x tumours matrix, per-tumour drop table) for one project."""
    maf_dir = root / config["paths"]["raw_maf"] / project
    manifest = pd.read_csv(maf_dir / "manifest.tsv", sep="\t", dtype=str)
    samples = pd.read_csv(
        root / "results" / "tables" / f"samples_{project}.tsv", sep="\t", dtype=str
    )
    chosen = chosen_samples(samples)
    if chosen["barcode"].duplicated().any():
        raise ValueError(f"{project}: duplicate chosen barcodes")
    cols, drops = {}, []
    for i, r in enumerate(chosen.itertuples(index=False), 1):
        path = chosen_file_path(r.file_id, manifest, maf_dir)
        maf = restrict_to_barcode(P.read_maf(path), r.barcode)
        mat, drop = P.build_sbs96_matrix(
            maf, sample_col=SAMPLE_COL, return_dropped=True, adjacent_snvs=adjacent_snvs
        )
        cols[r.barcode] = mat[r.barcode]
        d = drop.loc[[r.barcode]].reset_index().rename(columns={SAMPLE_COL: "tumour"})
        d.insert(1, "project", project)
        d.insert(2, "case_submitter_id", r.case_submitter_id)
        d.insert(3, "file_id", r.file_id)
        drops.append(d)
        if i % 100 == 0:
            logger.info("%s: %d/%d tumours", project, i, len(chosen))
    matrix = pd.DataFrame(cols, index=P.SBS96_CHANNELS)
    matrix.index.name = "MutationType"
    return matrix, pd.concat(drops, ignore_index=True)


def aggregate_drops(drops: pd.DataFrame, project: str) -> dict:
    """Project totals and % of input SNP rows dropped per reason.

    SNP rows = total input rows minus ``not_snp`` (indels / DNP / TNP / ONP). ``pct_*``
    columns are percentages of SNP rows; ``not_snp`` is shown as a % of ALL rows.
    """
    total = int(drops["total"].sum())
    not_snp = int(drops["not_snp"].sum())
    snp_rows = total - not_snp
    kept = drops["kept"]
    row = {
        "project": project,
        "n_tumours": len(drops),
        "total_rows": total,
        "snp_rows": snp_rows,
        "kept_snvs": int(kept.sum()),
        "median_snvs": float(kept.median()),
        "q1_snvs": float(kept.quantile(0.25)),
        "q3_snvs": float(kept.quantile(0.75)),
        "n_zero_snv_tumours": int((kept == 0).sum()),
        "pct_kept_of_snp_rows": 100 * int(kept.sum()) / snp_rows if snp_rows else float("nan"),
    }
    for reason in P.DROP_REASONS:
        row[reason] = int(drops[reason].sum())
    for reason in P.DROP_REASONS:
        if reason == "not_snp":
            row["pct_not_snp_of_all_rows"] = 100 * not_snp / total if total else float("nan")
        else:
            row[f"pct_{reason}"] = 100 * row[reason] / snp_rows if snp_rows else float("nan")
    row["dbs_events"] = int(drops["dbs_events"].sum())
    row["mnv_events"] = int(drops["mnv_events"].sum())
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "config.yaml")
    ap.add_argument(
        "--adjacent-snvs",
        choices=P.ADJACENT_SNV_MODES,
        default=None,
        help="override profile.adjacent_snvs from the config",
    )
    ap.add_argument("--projects", nargs="+", default=None, help="default: all in the config")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = yaml.safe_load(args.config.read_text())
    mode = args.adjacent_snvs or read_adjacent_mode(config)
    suffix = mode_suffix(mode)
    all_projects = [p["id"] for p in config["projects"]]
    projects = args.projects or all_projects
    root = args.root
    tables = root / "results" / "tables"
    processed = root / config["paths"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    logger.info("adjacent_snvs=%s (suffix %r)", mode, suffix)

    for project in projects:
        t0 = time.time()
        matrix, drops = build_project(project, config, root, mode)
        matrix.to_csv(processed / f"sbs96_{project}{suffix}.tsv", sep="\t")
        drops.to_csv(tables / f"profile_drops_{project}{suffix}.tsv", sep="\t", index=False)
        s = aggregate_drops(drops, project)
        zero = drops.loc[drops["kept"] == 0, "tumour"].tolist()
        logger.info(
            "%s: %d tumours, SNVs median %.0f (IQR %.0f-%.0f), zero-SNV tumours %d %s, %.0fs",
            project,
            s["n_tumours"],
            s["median_snvs"],
            s["q1_snvs"],
            s["q3_snvs"],
            s["n_zero_snv_tumours"],
            zero,
            time.time() - t0,
        )

    # Aggregate from the per-project files on disk so partial reruns stay consistent.
    rows = []
    for project in all_projects:
        f = tables / f"profile_drops_{project}{suffix}.tsv"
        if f.is_file():
            rows.append(aggregate_drops(pd.read_csv(f, sep="\t"), project))
    if rows:
        out = tables / f"profile_drops_summary{suffix}.tsv"
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False, float_format="%.4g")
        logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
