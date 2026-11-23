#!/usr/bin/env python
"""Select one tumour sample per patient and derive smoking / MSI groups (D2, D3, D4).

Reads data/raw (manifests, MAFs, clinical, msi) and writes small, committed tables:

    results/tables/samples_<project>.tsv    every candidate sample, with chosen + reason (the log)
    results/tables/clinical_<project>.tsv   one row per CHOSEN case, with derived columns

Rules live in src/fingerprints/samples.py. Run:  uv run python scripts/select_samples.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fingerprints import samples as S  # noqa: E402


def process(project: str, cfg: dict, root: Path, out_dir: Path) -> dict:
    raw_maf = root / cfg["paths"]["raw_maf"] / project
    raw_clin = root / cfg["paths"]["raw_clinical"] / project
    manifest = pd.read_csv(raw_maf / "manifest.tsv", sep="\t", dtype=str)
    barcodes = {
        r.file_id: S.read_maf_barcodes(raw_maf / Path(r.file_name).name)
        for r in manifest.itertuples(index=False)
    }
    samples = S.select_samples(S.collect_candidates(manifest, barcodes))
    samples.to_csv(out_dir / f"samples_{project}.tsv", sep="\t", index=False)

    # Same barcode in more than one MAF file: log them in a separate new table (the
    # samples table above is unchanged) and compare file contents.
    with_dups = S.collect_candidates(manifest, barcodes, log_duplicates=True)
    file_name = dict(zip(manifest["file_id"], manifest["file_name"], strict=True))
    dups = S.check_duplicate_contents(
        with_dups,
        lambda fid, bc: S.maf_content_hash(raw_maf / Path(file_name[fid]).name, bc),
        strict=False,
    )
    dups = dups.assign(reason="duplicate_barcode_other_file")
    dups.to_csv(out_dir / f"duplicate_barcodes_{project}.tsv", sep="\t", index=False)

    chosen = S.chosen_only(samples)
    clinical = pd.read_csv(raw_clin / "clinical.tsv", sep="\t", dtype=str)
    keep = [
        "case_id",
        "gender",
        "age_at_index",
        "vital_status",
        "primary_diagnosis",
        "ajcc_pathologic_stage",
        "tobacco_smoking_status",
        "pack_years_smoked",
        "tobacco_status_conflict",
    ]
    df = chosen.drop(columns=["reason", "chosen"]).merge(
        clinical[[c for c in keep if c in clinical]], on="case_id", how="left"
    )
    summary: dict = {
        "project": project,
        "maf_files": len(manifest),
        "cases_with_maf": int(samples["case_id"].nunique()),
        "candidates": len(samples),
        "chosen": len(chosen),
        "chosen_by_type": chosen["sample_type_code"].value_counts().to_dict(),
        "duplicate_barcode_rows": len(dups),
        "duplicate_barcode_content_differs": int((~dups["content_identical"]).sum()),
        "dropped_patients": int(samples["case_id"].nunique() - len(chosen)),
        "dropped_empty_maf_only": int(
            (
                samples.groupby("case_id")["reason"].apply(
                    lambda r: (r == "empty_maf_no_barcode").all()
                )
            ).sum()
        ),
        "dropped_no_01_or_06": int(
            samples.groupby("case_id")["reason"]
            .apply(lambda r: (r == "dropped_no_01_or_06").any())
            .sum()
        ),
        "clinical_cases": int(clinical["case_id"].nunique()),
        "clinical_without_maf": int(len(set(clinical["case_id"]) - set(manifest["case_id"]))),
        "dropped_by_type": samples[samples["reason"] == "dropped_no_01_or_06"]["sample_type_code"]
        .value_counts()
        .to_dict(),
        "cases_multiple_candidates": int((samples.groupby("case_id").size() > 1).sum()),
        "reasons": samples["reason"].value_counts().to_dict(),
        "cases_missing_clinical": int(df["gender"].isna().sum()) if "gender" in df else None,
    }
    if cfg_cancer_type(cfg, project) == "lung":
        df = S.add_smoking_groups(df)
        summary["smoking"] = S.smoking_counts(df)
    msi_path = raw_clin / "msi.tsv"
    if msi_path.exists():
        msi_raw = pd.read_csv(msi_path, sep="\t", dtype=str)
        msi = S.choose_msi(msi_raw)
        df = df.merge(msi, on="case_id", how="left")
        has = df["msi_status"].notna()
        no_maf = set(msi_raw["case_id"]) - set(manifest["case_id"])
        no_maf_status = (
            S.choose_msi(msi_raw[msi_raw["case_id"].isin(no_maf)]).set_index("case_id")[
                "msi_status"
            ]
            if no_maf
            else pd.Series(dtype=str)
        )
        summary["msi"] = {
            "chosen_cases": len(df),
            "with_msi_call": int(has.sum()),
            "no_msi_call": int((~has).sum()),
            "msi_cases_without_maf": len(no_maf),
            "msi_cases_without_maf_msi_h": int((no_maf_status == "MSI").sum()),
            "status": df["msi_status"].value_counts().to_dict(),
            "msi_fraction_of_called": (
                float((df["msi_status"] == "MSI").sum() / has.sum()) if has.any() else None
            ),
            "strategy": df["msi_strategy"].value_counts().to_dict(),
            "borderline": int(df["msi_borderline"].fillna(False).astype(bool).sum()),
            "wxs_wgs_disagree": int(df["msi_wxs_wgs_disagree"].fillna(False).astype(bool).sum()),
        }
    # Newer columns go last so existing column positions never move.
    new_cols = [c for c in ("site_of_resection_or_biopsy", "tumor_of_origin") if c in clinical]
    df = df.merge(clinical[["case_id", *new_cols]], on="case_id", how="left")
    df.to_csv(out_dir / f"clinical_{project}.tsv", sep="\t", index=False)
    return summary


def _plain(obj):
    """Make value_counts() output (numpy keys/values) JSON-safe."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj.item() if hasattr(obj, "item") else obj


def _flatten(d: dict, prefix: str = "") -> dict:
    """Nested summary -> flat columns (smoking.by_code.1 etc.); lists stay as JSON strings."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def cfg_cancer_type(cfg: dict, project: str) -> str:
    return next(p["cancer_type"] for p in cfg["projects"] if p["id"] == project)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "config.yaml")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="Root for data paths")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--projects", nargs="+")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    projects = args.projects or [p["id"] for p in cfg["projects"]]
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [process(p, cfg, args.root, args.out) for p in projects]
    flat = [_flatten(_plain(s)) for s in summaries]
    summary_df = pd.DataFrame(flat)
    for col in summary_df.columns:  # NaN for projects without MSI would turn counts into floats
        if summary_df[col].dtype == float and summary_df[col].dropna().mod(1).eq(0).all():
            summary_df[col] = summary_df[col].astype("Int64")
    summary_df.to_csv(args.out / "sample_selection_summary.tsv", sep="\t", index=False)
    print(json.dumps(_plain(summaries), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
