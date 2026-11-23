#!/usr/bin/env bash
# Full reproducible order for Cancer Fingerprints (Phase 2). Run from anywhere:
#   bash scripts/run_pipeline.sh
# Every step is idempotent: downloads skip files already present (checksum-checked), and the
# analysis steps overwrite their outputs deterministically. Needs `uv sync --extra signatures`
# done once beforehand (this script never runs `uv sync`). Network is needed for step 1 only.
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n== %s ==\n' "$1"; }

step "1. download_data (GDC open-access MAFs, clinical, MSI metadata)"
uv run python scripts/download_data.py

step "2. download_signatures (COSMIC v3.4 GRCh38 exome copy from SigProfilerAssignment)"
uv run python scripts/download_signatures.py

step "3. select_samples (one tumour per case, D2; smoking / MSI groups)"
uv run python scripts/select_samples.py

step "4. build_profiles (SBS96 per chosen tumour; main + melanoma count_as_sbs sensitivity)"
uv run python scripts/build_profiles.py
uv run python scripts/build_profiles.py --adjacent-snvs count_as_sbs --projects TCGA-SKCM

step "5. run_fits (NNLS refit; main, melanoma count_as_sbs, with/without SBS18)"
uv run python scripts/run_fits.py
uv run python scripts/run_fits.py --adjacent-snvs count_as_sbs --projects TCGA-SKCM
uv run python scripts/run_fits.py --drop-signature SBS18 \
    --projects TCGA-LUAD TCGA-LUSC TCGA-COAD TCGA-READ

step "6. run_validation (smoking SBS4, melanoma SBS7, colorectal MMR; figures + stats)"
uv run python scripts/run_validation.py

echo; echo "Pipeline finished."
