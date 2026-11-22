# Cancer Fingerprints

Reads a tumour's somatic mutations and explains, in plain English, what most likely caused them
(using COSMIC single-base-substitution mutational signatures).

**Status: work in progress.**

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync
uv run fingerprints --help
```

## Run tests and lint

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest                    # all tests
uv run pytest -m "not network"   # what CI runs (no internet, no genome reference)
```

## Input formats and limitations

- v1 reads GDC masked somatic mutation files (MAF) and takes each mutation's trinucleotide
  context from the MAF's `CONTEXT` column.
- MAFs without a `CONTEXT` column, and VCF input, need a reference-genome (GRCh38) context
  lookup. That is not supported yet; it is planned for the app phase.
- `fingerprints profile` stops with a clear error if the `CONTEXT` column is missing.

## Signatures and citations

- Reference signatures: **COSMIC SBS v3.4**, GRCh38, exome-adjusted (TCGA data are whole-exome).
  They come from the copy bundled with SigProfilerAssignment (the optional `signatures` extra,
  `uv sync --extra signatures`), and the version is pinned in `config/`.
- COSMIC files are not committed to this repository. COSMIC's licence terms apply to them.

References (checked against PubMed on 2026-11-22; PMIDs given so they can be re-checked):

1. COSMIC Mutational Signatures, <https://cancer.sanger.ac.uk/signatures/>.
   Sondka Z. et al. (2024) COSMIC: a curated database of somatic variants and clinical data for
   cancer. *Nucleic Acids Res* 52(D1):D1210–D1217. doi:10.1093/nar/gkad986. PMID 38183204.
2. Alexandrov L.B. et al. (2020) The repertoire of mutational signatures in human cancer.
   *Nature* 578:94–101. doi:10.1038/s41586-020-1943-3. PMID 32025018.
3. Díaz-Gay M. et al. (2023) Assigning mutational signatures to individual samples and individual
   somatic mutations with SigProfilerAssignment. *Bioinformatics* 39(12):btad756.
   doi:10.1093/bioinformatics/btad756. PMID 38096571.

## Disclaimer

This project is for research and education only. It is not a medical device and must not be
used for diagnosis or to guide treatment.
