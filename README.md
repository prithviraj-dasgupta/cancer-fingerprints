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

## Method in one line

Each tumour's 96-channel SBS profile is decomposed into a cohort-specific subset of COSMIC
reference signatures by **non-negative least squares (NNLS) signature refitting**
(`scipy.optimize.nnls`; no sparsity penalty). SigProfilerAssignment supplies only the bundled
COSMIC v3.4 reference file and an optional cross-check. This is refitting to a published
reference, not a machine-learning model trained on COSMIC data.

Exposures are NNLS-attributed mutation counts. Their sum can differ from the observed SNV count
when the fit is poor, so the per-tumour reconstruction cosine is reported with every result.

## How to reproduce

```bash
uv sync --extra signatures      # needs the COSMIC reference bundled with SigProfilerAssignment
bash scripts/run_pipeline.sh    # download -> select samples -> profiles -> fits -> validation
```

Downloads are open-access GDC masked MAFs (TCGA, GRCh38). Raw data stay in `data/` (gitignored).
Every table and figure below is written by a script to `results/`; the full generated report is
[results/validation_summary.md](results/validation_summary.md).

## Results: does the fit recover known biology?

The analyses and settings were fixed before the results were seen (see `docs/decisions.md`, D7).
Tumours with at least 50 SNVs are analysed; 20 and 100 SNVs and a fit-quality filter are
sensitivity checks. These results are *consistent with* the expected biology; they do not prove
the cause of any individual tumour.

| Check | Result (primary setting) |
|---|---|
| **Lung (LUAD):** SBS4 (tobacco) in smokers vs lifelong never-smokers | median 0.50 (n=273) vs 0.11 (n=33); AUC 0.82 (95% CI 0.73-0.90), p=2.4e-9 |
| **Melanoma (SKCM):** summed UV signatures SBS7a-d | median 0.91 of mutations (n=417); the largest component in 94.7% of tumours |
| **Colorectal (COAD+READ):** summed mismatch-repair signatures, MSI vs MSS | median 0.60 (n=80) vs 0.09 (n=458); AUC 0.97 (95% CI 0.95-0.99) |

Things the numbers do *not* show cleanly (all reported in the full summary):

- **SNV filter and never-smokers.** The 50-SNV filter ignores smoking status but removes about half
  of LUAD never-smokers (69 chosen, 63 with 20+ SNVs, 33 with 50+, 17 with 100+). The effect size
  is 0.88 at 20 SNVs and 0.72 at 100. The primary stays at 50, as pre-specified.
- **Holds across thresholds.** The LUAD smoker vs never-smoker AUC is 0.88 at 20 SNVs, 0.82 at 50 and
  0.72 at 100 (every CI lower bound above 0.5). The 50-SNV primary was fixed in advance and was not
  changed after seeing results.
- **SBS4 in never-smokers.** A minority of reported never-smokers show tobacco-like SBS4 exposure
  (11 of 33 LUAD never-smokers have an SBS4 fraction above 0.25). Possible reasons include
  misreported smoking history, second-hand smoke, or overlap between SBS4 and similar signatures;
  this analysis cannot tell these apart. For context, whole-genome sequencing of never-smoker lung
  cancers found no strong tobacco signatures, even with second-hand smoke exposure (Zhang et al.
  2021, *Nat Genet*).
- **No dose-response in the SBS4 fraction.** SBS4 fraction does not track pack-years in ever-smokers
  (Spearman rho 0.03 in LUAD, -0.01 in LUSC). This is expected: a fraction is compositional and
  pack-years are self-reported and noisy. SBS4 mutation *count* vs pack-years is a secondary test:
  rho 0.15 in LUAD (p=0.0075, unadjusted for multiple tests; counts also track overall mutation
  burden) and 0.04 in LUSC (p=0.47). LUSC has only 14 never-smokers at 50+ SNVs, so it is
  descriptive.
- **Background MMR in MSS tumours.** Median summed MMR is 0.09 in MSS tumours. This is most likely
  SBS1/SBS6 overlap plus the noise floor of NNLS refitting, not real mismatch-repair deficiency.
- **MSI prevalence differs by site, as expected.** MSI is 17.5% in colon (COAD, 75 of 428) and 5.8%
  in rectum (READ, 9 of 155); MSI is much rarer in rectal cancer. READ has only 6 MSI tumours at
  50+ SNVs, so its result is imprecise. Seven MSI tumours have low summed MMR (below 0.25); they are
  listed with MSI score and mutation burden in
  `results/tables/validation_colorectal_msi_low_mmr.tsv`. Possible reasons include MSIsensor2 false
  calls, low mutation counts and POLE-like signatures; none is established here.
- **Melanomas without a UV-dominant profile are flagged, not excluded.** 22 of 417 SKCM tumours have
  summed SBS7a-d at or below 0.5 (`results/tables/validation_melanoma_non_uv.tsv`). The downloaded
  clinical fields hold only histology and the sampled site, not acral/mucosal subtype, so we cannot
  yet say whether these are acral or mucosal melanomas.
- **Cases without data.** 90 clinical cases (LUAD 26, LUSC 14, COAD 33, READ 17) have no open-access
  masked MAF at GDC (only controlled-access or germline calls), so they cannot be analysed.
- **Filters that did nothing.** Tri-allelic and multi-allelic filters removed no mutations in these
  GDC masked MAFs, which already hold one alternate allele per site. They matter for other inputs.

## Demo (local only)

The Streamlit demo runs locally only. Under COSMIC's licence terms, COSMIC data may not be
exposed on a free-to-access platform, so there is no public hosted version. Screenshots will be
added here. Permission will be requested from COSMIC before any hosting.

Method and scope decisions are recorded in [docs/decisions.md](docs/decisions.md).

## Signatures and citations

- Reference signatures: **COSMIC SBS v3.4**, GRCh38, exome-adjusted (TCGA data are whole-exome).
  They come from the copy bundled with SigProfilerAssignment (the optional `signatures` extra,
  `uv sync --extra signatures`), and the version is pinned in `config/`.
- COSMIC files are not committed to this repository. COSMIC's licence terms apply to them.

References (checked against PubMed on 2026-11-22; PMIDs given so they can be re-checked):

1. **Mutational signatures:** Alexandrov L.B. et al. (2020) The repertoire of mutational
   signatures in human cancer. *Nature* 578:94–101. doi:10.1038/s41586-020-1943-3.
   PMID 32025018.
2. **COSMIC** (<https://cancer.sanger.ac.uk/signatures/>): Sondka Z. et al. (2024) COSMIC: a
   curated database of somatic variants and clinical data for cancer. *Nucleic Acids Res*
   52(D1):D1210–D1217. doi:10.1093/nar/gkad986. PMID 38183204.
3. **SigProfilerAssignment:** Díaz-Gay M. et al. (2023) Assigning mutational signatures to individual samples and individual
   somatic mutations with SigProfilerAssignment. *Bioinformatics* 39(12):btad756.
   doi:10.1093/bioinformatics/btad756. PMID 38096571.
4. **Never-smoker lung cancer genomes (context):** Zhang T. et al. (2021) Genomic and evolutionary
   classification of lung cancer in never smokers. *Nat Genet* 53(9):1348–1359.
   doi:10.1038/s41588-021-00920-0. PMID 34493867.

## Known limitations

- **Colibactin (SBS88) is not fitted.** It is a COSMIC v3.4 signature with colorectal relevance
  (Pleguezuelos-Manzano et al. 2020), but it was published after Alexandrov et al. 2020, on which
  the cohort signature subsets are based. SBS36 and SBS10c/d are also not fitted.
- **SBS1 and SBS6 overlap** (cosine similarity about 0.9), so some low-level SBS6 ("mismatch
  repair") exposure in microsatellite-stable colorectal tumours may really be SBS1.
- **MSI labels are bioinformatics-derived** (GDC MSIsensor2 calls, 20% cutoff), not clinical MSI
  or mismatch-repair tests.
- **Signature subsets rest on figure estimates.** Prevalences were read from Extended Data
  Fig. 5 of Alexandrov et al. 2020 and are approximate. The paper has no TCGA-only breakdown.
- **Signature activity is not a diagnosis.** Refitting reports which known signatures best
  explain a tumour's mutations; it does not prove a cause.

## Disclaimer

This project is for research and education only. It is not a medical device and must not be
used for diagnosis or to guide treatment.
