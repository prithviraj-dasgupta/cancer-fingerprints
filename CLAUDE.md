# Cancer Fingerprints

> Reads a tumour's mutations and explains, in plain English, what most likely caused them.

## Project overview

Different causes of cancer leave different patterns of mutation in DNA. Tobacco smoke
tends to cause C>A changes, UV light causes C>T changes at dipyrimidines, and some
DNA-repair defects leave their own distinctive patterns. These recurring patterns are
called **mutational signatures**, and they act like fingerprints at a crime scene.

This project:
1. takes a tumour's somatic mutations
2. builds its 96-channel single-base-substitution (SBS) profile
3. fits that profile against the published COSMIC reference signatures
4. reports what fraction of mutations each likely cause explains, in plain language,
   e.g. "~60% tobacco smoking, ~25% ageing, ~15% DNA-repair defect".

**Purpose:** a portfolio project on GitHub for Prithviraj Dasgupta (cancer biology /
bioinformatics background; targeting UK Research Associate, Assistant Scientist and
bioinformatics roles). It should show real cancer genomics skill *and* clear AI/ML and
software engineering practice. It is for research and education only, not diagnosis.

## Scope (v1)

- **Cancer types:** lung (TCGA-LUAD, TCGA-LUSC), skin melanoma (TCGA-SKCM), colorectal (TCGA-COAD/READ)
- **Data:** GDC **open-access** masked somatic mutation files (MAF) from TCGA, GRCh38; clinical tables
  (e.g. smoking history for lung)
- **Signatures:** COSMIC SBS reference signatures (v3.x). Check the licence before redistributing any
  COSMIC files; download them in a script rather than committing them to the repo
- **Validation (the key result):**
  - lung tumours from smokers show more SBS4 (tobacco) than lung tumours from never-smokers
  - melanomas are dominated by SBS7a/b (UV)
  - colorectal MSI/hypermutated tumours show mismatch-repair signatures (e.g. SBS6/15/26/44)
- **Demo:** Streamlit app. Upload a MAF/VCF → mutation spectrum + signature pie chart + plain-English summary
- **Out of scope for v1:** de novo signature extraction, DBS/ID signatures, copy-number signatures

## Planned stack

- Python 3.11+, managed with `uv` (or conda if genome tools need it)
- `SigProfilerMatrixGenerator` (96-channel matrices) and `SigProfilerAssignment` (fitting to COSMIC)
- pandas, numpy, matplotlib/seaborn (static figures), plotly (app)
- Streamlit for the demo
- pytest for tests; ruff for lint and format
- Snakemake for the end-to-end pipeline (optional; plain scripts first if simpler)

## Suggested repo layout

```
cancer-fingerprints/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── config/            # cancer types, paths, signature version
├── scripts/           # download_data.py, download_signatures.py
├── src/fingerprints/  # profile.py, fit.py, explain.py, plots.py
├── app/               # streamlit_app.py
├── notebooks/         # exploration only; final logic lives in src/
├── tests/
├── data/              # gitignored: raw/, processed/
└── results/           # figures and tables used in the README
```

## Working rules

- Never commit raw data, genome references or large files. `data/` stays gitignored.
- Every result in the README must come from a script in the repo (reproducible).
- Plain-English explanations live in one mapping file (signature → aetiology text),
  with the COSMIC aetiology cited, so the biology can be reviewed in one place.
- Say clearly where an aetiology is "unknown" or "proposed". Don't overstate.
- Keep the app usable with a small bundled **synthetic** example file so anyone can try it.
- Small, reviewable commits with clear messages.
- Prithviraj reviews the biology; flag anything biologically uncertain for him.

## Milestones (target: Dec 20, 2026)

1. **Setup:** repo scaffold, environment, data and signature download scripts
2. **Profiles:** MAF → 96-channel SBS matrix per tumour, with tests on a tiny fixture
3. **Fitting:** assign COSMIC signatures per tumour; save exposures table
4. **Validation:** smokers vs never-smokers (SBS4), melanoma UV (SBS7), colorectal MMR. Figures + stats
5. **Explain:** plain-English report generator from exposures
6. **App:** Streamlit demo with upload + example file
7. **README:** the "crime-scene" story, figures, method, limitations, how to run, disclaimer

## Context

- This repo may also be worked on by Paperclip agents (Claude Code adapter), with this file as
  their brief. Paperclip runs locally (UI at http://127.0.0.1:3100).
