# Decisions

Method and scope decisions for Cancer Fingerprints, with who made them and when. Biology
decisions are made by Prithviraj Dasgupta. Each entry says where it is (or will be) implemented.

## 2026-11-22: Phase 1 review

### D1. Demo hosting and COSMIC licence
- The COSMIC terms say COSMIC data may not be exposed on a free-to-access platform. So **no
  public hosted demo for now**: the Streamlit app runs locally only, and the README shows
  screenshots or a GIF instead.
- Prithviraj will ask COSMIC for permission before any hosting.
- Fitting is described as **NNLS signature refitting** against the COSMIC reference, never as a
  trained ML model. COSMIC's terms also restrict using COSMIC data to train AI models.
- *Where:* README, app (Phase 6).

### D2. One tumour sample per patient
- TCGA MAFs are per aliquot, and some patients have several.
- Rule:
  1. Prefer a primary tumour (sample type code `-01`).
  2. If there is none, use a metastatic sample (`-06`).
  3. Otherwise drop the patient.
- The same rule applies to SKCM, so most SKCM cases will use `-06`.
- If a patient has several aliquots of the chosen type, pick deterministically: the lowest
  aliquot barcode.
- Every choice (kept and dropped, with the reason) is logged to a samples table.
- `profile.py` still counts per `Tumor_Sample_Barcode`; selection happens downstream.
- *Where:* sample selection step, Phase 2 / validation.

### D3. Smoking groups (lung SBS4 validation)
- **Primary contrast:** lifelong non-smokers (GDC tobacco code 1) vs current smokers (2) plus
  reformed smokers ≤15 years (4).
- **Sensitivity analysis:** all ever-smokers (codes 2–5) vs lifelong non-smokers.
- **Dose-response:** Spearman correlation of SBS4 with `pack_years_smoked` in ever-smokers.
- The main test is in **LUAD**. LUSC is reported descriptively, because it has only 18 lifelong
  non-smokers. There is also a pooled model adjusted for cancer type.
- Some overlap is expected, because SBS4 can appear in never-smokers. Results are not to be
  over-interpreted.
- Cases flagged with `tobacco_status_conflict` need a stated rule when the analysis is written.
- *Where:* validation, Milestone 4.

### D4. MSI status (colorectal MMR validation)
- Use GDC `msi_status` (MSI/MSS on the tumour BAM records).
- Before documenting it as MSIsensor-derived, check the GDC docs. That check is still to do;
  record the source once confirmed.
- When a case's WXS and WGS calls disagree, prefer WXS, because the MAFs are WXS.
- Sanity check: MSI-H should be roughly 10–20% of COAD, and those tumours should be hypermutated.
- *Where:* validation, Milestone 4.

### D5. Adjacent SNVs are doublets, not SBS
- Same-sample SNVs at adjacent positions are treated as doublet base substitutions (DBS).
- They are excluded from the SBS96 counts, and how many there were is reported, matching
  SigProfilerMatrixGenerator and COSMIC methods.
- DBS signatures themselves stay out of scope for v1.
- *Where:* `src/fingerprints/profile.py`, implemented in PR #1.

### D6. Tri-allelic sites are dropped
- Rows where both tumour alleles differ from the reference and from each other are dropped and
  counted under a named drop reason.
- *Where:* `src/fingerprints/profile.py`, implemented in PR #1.

### D7. Fitting settings (approved)
- Reference: COSMIC SBS **v3.4, GRCh38, exome-adjusted** (TCGA is whole-exome), from the copy
  bundled with SigProfilerAssignment 1.1.5. Fitting must pass `cosmic_version=3.4`, because the
  package defaults to 3.6.
- Skip tumours with **fewer than 50 SNVs** and report how many were excluded. Also report a
  sensitivity check at 20 and 100.
- Exclude possible sequencing-artefact signatures: **SBS27, SBS43, SBS45–60, SBS95**.
- Cancer-type signature subsets are defined in `config/` and reviewed by Prithviraj **before
  Phase 2 fitting runs**.
- *Where:* `config/config.yaml`, `fit.py` (Phase 2).

### D8. Citations
- Signatures: Alexandrov L.B. et al. (2020) The repertoire of mutational signatures in human
  cancer. *Nature* 578:94–101.
- COSMIC: Sondka Z. et al. (2024) *Nucleic Acids Res* 52(D1):D1210–D1217.
- SigProfilerAssignment: Díaz-Gay M. et al. (2023) *Bioinformatics* 39(12):btad756.
- *Where:* README.

## Earlier decisions (Phase 1 build)

### D0a. Trinucleotide context from the MAF, not a reference genome
- `profile.py` reads context from the GDC MAF `CONTEXT` column, so CI and the app need no
  multi-GB genome. It does not use SigProfilerMatrixGenerator.
- A MAF without `CONTEXT` raises a clear error.
- VCF input needs a GRCh38 context lookup; that is planned for the app phase.

### D0b. COSMIC signatures come from SigProfilerAssignment, not the COSMIC website
- The COSMIC website is never scraped or automated.
- The bundled v3.4 exome file is copied into gitignored `data/` and checked against a sha256
  pinned in `config/`.
- COSMIC files are never committed.
