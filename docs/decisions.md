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
- Use GDC `msi_status` (MSI/MSS) and `msi_score` on tumour aligned-reads (BAM) files.
- **Source (checked 2026-11-22):**
  - The GDC computes these with **MSIsensor2**, from the tumour BAM alone; the original
    MSIsensor uses a matched normal.
  - The score is the fraction of valid microsatellite sites that are unstable.
  - A BAM is called MSI using a **20% score cutoff**.
  - Quotes: "The GDC adopts MSIsensor2 to derive Microsatellite Instability (MSI) information
    from tumor DNA-Seq data" and "determined using a MSI score cutoff value of 20%". From the GDC
    DNA-Seq pipeline docs:
    <https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/DNA_Seq_Variant_Calling_Pipeline/#microsatellite-instability-detection>.
  - The fields have been queryable for all projects since GDC Data Release 38.0 (Aug 2023).
    That release says they "were produced using MSISensor2"
    (<https://docs.gdc.cancer.gov/Data/Release_Notes/Data_Release_Notes/>).
  - The GDC data dictionary describes the fields only as "MSIsensor"
    (`aligned_reads.yaml`,
    <https://github.com/NCI-GDC/gdcdictionary/blob/develop/src/gdcdictionary/schemas/aligned_reads.yaml>;
    also `https://api.gdc.cancer.gov/files/_mapping`).
  - No MSIsensor2 version is given, and the docs don't say whether exactly 0.20 counts as MSI.
    In our data, MSS scores reach at most 0.1995 and MSI scores start at 0.209.
- These are **bioinformatics-derived labels, not clinical MSI or MMR tests**. The GDC says so,
  and the README must state this limitation for the colorectal MMR check.
- One case can have several BAMs. In TCGA-COAD/READ there are 1,294 labelled BAMs: 636 WXS and
  658 WGS (live API check, Data Release 46.0). The GDC warns that their statuses can disagree.
- When a case's WXS and WGS calls disagree, prefer WXS, because the MAFs are WXS. Flag
  disagreeing cases.
- **Borderline scores (decided after PR #1):** flag cases with an MSI score of 0.15–0.25 as
  borderline. Rerun the MMR comparison with them excluded, as a sensitivity check.
- **Sanity check:**
  - MSI-H should be roughly 10–20% of COAD, and those tumours should be hypermutated.
  - Count **cases, not BAM files**.
  - For the expected hypermutated/MSI fraction, cite: The Cancer Genome Atlas Network (2012)
    Comprehensive molecular characterization of human colon and rectal cancer. *Nature*
    487:330–337.
- *Where:* validation, Milestone 4.

### D5. Adjacent SNVs: SBS and DBS catalogues kept mutually exclusive (default)
- Same-sample SNVs on the same chromosome at consecutive positions are treated as doublet (or
  longer) substitutions.
- **By default they are excluded from the SBS96 counts**, so the SBS and DBS catalogues are
  mutually exclusive. This is a conservative choice.
  - The number of such events is always reported: `dbs_events` for runs of 2, `mnv_events` for
    runs of 3 or more.
- Revised 2026-11-22. The first version of this decision said exclusion matched
  SigProfilerMatrixGenerator. It does not: **SigProfilerMatrixGenerator 1.6.0 counts these SNVs
  in both** the SBS and the DBS matrices.
  - **Literature check (2026-11-22).** Whether COSMIC v3 / PCAWG removed DBS-forming adjacent
    SNVs from the SBS catalogues is **not stated** in any of these:
    - the Methods or Supplementary Note 2 of Alexandrov et al. 2020
      (doi:10.1038/s41586-020-1943-3);
    - Bergstrom et al. 2019 on SigProfilerMatrixGenerator (doi:10.1186/s12864-019-6041-2);
    - the PCAWG Consortium 2020 paper and its Supplementary Information
      (doi:10.1038/s41586-020-1969-6).
  - Excluding them by default is therefore **our own conservative choice, not a cited
    convention**.
  - The totals reported in Alexandrov 2020 are consistent with mutually exclusive counting but
    do not establish it: 79,793,266 SBS + 814,191 DBS + 4,122,233 indels = 84,729,690
    mutations.
  - A definitive answer probably lies in the PCAWG consensus-calling and MNV-merging code or
    the PCAWG Synapse data (login required).
- Configurable via `profile.adjacent_snvs` in `config/config.yaml`, which the CLI reads as the
  default; `--adjacent-snvs` overrides it:
  - `exclude` (default);
  - `count_as_sbs` counts them as ordinary SBS, as SigProfilerMatrixGenerator 1.6.0 does.
  - SBS counts still differ from SigProfilerMatrixGenerator for other reasons: tri-allelic and
    strand drops, context taken from the MAF `CONTEXT` column, and different de-duplication.
- Multi-allelic positions are dropped before adjacency is assessed (see D6b). They therefore
  never join a run, and an SNV next to one is kept as SBS.
- Only pairs reported as two separate SNP rows are affected. GDC rows already merged by the
  caller (DNP/TNP/ONP) are dropped as `not_snp` in both modes. A neighbour dropped for another
  reason does not make a valid SNV adjacent.
- **Phase 2 sensitivity check:** rerun the melanoma SBS7 result with `count_as_sbs`. This is
  where the choice matters most:
  - melanoma: DBS1 CC>TT would otherwise add to the SBS7a/b C>T channels;
  - smoking-related lung cancer: DBS2 CC>AA would otherwise add to the SBS4 C>A channels.
- DBS signatures themselves stay out of scope for v1.
- *Where:* `src/fingerprints/profile.py`, config, CLI `--adjacent-snvs`; implemented in PR #1.

### D6. Tri-allelic sites are dropped
- Rows where both tumour alleles differ from the reference and from each other are dropped and
  counted under a named drop reason.
- *Where:* `src/fingerprints/profile.py`, implemented in PR #1.

### D6b. Multi-allelic positions are dropped (decided 2026-11-22)
- When one sample has two or more different alt alleles at the same position, all rows at that
  position are dropped and counted under `multiallelic_position`. They are more likely
  artefacts than real events.
- This is applied after the other row checks and after de-duplication, and before adjacency
  detection.
- Only rows that passed every other check count towards making a position multi-allelic. An alt
  that fails a check (e.g. `context_mismatch`) does not make the position multi-allelic.
  Confirmed after PR #1.
- A real variant that the caller reported twice with different alts is therefore lost. This is
  accepted, because it is counted in the drop summary.
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

#### D7 sign-off: cohort signature subsets (approved 2026-11-22)
Subsets are in `config/config.yaml` (`fitting.signature_subsets`), proposed from Alexandrov et
al. 2020 (ED Fig. 5, Fig. 3, ED Fig. 6a).
1. **Core/minor cut-off:** about 20% of tumours in ED Fig. 5. Approved.
2. **SBS18** is kept in lung **and added to colorectal**. Run a with/without-SBS18
   sensitivity check for the smoker vs never-smoker result.
3. **Newer colorectal signatures** (SBS88 colibactin, SBS36, SBS10c/d) are **not** in the v1
   fit. SBS88 (Pleguezuelos-Manzano et al. 2020) is listed as a known limitation in the README.
4. **Colorectal:** SBS21 is dropped (near-duplicate of SBS26). There is **no
   minimum-exposure gate and no separate MSS list**, because both would use MSI status to
   shape the fit that is then tested against MSI status (circular).
   - MMR validation: test the **summed MMR exposure (SBS6+14+15+20+26+44)** in MSI vs MSS.
   - Report the SBS1/SBS6 overlap as a limitation.
5. **Plain-English summaries** group signatures, while tables keep the individual ones:
   - SBS1 + SBS5 + SBS40a: "background / ageing-related processes (partly unknown cause)";
   - SBS7a–d + SBS38: "UV light (SBS38 proposed)".
   - Phase 3 (2026-11-24): the report and app warn when the fit cosine is below **0.90** (poor fit,
     check the cancer type). This is separate from the 0.8 validation sensitivity cut above.
6. **APOBEC (SBS2/SBS13)** is not added to melanoma or colorectal: they fall below the 20%
   rule, which is applied consistently.
7. **Prevalences** are figure estimates. Label them approximate and cite the figure (ED Fig. 5).
- Final subsets: LUAD and LUSC: SBS1, 2, 4, 5, 13, 18. SKCM: SBS1, 5, 7a–d, 38. COAD and READ
  (shared): SBS1, 5, 40a, 44, 6, 15, 26, 14, 20, 10a, 10b, 28, 18.
- **Fitting method (2026-11-22):** direct unpenalised NNLS (`scipy.optimize.nnls`) on exactly
  the approved subset, not SigProfilerAssignment's `cosmic_fit`.
  - `cosmic_fit` applies its own forward/backward signature selection, which can add or drop
    SBS4, SBS7 or MMR signatures and would confound the validations (it zeroed SBS18 in all 25
    tested LUAD tumours).
  - A cross-check on 30 LUAD tumours: per-tumour exposure correlation SBS4 0.98, SBS13 0.94,
    SBS2 0.86, SBS5 0.64, SBS1 0.60.
  - Overlapping weak signatures (SBS1/SBS5/SBS40a, SBS18, SBS6) can trade exposure, so
    validation uses grouped or summed exposures.
- **Validation specification (fixed before any result was seen):**
  - Report the per-tumour reconstruction cosine alongside every result. As a sensitivity
    analysis, rerun with tumours below cosine 0.8 excluded. This is a quality filter and never
    uses smoking or MSI status.
  - Melanoma: the primary measure is the summed fraction of SBS7a–d. Secondary: SBS7a+b only,
    and SBS7a–d + SBS38.
  - Lung: SBS4 fraction.
  - Colorectal: summed MMR fraction (SBS6+14+15+20+26+44).
  - Exposures are NNLS-attributed counts and can differ from observed SNV counts when the fit is
    poor.
- Phase 2 sensitivity checks (all reported): SBS18 with/without (lung smoker vs never-smoker);
  `min_snvs` 20 and 100; `adjacent_snvs: count_as_sbs` (melanoma SBS7, D5); MSI scores
  0.15–0.25 excluded (D4).

### D8. Citations
- Signatures: Alexandrov L.B. et al. (2020) The repertoire of mutational signatures in human
  cancer. *Nature* 578:94–101.
- COSMIC: Sondka Z. et al. (2024) *Nucleic Acids Res* 52(D1):D1210–D1217.
- SigProfilerAssignment: Díaz-Gay M. et al. (2023) *Bioinformatics* 39(12):btad756.
- Never-smoker lung cancer context (added 2026-11-23): Zhang T. et al. (2021) Genomic and
  evolutionary classification of lung cancer in never smokers. *Nat Genet* 53(9):1348–1359,
  doi:10.1038/s41588-021-00920-0, PMID 34493867 (verified on PubMed by Prithviraj). Its abstract
  says no strong tobacco smoking signatures were detected in never-smoker lung cancers, even with
  second-hand smoke exposure.
- UV signatures SBS7c/7d (added Phase 4): Saini N. et al. (2016) *PLoS Genetics*,
  doi:10.1371/journal.pgen.1006385, as listed on the COSMIC SBS7c/7d pages. **Full reference
  (title, volume, pages) not yet verified: Prithviraj to check on PubMed.**
- Expected colon MSI/hypermutated fraction (see D4): The Cancer Genome Atlas Network (2012)
  *Nature* 487:330–337.
- *Where:* README.

## Earlier decisions (Phase 1 build)

### D0a. Trinucleotide context from the MAF, not a reference genome
- `profile.py` reads context from the GDC MAF `CONTEXT` column, so CI and the app need no
  multi-GB genome. It does not use SigProfilerMatrixGenerator.
- A MAF without `CONTEXT` raises a clear error.
- VCF input needs a GRCh38 context lookup; not yet implemented (the app currently accepts MAF only).

### D0b. COSMIC signatures come from SigProfilerAssignment, not the COSMIC website
- The COSMIC website is never scraped or automated.
- The bundled v3.4 exome file is copied into gitignored `data/` and checked against a sha256
  pinned in `config/`.
- COSMIC files are never committed.

## 2026-11-23: Phase 2 review answers

### D9. Phase 2 review answers (Prithviraj)
Decided after the validation results were first seen. They add reporting and extra tests; no
pre-specified primary analysis or number changes.
- **(a) Primary threshold.** Keep the pre-specified >= 50 SNV filter as the headline. The 20 and
  100 SNV results are shown in the same table as sensitivity analyses. The LUAD result holds
  across thresholds (AUC 0.72-0.88). Never switch the headline to the better-looking number after
  seeing results.
- **(b) Never-smokers with SBS4.** README wording: a minority of reported never-smokers show
  tobacco-like SBS4 exposure; possible reasons include misreported smoking history, second-hand
  smoke, or overlap between SBS4 and similar signatures; the analysis cannot tell these apart.
  Zhang et al. 2021 (D8) is cited as context only.
- **(c) Pack-years.** No pack-years dose-response is expected from the fraction: an exposure
  fraction is compositional and pack-years are self-reported and noisy. An extra secondary test is
  added: Spearman of the SBS4 mutation COUNT (NNLS-attributed exposure) vs pack-years in
  ever-smokers (LUAD, LUSC), same filters and CI method, shown beside the fraction version. The count scales
  with total mutational burden, so it may partly track burden; it is secondary, has no multiplicity
  correction, and was added after the fraction result was seen (at Prithviraj's request).
- **(d) Low summed MMR in MSS.** Likely SBS1/SBS6 overlap plus the noise floor of NNLS refitting,
  not real mismatch-repair deficiency.
- **(e) READ MSI.** A READ MSI share of 5.8% is expected (MSI is rarer in rectal than colon
  cancer). Report colon and rectum prevalence separately (the 10-20% check applies to colon
  only). Keep MSI tumours with low summed MMR in the analysis and list them in a supplementary
  table (listing cut 0.25, fixed in advance; a listing aid, not a test).
- **(f) Non-UV melanomas.** Flag, do not exclude: summed SBS7a-d fraction <= 0.5. Show primary
  site / histology fields where the clinical data has them.
- **(g) Confirmed.** The "consistent with" wording rule (AUC 95% CI excludes 0.5 in the expected
  direction), the 0.05 materiality threshold for the D5 check (on the median paired difference),
  and the van Elteren stratified test for the pooled lung model.
- *Where:* `src/fingerprints/validate.py`, `scripts/run_validation.py`,
  `results/validation_summary.md`, `results/tables/validation_*.tsv`.
