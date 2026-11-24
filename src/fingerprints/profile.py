"""Build 96-channel single-base-substitution (SBS96) profiles from MAF files.

This is a pure pandas implementation. It does **not** need a reference genome:
the trinucleotide context is taken from the ``CONTEXT`` column that GDC adds to
its masked somatic MAFs.

GDC ``CONTEXT`` column
----------------------
The GDC MAF spec (column 112, https://docs.gdc.cancer.gov/Data/File_Formats/MAF_Format/)
defines ``CONTEXT`` as "The reference allele per VCF specs, and its five
flanking base pairs". It is plus-strand reference sequence (all GDC MAF rows
report ``Strand == "+"``). For a SNV in practice this is an 11-mer:
5 bases 5', the reference base, 5 bases 3'. The spec wording does not state the
length explicitly, so this module does not hard-code 11: it accepts any
odd-length context of at least 3 bases and takes the middle base as the
variant position. The middle base must equal ``Reference_Allele``; rows where
it does not are dropped and counted as ``context_mismatch``, never trusted.

Genome build
------------
Nothing here depends on the genome build: ``CONTEXT`` and ``Reference_Allele``
come from the same reference, so they are self-consistent whichever build the
MAF used, and ``NCBI_Build`` is not read. (The build matters later, when
choosing the matching COSMIC reference file, and for a future FASTA lookup.)

Output format and channel order
-------------------------------
Two orders of the same 96 labels (``5'[REF>ALT]3'``, e.g. ``A[C>A]A``) are
provided:

- :data:`SBS96_CHANNELS` is plain alphabetical order: 5' base outermost, then
  substitution, then 3' base (``A[C>A]A, A[C>A]C, A[C>A]G, A[C>A]T, A[C>G]A,
  ...``). This is the row order of the COSMIC reference files bundled with
  SigProfilerAssignment (e.g. ``COSMIC_v3.4_SBS_GRCh38.txt``) and of
  SigProfilerMatrixGenerator output. **The matrix always uses this order**, so
  it lines up row-for-row with the COSMIC signature matrix.
- :data:`SBS96_PLOT_ORDER` is the familiar spectrum-plot order: six blocks of
  16 by substitution (C>A, C>G, C>T, T>A, T>C, T>G), each block 5' base then
  3' base. Use it only for plotting (``matrix.loc[SBS96_PLOT_ORDER]``).

Adjacent SNVs (doublets)
------------------------
Two SNV rows in the same sample, on the same chromosome, at consecutive
positions (``pos`` and ``pos + 1``) look like one doublet base substitution
(DBS) that was reported as two SNVs. Longer runs (3+ consecutive positions) look
like multi-nucleotide variants (MNVs). The ``adjacent_snvs`` option controls
what happens to them (project decision D5 in ``docs/decisions.md``):

- ``"exclude"`` (default): every row in such a run is left out of the SBS96
  counts (drop reason ``adjacent_snv_dbs``), so the SBS and DBS catalogues stay
  mutually exclusive. This is a conservative project choice. Whether it matches the
  convention used to extract the COSMIC reference signatures (PCAWG) is **not
  yet verified**.
- ``"count_as_sbs"``: the rows are counted as ordinary SBS, as
  SigProfilerMatrixGenerator 1.6.0 does. Its ``catalogue_generator_single``
  counts same-sample SNVs at ``pos``/``pos+1`` as a DBS, but its SBS loop still
  counts every SNV, so they end up in both catalogues. (SigProfiler only
  writes runs of 3+ within 5 bp to an "MNS" seqInfo file; they get no special
  SBS treatment.) This matches SigProfiler's handling of adjacent SNVs only.
  Overall SBS96 counts can still differ from SigProfilerMatrixGenerator's,
  because this module also applies its own drops (e.g. ``tri_allelic``,
  ``non_plus_strand``), takes the context from ``CONTEXT``, not a genome
  FASTA, and de-duplicates differently.

In both modes the runs are detected and reported in the same way:
``adjacent_run_id`` / ``adjacent_run_len`` per row, and per-sample
``dbs_events`` (runs of exactly 2) and ``mnv_events`` (runs of 3+) in the drop
summary. So the event counts are identical whichever mode is used.

Only rows that passed every other check, *after* de-duplication, are
considered. So an exact duplicate (same position) can never look like a
neighbour, and a row dropped for another reason (e.g. ``context_mismatch``)
does not turn its valid neighbour into a doublet. Runs are found over distinct
positions, and run length counts positions, not rows. GDC already reports
caller-merged doublets as ``Variant_Type == "DNP"`` rows, which are dropped as
``not_snp`` and not counted as events. This rule only concerns pairs reported
as two separate SNP rows.

Multi-allelic positions
-----------------------
If one sample has two or more *different* alt alleles at the same chromosome
and position, all rows at that position are dropped as
``multiallelic_position`` (project decision: these are more likely artefacts
than real events). This is checked among rows that passed every other check,
after de-duplication, so an exact repeat of a single variant is still just a
``duplicate``. It is checked before adjacency, so these rows never join a
doublet run. Because dropped rows never make a neighbour adjacent, a valid SNV
next to a multi-allelic position is kept as SBS. The behaviour is the same in
both ``adjacent_snvs`` modes. The same position in *different* samples is not
multi-allelic.

Future hook (not built in v1)
-----------------------------
VCF input, or MAFs without ``CONTEXT``, will need the trinucleotide looked up
in a reference FASTA (e.g. with ``pyfaidx``). The intended place for that is
:func:`annotate_sbs96`: fill a ``CONTEXT``-like column from the FASTA before the
existing checks run, so every downstream rule (centre-base check, pyrimidine
folding, drop accounting) stays identical. Not implemented now because it needs
a multi-GB genome, which would break the "CI needs no internet/genome" rule.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SBS96_CHANNELS",
    "SBS96_PLOT_ORDER",
    "SUBSTITUTIONS",
    "DROP_REASONS",
    "EVENT_COLUMNS",
    "ADJACENT_SNV_MODES",
    "read_maf",
    "annotate_sbs96",
    "build_sbs96_matrix",
    "drop_summary",
]

# --------------------------------------------------------------------------
# Channel definitions
# --------------------------------------------------------------------------

#: The six pyrimidine-centred substitution classes.
SUBSTITUTIONS: list[str] = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
_BASES = ["A", "C", "G", "T"]

#: 96 SBS channel labels in the row order of the COSMIC reference files shipped
#: with SigProfilerAssignment: plain alphabetical order, i.e. 5' base outermost,
#: then substitution, then 3' base. First = ``A[C>A]A``, then ``A[C>A]C``,
#: ``A[C>A]G``, ``A[C>A]T``, ``A[C>G]A``, ...; last = ``T[T>G]T``.
#: Equal to ``sorted(SBS96_PLOT_ORDER)``. The SBS96 matrix uses this order.
SBS96_CHANNELS: list[str] = [
    f"{five}[{sub}]{three}" for five in _BASES for sub in SUBSTITUTIONS for three in _BASES
]

#: The same 96 labels in spectrum-plot order: six blocks of 16 by substitution
#: (C>A, C>G, C>T, T>A, T>C, T>G), each block 5' base then 3' base.
#: First = ``A[C>A]A``, ``A[C>A]C``, ..., ``T[C>A]T``, ``A[C>G]A``, ...
#: For plotting only; never use it to align with COSMIC files.
SBS96_PLOT_ORDER: list[str] = [
    f"{five}[{sub}]{three}" for sub in SUBSTITUTIONS for five in _BASES for three in _BASES
]

#: Reasons a MAF row can be excluded, in the order they are checked.
#: Each row is assigned the *first* reason that applies.
DROP_REASONS: list[str] = [
    "not_snp",  # Variant_Type is not SNP (DNP/TNP/ONP/INS/DEL) or alleles not 1 bp
    "non_acgt",  # ref or alt allele is not one of A/C/G/T
    "no_alt",  # neither tumour allele differs from the reference
    "tri_allelic",  # Allele1 != ref, Allele2 != ref, Allele1 != Allele2
    "non_plus_strand",  # a Strand column exists and this row is not "+"
    "missing_context",  # CONTEXT absent, too short, even length, or non-ACGT trinucleotide
    "context_mismatch",  # centre base of CONTEXT != Reference_Allele
    "missing_position",  # Start_Position missing/non-numeric (cannot dedupe safely)
    "duplicate",  # same sample / chrom / position / ref / alt seen earlier
    "multiallelic_position",  # >= 2 different alts at one sample / chrom / position
    "adjacent_snv_dbs",  # in a run of >= 2 adjacent same-sample SNVs (adjacent_snvs="exclude")
]

#: Allowed values of the ``adjacent_snvs`` option (see module docstring).
AdjacentSNVMode = Literal["exclude", "count_as_sbs"]
ADJACENT_SNV_MODES: tuple[str, ...] = get_args(AdjacentSNVMode)


def _check_adjacent_mode(mode: str) -> None:
    if mode not in ADJACENT_SNV_MODES:
        raise ValueError(f"adjacent_snvs must be one of {list(ADJACENT_SNV_MODES)}, got {mode!r}")


#: Extra per-sample event counts in :func:`drop_summary` (not row counts).
EVENT_COLUMNS: list[str] = ["dbs_events", "mnv_events"]

_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_REQUIRED_COLS = [
    "Tumor_Sample_Barcode",
    "Chromosome",
    "Start_Position",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
]


def _revcomp(seqs: pd.Series) -> pd.Series:
    """Reverse complement of each upper-case ACGT string in a Series."""
    return seqs.str.translate(_COMPLEMENT).str[::-1]


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _is_gzip(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def read_maf(path: str | Path) -> pd.DataFrame:
    """Read a (optionally gzipped) tab-separated MAF file.

    Leading lines starting with ``#`` (GDC writes e.g. ``#version gdc-1.0.0``
    and ``#annotation.spec ...``) are skipped. Only *leading* comment lines are
    skipped, so a ``#`` inside a data field is never treated as a comment.
    Gzip is detected from the file's magic bytes, not its extension.
    All columns are read as strings (``dtype=str``) to avoid mixed-type
    warnings on large files; ``Start_Position`` is converted to nullable int.
    """
    path = Path(path)
    gz = _is_gzip(path)
    opener = gzip.open if gz else open
    n_comment = 0
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            n_comment += 1
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=n_comment,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        compression="gzip" if gz else None,
        encoding="utf-8",
        low_memory=False,
    )
    if "Start_Position" in df.columns:
        df["Start_Position"] = pd.to_numeric(df["Start_Position"], errors="coerce").astype("Int64")
    logger.info("Read %d rows x %d columns from %s", len(df), df.shape[1], path)
    return df


# --------------------------------------------------------------------------
# Per-row annotation
# --------------------------------------------------------------------------


def _pick_alt(ref: pd.Series, a1: pd.Series, a2: pd.Series) -> pd.Series:
    """Choose the somatic alt allele for each row (vectorised).

    GDC MAFs put the variant allele in ``Tumor_Seq_Allele2``;
    ``Tumor_Seq_Allele1`` is usually equal to the reference (heterozygous call).
    So: use Allele2 if it differs from the reference, otherwise fall back to
    Allele1 if *that* differs, otherwise there is no alt (empty string,
    dropped as ``no_alt``). Allele1 == Allele2 != ref (homozygous / LOH call)
    gives that allele and is kept.

    Tri-allelic rows, where Allele1 != ref, Allele2 != ref **and**
    Allele1 != Allele2, still get Allele2 here. But
    :func:`annotate_sbs96` drops them as ``tri_allelic`` (project decision), so
    they never reach the matrix. Inputs must already be cleaned (upper-case,
    stripped, NaN -> "").
    """
    use2 = (a2 != "") & (a2 != ref)
    use1 = (a1 != "") & (a1 != ref)
    return pd.Series(np.where(use2, a2, np.where(use1, a1, "")), index=ref.index, dtype=object)


def _clean(series: pd.Series) -> pd.Series:
    return series.astype(object).where(series.notna(), "").astype(str).str.strip().str.upper()


def _keep_or_none(values: pd.Series, keep: pd.Series) -> pd.Series:
    """Object Series: ``values`` where ``keep`` is True, real ``None`` elsewhere."""
    out = pd.Series([None] * len(values), index=values.index, dtype=object)
    out[keep] = values[keep]
    return out


def _centre_trinucleotide(ctx: pd.Series) -> pd.Series:
    """Middle three bases of each odd-length (>= 3) context; "" otherwise.

    Vectorised per distinct context length (real MAFs have one or two lengths).
    """
    clen = ctx.str.len()
    trinuc = pd.Series("", index=ctx.index, dtype=object)
    for length in clen[(clen >= 3) & (clen % 2 == 1)].unique():
        mask = clen == length
        mid = int(length) // 2
        trinuc[mask] = ctx[mask].str.slice(mid - 1, mid + 2)
    return trinuc


def annotate_sbs96(
    maf: pd.DataFrame,
    sample_col: str = "Tumor_Sample_Barcode",
    context_col: str = "CONTEXT",
    adjacent_snvs: AdjacentSNVMode = "exclude",
) -> pd.DataFrame:
    """Annotate every MAF row with its SBS96 channel or the reason it was dropped.

    Returns a copy of ``maf`` (with a fresh RangeIndex) and six extra columns:

    - ``alt_allele``: chosen tumour allele (see :func:`_pick_alt`); ``None`` if none
    - ``trinuc_plus``: plus-strand trinucleotide (before pyrimidine folding)
    - ``sbs96_channel``: COSMIC label (e.g. ``A[C>T]G``) or ``None`` if dropped
    - ``drop_reason``: one of :data:`DROP_REASONS` or ``None`` if kept
    - ``adjacent_run_id`` / ``adjacent_run_len``: for rows in a run of adjacent
      SNVs, an ID shared by the rows of one run and the run length; ``<NA>``
      otherwise. Filled in both ``adjacent_snvs`` modes.

    ``adjacent_snvs``: ``"exclude"`` (default) drops rows in adjacent-SNV runs
    as ``adjacent_snv_dbs``; ``"count_as_sbs"`` counts them as ordinary SBS,
    as SigProfilerMatrixGenerator 1.6.0 does. Any other value raises ``ValueError``.
    See the module docstring.

    Pyrimidine convention: COSMIC channels are named by the pyrimidine (C or T)
    partner of the mutated base pair. If ``Reference_Allele`` is a purine (A/G),
    the ref, the alt **and** the trinucleotide are reverse-complemented
    (complement each base *and* reverse the order, so the 5' and 3' flanks swap).
    Example: plus-strand ``CGT`` with G>A becomes ``ACG`` with C>T -> ``A[C>T]G``.

    Raises ``ValueError`` if required columns (including ``CONTEXT``) are
    missing, or if any row has a missing/empty sample ID: such rows could not be
    attributed to a sample, so they are rejected loudly rather than dropped.
    Sample IDs and chromosome names are whitespace-stripped before use, and the
    stripped values are written back into the returned ``sample_col`` and
    ``Chromosome`` columns (``" S1"`` and ``"S1"`` are one sample).

    If a ``Strand`` column is present, rows whose value is not ``+`` (including
    empty) are dropped as ``non_plus_strand``, because ``CONTEXT`` is assumed
    to be plus-strand sequence.
    """
    _check_adjacent_mode(adjacent_snvs)
    required = [sample_col if c == "Tumor_Sample_Barcode" else c for c in _REQUIRED_COLS]
    missing = [c for c in required if c not in maf.columns]
    if missing:
        raise ValueError(f"MAF is missing required columns: {missing}")
    if context_col not in maf.columns:
        # Hook point for a future reference-FASTA lookup (see module docstring).
        raise ValueError(
            f"MAF is missing the required {context_col!r} column (plus-strand reference "
            "sequence centred on the variant, as in GDC masked somatic MAFs). MAFs without "
            f"{context_col!r} and VCF input need a reference-genome context lookup, which "
            "is not supported yet (planned for a later release)."
        )

    # Fresh RangeIndex so positional bookkeeping below is safe for any input index.
    out = maf.reset_index(drop=True)

    raw_sample = out[sample_col]
    sample = raw_sample.astype(object).where(raw_sample.notna(), "").astype(str).str.strip()
    no_sample = sample == ""
    if no_sample.any():
        raise ValueError(
            f"{int(no_sample.sum())} of {len(out)} MAF rows have a missing or empty "
            f"{sample_col!r} value; cannot assign them to a sample."
        )
    # Normalised once and used everywhere (dedupe, adjacency, matrix columns):
    # surrounding whitespace is stripped, so " S1" and "S1" are the same sample.
    out[sample_col] = sample
    chrom = out["Chromosome"].astype(object).where(out["Chromosome"].notna(), "")
    chrom = chrom.astype(str).str.strip()
    out["Chromosome"] = chrom

    ref = _clean(out["Reference_Allele"])
    a1 = _clean(out["Tumor_Seq_Allele1"])
    a2 = _clean(out["Tumor_Seq_Allele2"])
    vtype = _clean(out["Variant_Type"])
    ctx = _clean(out[context_col])
    alt = _pick_alt(ref, a1, a2)

    is_snp = (vtype == "SNP") & (ref.str.len() == 1) & (a1.str.len() <= 1) & (a2.str.len() <= 1)
    # Ref and both tumour alleles must be ACGT (alleles may be empty); "-"/"N" -> non_acgt.
    alleles_ok = ref.isin(_BASES) & (a1.isin(_BASES) | (a1 == "")) & (a2.isin(_BASES) | (a2 == ""))
    has_alt = alt != ""
    tri_allelic = (a1 != "") & (a2 != "") & (a1 != ref) & (a2 != ref) & (a1 != a2)
    if "Strand" in out.columns:
        plus_strand = _clean(out["Strand"]) == "+"
    else:
        plus_strand = pd.Series(True, index=out.index)
    trinuc = _centre_trinucleotide(ctx)
    trinuc_ok = trinuc.str.fullmatch("[ACGT]{3}").fillna(False).astype(bool)
    centre_ok = trinuc.str.slice(1, 2) == ref
    position_ok = pd.to_numeric(out["Start_Position"], errors="coerce").notna()

    reason = pd.Series([None] * len(out), index=out.index, dtype=object)

    def _assign(bad: pd.Series, name: str) -> None:
        reason[bad & reason.isna()] = name

    _assign(~is_snp, "not_snp")
    _assign(~alleles_ok, "non_acgt")
    _assign(~has_alt, "no_alt")
    _assign(tri_allelic, "tri_allelic")
    _assign(~plus_strand, "non_plus_strand")
    _assign(~trinuc_ok, "missing_context")
    _assign(~centre_ok, "context_mismatch")
    _assign(~position_ok, "missing_position")

    # Duplicates: only among rows that survived every other check, so a
    # duplicate is counted once and never double-counted with another reason.
    key = pd.DataFrame(
        {
            "s": sample,
            "c": chrom,
            "p": pd.to_numeric(out["Start_Position"], errors="coerce"),
            "r": ref,
            "a": alt,
        }
    )
    kept = reason.isna()
    dup = kept & key.where(kept).duplicated(keep="first")
    reason[dup] = "duplicate"
    kept = reason.isna()

    # Multi-allelic positions: >= 2 different alts among kept rows at one
    # sample/chrom/pos -> drop every row there (see module docstring).
    n_alts = key[kept].groupby(["s", "c", "p"])["a"].transform("nunique")
    multi = n_alts[n_alts > 1].index
    reason[multi] = "multiallelic_position"
    kept = reason.isna()

    # Adjacent SNVs -> doublets / MNVs (see module docstring). Only rows still
    # kept after every check and dedupe take part, so dropped rows and exact
    # duplicates never create adjacency.
    # Runs are detected over DISTINCT (sample, chrom, pos) among kept rows;
    # run length = number of distinct positions. (Multi-allelic positions were
    # dropped above, so in practice each position has one kept row.)
    run_id = pd.Series(pd.NA, index=out.index, dtype="Int64")
    run_len = pd.Series(pd.NA, index=out.index, dtype="Int64")
    k = key.loc[kept, ["s", "c", "p"]]
    if len(k):
        loci = k.drop_duplicates().sort_values(["s", "c", "p"], kind="stable")
        same_run = (
            (loci["s"] == loci["s"].shift())
            & (loci["c"] == loci["c"].shift())
            & (loci["p"] - loci["p"].shift() == 1)
        )
        loci = loci.assign(rid=(~same_run).cumsum().to_numpy())
        loci["rlen"] = loci.groupby("rid")["p"].transform("size")
        loci = loci[loci["rlen"] >= 2]
        merged = k.reset_index().merge(loci, on=["s", "c", "p"], how="inner")
        idx = merged["index"].to_numpy()
        run_id[idx] = merged["rid"].to_numpy()
        run_len[idx] = merged["rlen"].to_numpy()
        if adjacent_snvs == "exclude":
            reason[idx] = "adjacent_snv_dbs"
    kept = reason.isna()

    # Pyrimidine folding, vectorised.
    purine = ref.isin(["A", "G"])
    r = ref.where(~purine, _revcomp(ref))
    a = alt.where(~purine, _revcomp(alt))
    t = trinuc.where(~purine, _revcomp(trinuc))
    label = t.str.slice(0, 1) + "[" + r + ">" + a + "]" + t.str.slice(2, 3)

    out["alt_allele"] = _keep_or_none(alt, has_alt)
    out["trinuc_plus"] = _keep_or_none(trinuc, trinuc != "")
    out["sbs96_channel"] = _keep_or_none(label, kept)
    out["drop_reason"] = reason
    out["adjacent_run_id"] = run_id
    out["adjacent_run_len"] = run_len
    return out


# --------------------------------------------------------------------------
# Matrix building
# --------------------------------------------------------------------------


def _samples_in_order(maf: pd.DataFrame, sample_col: str) -> list[str]:
    return list(pd.unique(maf[sample_col].astype(str)))


def drop_summary(annotated: pd.DataFrame, sample_col: str = "Tumor_Sample_Barcode") -> pd.DataFrame:
    """Per-sample counts of kept SNVs and of each drop reason, plus event counts.

    Index = samples (MAF order). Columns = ``kept`` + :data:`DROP_REASONS` +
    ``total`` (row counts; ``kept`` + reasons == ``total`` == input rows for the
    sample), then :data:`EVENT_COLUMNS`. These are *event* counts, not rows:
    ``dbs_events`` = adjacent-SNV runs of exactly 2 rows, ``mnv_events`` = runs
    of 3+ rows. They are not part of ``total``.
    """
    samples = _samples_in_order(annotated, sample_col)
    status = annotated["drop_reason"].fillna("kept")
    tab = pd.crosstab(annotated[sample_col].astype(str), status)
    cols = ["kept", *DROP_REASONS]
    tab = tab.reindex(index=samples, columns=cols, fill_value=0).astype(int)
    tab["total"] = tab.sum(axis=1)

    runs = annotated.loc[
        annotated["adjacent_run_id"].notna(), [sample_col, "adjacent_run_id", "adjacent_run_len"]
    ]
    runs = runs.drop_duplicates(subset="adjacent_run_id")
    runs_sample = runs[sample_col].astype(str)
    is_dbs = runs["adjacent_run_len"] == 2
    tab["dbs_events"] = (
        runs_sample[is_dbs].value_counts().reindex(samples, fill_value=0).astype(int)
    )
    tab["mnv_events"] = (
        runs_sample[~is_dbs].value_counts().reindex(samples, fill_value=0).astype(int)
    )
    tab.index.name = sample_col
    tab.columns.name = None
    return tab


def build_sbs96_matrix(
    maf: str | Path | pd.DataFrame,
    sample_col: str = "Tumor_Sample_Barcode",
    return_dropped: bool = False,
    context_col: str = "CONTEXT",
    adjacent_snvs: AdjacentSNVMode = "exclude",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build a 96 x n_samples SBS count matrix from a MAF.

    Parameters
    ----------
    maf
        Path to a MAF (plain or gzipped) or an already-loaded DataFrame.
    sample_col
        Column identifying the tumour sample. Default ``Tumor_Sample_Barcode``
        (aliquot-level barcode in TCGA; no patient-level merging is done here).
    return_dropped
        If True, also return the per-sample drop summary from :func:`drop_summary`.
    context_col
        Column holding plus-strand reference context centred on the variant.
    adjacent_snvs
        ``"exclude"`` (default, conservative: SBS and DBS kept mutually
        exclusive) or ``"count_as_sbs"`` (counts them as ordinary
        SBS, as SigProfilerMatrixGenerator 1.6.0 does). Event counts are
        reported either way. See the module docstring.

    Returns
    -------
    matrix
        Index = :data:`SBS96_CHANNELS` (alphabetical / COSMIC-file order, not
        plot order; see :data:`SBS96_PLOT_ORDER`), columns = samples in order of
        first appearance in the MAF, integer counts. Every sample in the MAF
        appears, even one with zero usable SNVs (an all-zero column).
    dropped
        Only when ``return_dropped=True``.

    The drop summary is always logged at INFO level (totals per reason), so no
    row is ever dropped silently.
    """
    _check_adjacent_mode(adjacent_snvs)  # fail before reading a large file
    df = read_maf(maf) if isinstance(maf, (str, Path)) else maf
    annotated = annotate_sbs96(
        df, sample_col=sample_col, context_col=context_col, adjacent_snvs=adjacent_snvs
    )
    samples = _samples_in_order(annotated, sample_col)

    kept = annotated[annotated["drop_reason"].isna()]
    counts = pd.crosstab(kept["sbs96_channel"], kept[sample_col].astype(str))
    matrix = counts.reindex(index=SBS96_CHANNELS, columns=samples, fill_value=0).astype("int64")
    matrix.index.name = "MutationType"  # SigProfiler's name for the channel column
    matrix.columns.name = None

    summary = drop_summary(annotated, sample_col=sample_col)
    totals = summary.sum(axis=0)
    if int(matrix.to_numpy().sum()) != len(kept) or int(totals["kept"]) != len(kept):
        raise RuntimeError(  # internal consistency check; should never fire
            f"SBS96 matrix total {int(matrix.to_numpy().sum())} != kept rows {len(kept)}"
        )
    logger.info(
        "SBS96: %d samples, %d input rows, %d SNVs kept; dropped: %s",
        len(samples),
        int(totals["total"]),
        int(totals["kept"]),
        ", ".join(f"{r}={int(totals[r])}" for r in DROP_REASONS),
    )
    logger.info(
        "SBS96: adjacent-SNV events (%s): %d DBS (2 rows), %d MNV (3+ rows)",
        "excluded" if adjacent_snvs == "exclude" else "counted as SBS",
        int(totals["dbs_events"]),
        int(totals["mnv_events"]),
    )
    if return_dropped:
        return matrix, summary
    return matrix
