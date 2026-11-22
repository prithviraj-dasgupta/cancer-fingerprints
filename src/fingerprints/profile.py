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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SBS96_CHANNELS",
    "SBS96_PLOT_ORDER",
    "SUBSTITUTIONS",
    "DROP_REASONS",
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
    "non_plus_strand",  # a Strand column exists and this row is not "+"
    "missing_context",  # CONTEXT absent, too short, even length, or non-ACGT trinucleotide
    "context_mismatch",  # centre base of CONTEXT != Reference_Allele
    "missing_position",  # Start_Position missing/non-numeric (cannot dedupe safely)
    "duplicate",  # same sample / chrom / position / ref / alt seen earlier
]

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
    Allele1 if *that* differs, otherwise there is no alt (empty string).
    If both differ from the reference (and from each other) Allele2 is used,
    consistent with GDC's convention that Allele2 is the reported variant.
    Inputs must already be cleaned (upper-case, stripped, NaN -> "").
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
) -> pd.DataFrame:
    """Annotate every MAF row with its SBS96 channel or the reason it was dropped.

    Returns a copy of ``maf`` (with a fresh RangeIndex) and four extra columns:

    - ``alt_allele``: chosen tumour allele (see :func:`_pick_alt`); ``None`` if none
    - ``trinuc_plus``: plus-strand trinucleotide (before pyrimidine folding)
    - ``sbs96_channel``: COSMIC label (e.g. ``A[C>T]G``) or ``None`` if dropped
    - ``drop_reason``: one of :data:`DROP_REASONS` or ``None`` if kept

    Pyrimidine convention: COSMIC channels are named by the pyrimidine (C or T)
    partner of the mutated base pair. If ``Reference_Allele`` is a purine (A/G),
    the ref, the alt **and** the trinucleotide are reverse-complemented
    (complement each base *and* reverse the order, so the 5' and 3' flanks swap).
    Example: plus-strand ``CGT`` with G>A becomes ``ACG`` with C>T -> ``A[C>T]G``.

    Raises ``ValueError`` if required columns (including ``CONTEXT``) are
    missing, or if any row has a missing/empty sample ID: such rows could not be
    attributed to a sample, so they are rejected loudly rather than dropped.
    If a ``Strand`` column is present, rows whose value is not ``+`` (including
    empty) are dropped as ``non_plus_strand``, because ``CONTEXT`` is assumed
    to be plus-strand sequence.
    """
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
            "is not supported yet (planned for the app phase)."
        )

    # Fresh RangeIndex so positional bookkeeping below is safe for any input index.
    out = maf.reset_index(drop=True)

    sample = out[sample_col]
    no_sample = sample.isna() | (sample.astype(str).str.strip() == "")
    if no_sample.any():
        raise ValueError(
            f"{int(no_sample.sum())} of {len(out)} MAF rows have a missing or empty "
            f"{sample_col!r} value; cannot assign them to a sample."
        )

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
    _assign(~plus_strand, "non_plus_strand")
    _assign(~trinuc_ok, "missing_context")
    _assign(~centre_ok, "context_mismatch")
    _assign(~position_ok, "missing_position")

    # Duplicates: only among rows that survived every other check, so a
    # duplicate is counted once and never double-counted with another reason.
    key = pd.DataFrame(
        {
            "s": sample.astype(str),
            "c": out["Chromosome"].astype(str),
            "p": pd.to_numeric(out["Start_Position"], errors="coerce"),
            "r": ref,
            "a": alt,
        }
    )
    kept = reason.isna()
    dup = kept & key.where(kept).duplicated(keep="first")
    reason[dup] = "duplicate"
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
    return out


# --------------------------------------------------------------------------
# Matrix building
# --------------------------------------------------------------------------


def _samples_in_order(maf: pd.DataFrame, sample_col: str) -> list[str]:
    return list(pd.unique(maf[sample_col].astype(str)))


def drop_summary(annotated: pd.DataFrame, sample_col: str = "Tumor_Sample_Barcode") -> pd.DataFrame:
    """Per-sample counts of kept SNVs and of each drop reason.

    Index = samples (MAF order), columns = ``kept`` + :data:`DROP_REASONS`
    + ``total``. Rows sum to the number of input rows for that sample.
    """
    samples = _samples_in_order(annotated, sample_col)
    status = annotated["drop_reason"].fillna("kept")
    tab = pd.crosstab(annotated[sample_col].astype(str), status)
    cols = ["kept", *DROP_REASONS]
    tab = tab.reindex(index=samples, columns=cols, fill_value=0).astype(int)
    tab["total"] = tab.sum(axis=1)
    tab.index.name = sample_col
    tab.columns.name = None
    return tab


def build_sbs96_matrix(
    maf: str | Path | pd.DataFrame,
    sample_col: str = "Tumor_Sample_Barcode",
    return_dropped: bool = False,
    context_col: str = "CONTEXT",
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
    df = read_maf(maf) if isinstance(maf, (str, Path)) else maf
    annotated = annotate_sbs96(df, sample_col=sample_col, context_col=context_col)
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
    if return_dropped:
        return matrix, summary
    return matrix
