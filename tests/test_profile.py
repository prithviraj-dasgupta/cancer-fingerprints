"""Exact-count tests for the MAF -> SBS96 profile builder.

The fixture ``tests/fixtures/profile_small.maf`` is hand-built: every row has a
comment in the generator recording its expected channel or drop reason. The
expectations are repeated below so that the test file is self-contained.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd
import pytest

from fingerprints.profile import (
    DROP_REASONS,
    SBS96_CHANNELS,
    SBS96_PLOT_ORDER,
    SUBSTITUTIONS,
    annotate_sbs96,
    build_sbs96_matrix,
    read_maf,
)

FIXTURE = Path(__file__).parent / "fixtures" / "profile_small.maf"
S1 = "TCGA-FP-0001-01A-11D-A000-08"
S2 = "TCGA-FP-0002-01A-11D-A000-08"
S3 = "TCGA-FP-0003-01A-11D-A000-08"

# Non-zero cells of the expected matrix, per sample.
EXPECTED_CELLS = {
    S1: {
        "A[C>T]G": 2,  # C>T in ACG + G>A in CGT (purine, reverse-complemented)
        "G[C>A]T": 1,  # G>T in AGC -> GCT
        "G[T>C]A": 1,  # A>G in TAC -> GTA
        "C[T>A]G": 1,  # T>A in CTG
        "T[C>A]A": 1,  # alt carried in Tumor_Seq_Allele1
    },
    S2: {
        "T[C>A]T": 3,  # two C>A in TCT + G>T in AGA -> TCT
        "A[T>C]A": 1,
        "T[C>G]A": 1,  # G>C in TGA -> TCA
        "A[C>T]G": 1,  # same locus as an S1 mutation, different sample: kept
    },
    S3: {},  # indels only
}

EXPECTED_DROPS = {
    S1: {"kept": 6, "duplicate": 1, "not_snp": 1, "context_mismatch": 1, "total": 9},
    S2: {
        "kept": 6,
        "not_snp": 1,
        "missing_context": 1,
        "non_acgt": 1,
        "no_alt": 1,
        "total": 10,
    },
    S3: {"kept": 0, "not_snp": 2, "total": 2},
}


def _minimal_maf(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """rows: (ref, allele1, allele2, context) for a single sample, SNPs only."""
    return pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["X"] * len(rows),
            "Chromosome": ["chr1"] * len(rows),
            "Start_Position": list(range(1, len(rows) + 1)),
            "Variant_Type": ["SNP"] * len(rows),
            "Reference_Allele": [r[0] for r in rows],
            "Tumor_Seq_Allele1": [r[1] for r in rows],
            "Tumor_Seq_Allele2": [r[2] for r in rows],
            "CONTEXT": [r[3] for r in rows],
        }
    )


# ---------------------------------------------------------------- channels


def test_channel_list_is_96_unique():
    assert len(SBS96_CHANNELS) == 96
    assert len(set(SBS96_CHANNELS)) == 96
    assert len(SBS96_PLOT_ORDER) == 96
    assert set(SBS96_PLOT_ORDER) == set(SBS96_CHANNELS)


def test_channel_order_is_alphabetical_cosmic_file_order():
    """SBS96_CHANNELS = row order of COSMIC files bundled with SigProfilerAssignment.

    Pinned explicitly (the real COSMIC file is not available in CI): 5' base
    outermost, then substitution, then 3' base, i.e. plain alphabetical order.
    """
    assert SBS96_CHANNELS[:8] == [
        "A[C>A]A",
        "A[C>A]C",
        "A[C>A]G",
        "A[C>A]T",
        "A[C>G]A",
        "A[C>G]C",
        "A[C>G]G",
        "A[C>G]T",
    ]
    assert SBS96_CHANNELS[-4:] == ["T[T>G]A", "T[T>G]C", "T[T>G]G", "T[T>G]T"]
    assert sorted(SBS96_CHANNELS) == SBS96_CHANNELS
    # 4 blocks of 24 by 5' base; within each, 6 substitutions x 4 3' bases
    for b, five in enumerate("ACGT"):
        block = SBS96_CHANNELS[24 * b : 24 * (b + 1)]
        assert all(ch[0] == five for ch in block)
        assert [ch[2:5] for ch in block] == [s for s in SUBSTITUTIONS for _ in range(4)]
        assert [ch[-1] for ch in block] == list("ACGT") * 6


def test_plot_order_is_substitution_blocks():
    assert SBS96_PLOT_ORDER[:5] == ["A[C>A]A", "A[C>A]C", "A[C>A]G", "A[C>A]T", "C[C>A]A"]
    assert SBS96_PLOT_ORDER[15] == "T[C>A]T"
    assert SBS96_PLOT_ORDER[16] == "A[C>G]A"
    assert SBS96_PLOT_ORDER[-3:] == ["T[T>G]C", "T[T>G]G", "T[T>G]T"]
    for b, sub in enumerate(["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]):
        block = SBS96_PLOT_ORDER[16 * b : 16 * (b + 1)]
        assert all(ch[2:5] == sub for ch in block)
        assert [ch[0] for ch in block] == [x for x in "ACGT" for _ in range(4)]
        assert [ch[-1] for ch in block] == list("ACGT") * 4


# ------------------------------------------------------- single mutations


@pytest.mark.parametrize(
    ("ref", "alt", "context", "expected"),
    [
        ("C", "T", "TTTTACGTTTT", "A[C>T]G"),  # pyrimidine ref, no folding
        ("G", "A", "TTTTCGTTTTT", "A[C>T]G"),  # CGT -> ACG
        # Asymmetric flanks: complement-without-reverse would give T[C>A]G / A[T>C]G.
        ("G", "T", "CCCCAGCCCCC", "G[C>A]T"),  # AGC -> GCT
        ("A", "G", "GGGGTACGGGG", "G[T>C]A"),  # TAC -> GTA
        ("G", "C", "AAAATGAAAAA", "T[C>G]A"),  # TGA -> TCA
        ("A", "C", "CCCCGATCCCC", "A[T>G]C"),  # GAT -> ATC
        ("C", "A", "ACG", "A[C>A]G"),  # 3-mer context also accepted
    ],
)
def test_single_mutation_channel(ref, alt, context, expected):
    ann = annotate_sbs96(_minimal_maf([(ref, ref, alt, context)]))
    assert ann.loc[0, "drop_reason"] is None
    assert ann.loc[0, "sbs96_channel"] == expected


def test_alt_falls_back_to_allele1():
    ann = annotate_sbs96(_minimal_maf([("C", "A", "C", "GGGGTCAGGGG")]))
    assert ann.loc[0, "alt_allele"] == "A"
    assert ann.loc[0, "sbs96_channel"] == "T[C>A]A"


def test_allele2_preferred_when_both_differ():
    ann = annotate_sbs96(_minimal_maf([("C", "A", "T", "GGGGACGGGGG")]))
    assert ann.loc[0, "alt_allele"] == "T"
    assert ann.loc[0, "sbs96_channel"] == "A[C>T]G"


def test_lowercase_input_is_normalised():
    ann = annotate_sbs96(_minimal_maf([("c", "c", "t", "ttttacgtttt")]))
    assert ann.loc[0, "sbs96_channel"] == "A[C>T]G"


@pytest.mark.parametrize(
    ("ref", "a1", "a2", "context", "reason"),
    [
        ("C", "C", "T", "TTTTAAGTTTT", "context_mismatch"),
        ("C", "C", "T", "", "missing_context"),
        ("C", "C", "T", "AC", "missing_context"),  # too short
        ("C", "C", "T", "AAGCT", "context_mismatch"),  # 5-mer, centre G != C
        ("C", "C", "T", "ACGT", "missing_context"),  # even length: no centre
        ("C", "C", "T", "TTTTNCGTTTT", "missing_context"),  # N in trinucleotide
        ("C", "C", "N", "TTTTACGTTTT", "non_acgt"),
        ("C", "C", "C", "TTTTACGTTTT", "no_alt"),
    ],
)
def test_single_mutation_drop(ref, a1, a2, context, reason):
    ann = annotate_sbs96(_minimal_maf([(ref, a1, a2, context)]))
    assert ann.loc[0, "drop_reason"] == reason
    assert ann.loc[0, "sbs96_channel"] is None


def test_missing_context_column_raises(tmp_path):
    """No CONTEXT column at all is an error, not 'every row dropped'."""
    maf = _minimal_maf([("C", "C", "T", "ACG")]).drop(columns="CONTEXT")
    with pytest.raises(ValueError, match="missing the required 'CONTEXT' column"):
        annotate_sbs96(maf)
    with pytest.raises(ValueError, match="reference-genome context lookup"):
        build_sbs96_matrix(maf)
    # Same from a file path (the CLI passes a pathlib.Path).
    path = tmp_path / "no_context.maf"
    maf.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="CONTEXT"):
        build_sbs96_matrix(path, sample_col="Tumor_Sample_Barcode")


def test_missing_sample_id_raises():
    maf = _minimal_maf([("C", "C", "T", "ACG")] * 3)
    maf["Start_Position"] = [1, 2, 3]
    maf["Tumor_Sample_Barcode"] = ["X", None, "  "]
    with pytest.raises(ValueError, match="2 of 3 MAF rows have a missing or empty"):
        build_sbs96_matrix(maf)


def test_missing_position_not_merged_as_duplicate():
    maf = _minimal_maf([("C", "C", "T", "ACG"), ("C", "C", "T", "ACG"), ("C", "C", "T", "ACG")])
    maf["Start_Position"] = pd.array([pd.NA, pd.NA, 5], dtype="Int64")
    ann = annotate_sbs96(maf)
    assert list(ann["drop_reason"]) == ["missing_position", "missing_position", None]
    _, dropped = build_sbs96_matrix(maf, return_dropped=True)
    assert dropped.loc["X", "missing_position"] == 2
    assert dropped.loc["X", "duplicate"] == 0
    assert dropped.loc["X", "kept"] == 1


def test_non_plus_strand_dropped():
    maf = _minimal_maf([("C", "C", "T", "ACG")] * 3)
    maf["Strand"] = ["+", "-", ""]
    ann = annotate_sbs96(maf)
    assert list(ann["drop_reason"]) == [None, "non_plus_strand", "non_plus_strand"]
    # without a Strand column, nothing is dropped for strand
    assert annotate_sbs96(maf.drop(columns="Strand"))["drop_reason"].isna().all()


def test_repeated_mismatch_counted_as_mismatch_not_duplicate():
    row = ("C", "C", "T", "TTTTAAGTTTT")  # centre A != ref C
    maf = _minimal_maf([row, row])
    maf["Start_Position"] = [7, 7]  # identical locus
    ann = annotate_sbs96(maf)
    assert list(ann["drop_reason"]) == ["context_mismatch", "context_mismatch"]


def test_non_unique_input_index_is_safe():
    maf = _minimal_maf([("C", "C", "T", "ACG"), ("G", "G", "A", "CGT")])
    maf.index = [0, 0]
    assert list(annotate_sbs96(maf)["sbs96_channel"]) == ["A[C>T]G", "A[C>T]G"]


# ------------------------------------------------------------- full fixture


def test_read_maf_skips_header_lines():
    df = read_maf(FIXTURE)
    assert len(df) == 21
    assert df.columns[0] == "Hugo_Symbol"
    assert df["Start_Position"].dtype == "Int64"


def test_full_fixture_matrix_exact():
    matrix, dropped = build_sbs96_matrix(FIXTURE, return_dropped=True)

    assert list(matrix.index) == SBS96_CHANNELS
    assert matrix.index.name == "MutationType"
    assert list(matrix.columns) == [S1, S2, S3]
    assert all(str(t) == "int64" for t in matrix.dtypes)

    expected = pd.DataFrame(0, index=SBS96_CHANNELS, columns=[S1, S2, S3], dtype="int64")
    for sample, cells in EXPECTED_CELLS.items():
        for ch, n in cells.items():
            expected.loc[ch, sample] = n
    pd.testing.assert_frame_equal(matrix, expected, check_names=False)

    assert matrix.sum().to_dict() == {S1: 6, S2: 6, S3: 0}
    assert (matrix[S3] == 0).all()  # indel-only sample still present


def test_full_fixture_drop_summary_exact():
    _, dropped = build_sbs96_matrix(FIXTURE, return_dropped=True)
    assert list(dropped.columns) == ["kept", *DROP_REASONS, "total"]
    assert list(dropped.index) == [S1, S2, S3]
    for sample, counts in EXPECTED_DROPS.items():
        full = {c: 0 for c in dropped.columns} | counts
        assert dropped.loc[sample].to_dict() == full, sample
    # every input row is accounted for
    assert int(dropped["total"].sum()) == 21
    assert (dropped.drop(columns="total").sum(axis=1) == dropped["total"]).all()
    # kept matches the matrix column sums
    matrix = build_sbs96_matrix(FIXTURE)
    assert (matrix.sum() == dropped["kept"]).all()


def test_dataframe_input_equals_path_input():
    from_path = build_sbs96_matrix(FIXTURE)
    from_df = build_sbs96_matrix(read_maf(FIXTURE))
    pd.testing.assert_frame_equal(from_path, from_df)


def test_gzipped_input(tmp_path):
    gz = tmp_path / "profile_small.maf.gz"
    with open(FIXTURE, "rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    pd.testing.assert_frame_equal(build_sbs96_matrix(gz), build_sbs96_matrix(FIXTURE))

    # gzip detected by magic bytes, not by extension
    no_ext = tmp_path / "profile_small_noext"
    shutil.copy(gz, no_ext)
    pd.testing.assert_frame_equal(build_sbs96_matrix(no_ext), build_sbs96_matrix(FIXTURE))


def test_drop_summary_is_logged(caplog):
    with caplog.at_level("INFO", logger="fingerprints.profile"):
        build_sbs96_matrix(FIXTURE)
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "12 SNVs kept" in msg
    assert "not_snp=4" in msg and "duplicate=1" in msg
    assert "missing_position=0" in msg and "non_plus_strand=0" in msg
