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
    EVENT_COLUMNS,
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
        "T[C>T]G": 1,  # homozygous call (Allele1 == Allele2 != ref): kept
    },
    S2: {
        "T[C>A]T": 3,  # two C>A in TCT + G>T in AGA -> TCT
        "A[T>C]A": 1,
        "T[C>G]A": 1,  # G>C in TGA -> TCA
        "A[C>T]G": 2,  # same locus as an S1 mutation (other sample) + chr22:201
        "T[T>C]G": 1,  # chr19:502 A>G (CAA -> TTG), next to S1 chr19:501, other sample
        "C[C>T]G": 1,  # chrX:301, next to the multiallelic chrX:300: kept
    },
    S3: {},  # indels only
}

EXPECTED_DROPS = {
    S1: {
        "kept": 7,
        "duplicate": 1,
        "not_snp": 1,
        "context_mismatch": 1,
        "tri_allelic": 1,  # chr18:3000 C, A/T
        "adjacent_snv_dbs": 2,  # chr19:500-501
        "total": 13,
        "dbs_events": 1,
    },
    S2: {
        "not_snp": 1,
        "missing_context": 1,
        "non_acgt": 1,
        "no_alt": 1,
        "context_mismatch": 1,  # chr22:200; its neighbour chr22:201 is kept
        "adjacent_snv_dbs": 3,  # chr21:100-102
        "multiallelic_position": 2,  # chrX:300 C>T + C>A
        "kept": 9,
        "total": 19,
        "mnv_events": 1,
    },
    S3: {"kept": 0, "not_snp": 2, "total": 2},
}
N_FIXTURE_ROWS = 34


def _minimal_maf(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """rows: (ref, allele1, allele2, context) for a single sample, SNPs only."""
    return pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["X"] * len(rows),
            "Chromosome": ["chr1"] * len(rows),
            # 10 bp apart so default rows are never adjacent (doublet rule)
            "Start_Position": [10 * (i + 1) for i in range(len(rows))],
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


def test_tri_allelic_dropped():
    ann = annotate_sbs96(_minimal_maf([("C", "A", "T", "GGGGACGGGGG")]))
    assert ann.loc[0, "drop_reason"] == "tri_allelic"
    assert ann.loc[0, "sbs96_channel"] is None


def test_homozygous_call_kept():
    ann = annotate_sbs96(_minimal_maf([("C", "T", "T", "GGGGACGGGGG")]))
    assert ann.loc[0, "drop_reason"] is None
    assert ann.loc[0, "sbs96_channel"] == "A[C>T]G"


# ------------------------------------------------------- adjacent SNVs (DBS)


def _snvs(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    """rows: (sample, chrom, pos); every row a valid C>T in ACG."""
    maf = _minimal_maf([("C", "C", "T", "TTTTACGTTTT")] * len(rows))
    maf["Tumor_Sample_Barcode"] = [r[0] for r in rows]
    maf["Chromosome"] = [r[1] for r in rows]
    maf["Start_Position"] = [r[2] for r in rows]
    return maf


def test_adjacent_pair_excluded_one_dbs_event():
    maf = _snvs([("X", "chr1", 100), ("X", "chr1", 101), ("X", "chr1", 200)])
    ann = annotate_sbs96(maf)
    assert list(ann["drop_reason"]) == ["adjacent_snv_dbs", "adjacent_snv_dbs", None]
    assert list(ann["adjacent_run_len"].iloc[:2]) == [2, 2]
    assert ann.loc[0, "adjacent_run_id"] == ann.loc[1, "adjacent_run_id"]
    matrix, dropped = build_sbs96_matrix(maf, return_dropped=True)
    assert int(matrix["X"].sum()) == 1
    assert dropped.loc["X", "adjacent_snv_dbs"] == 2
    assert dropped.loc["X", "dbs_events"] == 1
    assert dropped.loc["X", "mnv_events"] == 0


def test_run_of_three_excluded_one_mnv_event():
    maf = _snvs([("X", "chr1", 102), ("X", "chr1", 100), ("X", "chr1", 101)])  # unsorted
    matrix, dropped = build_sbs96_matrix(maf, return_dropped=True)
    assert int(matrix["X"].sum()) == 0
    assert dropped.loc["X", "adjacent_snv_dbs"] == 3
    assert dropped.loc["X", "dbs_events"] == 0
    assert dropped.loc["X", "mnv_events"] == 1


def test_two_separate_runs_counted_separately():
    maf = _snvs([("X", "chr1", 100), ("X", "chr1", 101), ("X", "chr1", 300), ("X", "chr1", 301)])
    _, dropped = build_sbs96_matrix(maf, return_dropped=True)
    assert dropped.loc["X", "adjacent_snv_dbs"] == 4
    assert dropped.loc["X", "dbs_events"] == 2


@pytest.mark.parametrize(
    "rows",
    [
        [("X", "chr1", 100), ("Y", "chr1", 101)],  # different samples
        [("X", "chr1", 100), ("X", "chr2", 101)],  # different chromosomes
        [("X", "chr1", 100), ("X", "chr1", 102)],  # 2 bp apart
    ],
)
def test_not_adjacent_kept(rows):
    ann = annotate_sbs96(_snvs(rows))
    assert ann["drop_reason"].isna().all()
    assert ann["adjacent_run_id"].isna().all()


def test_duplicate_neighbour_does_not_make_run():
    # 100 (twice) + 101: the duplicate is removed first, leaving a pair 100/101.
    # 200 (twice): duplicate only, same position, never "adjacent" to itself.
    maf = _snvs([("X", "chr1", 100), ("X", "chr1", 100), ("X", "chr1", 101)])
    maf = pd.concat([maf, _snvs([("X", "chr1", 200), ("X", "chr1", 200)])], ignore_index=True)
    ann = annotate_sbs96(maf)
    assert list(ann["drop_reason"]) == [
        "adjacent_snv_dbs",
        "duplicate",
        "adjacent_snv_dbs",
        None,
        "duplicate",
    ]
    _, dropped = build_sbs96_matrix(maf, return_dropped=True)
    assert dropped.loc["X", "dbs_events"] == 1


def _snvs_alt(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    """rows: (sample, chrom, pos, alt); every row a valid C>alt in ACA context."""
    maf = _minimal_maf([("C", "C", r[3], "TTTTACATTTT") for r in rows])
    maf["Tumor_Sample_Barcode"] = [r[0] for r in rows]
    maf["Chromosome"] = [r[1] for r in rows]
    maf["Start_Position"] = [r[2] for r in rows]
    return maf


@pytest.mark.parametrize("mode", ["exclude", "count_as_sbs"])
def test_multiallelic_position_next_to_snv_any_order(mode):
    """42 C>T, 43 C>T, 43 C>A: 43a/43b dropped as multiallelic; 42 kept as SBS."""
    rows = [("X", "chr1", 42, "T"), ("X", "chr1", 43, "T"), ("X", "chr1", 43, "A")]
    results = []
    for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
        maf = _snvs_alt([rows[i] for i in order])
        ann = annotate_sbs96(maf, adjacent_snvs=mode)
        by_row = {
            (p, a): r
            for p, a, r in zip(
                ann["Start_Position"], ann["Tumor_Seq_Allele2"], ann["drop_reason"], strict=True
            )
        }
        assert by_row == {
            (42, "T"): None,
            (43, "T"): "multiallelic_position",
            (43, "A"): "multiallelic_position",
        }
        assert ann["adjacent_run_id"].isna().all()
        matrix, dropped = build_sbs96_matrix(maf, return_dropped=True, adjacent_snvs=mode)
        assert dropped.loc["X", "multiallelic_position"] == 2
        assert dropped.loc["X", "kept"] == 1
        assert dropped.loc["X", "dbs_events"] == 0
        assert dropped.loc["X", "mnv_events"] == 0
        assert matrix.loc["A[C>T]A", "X"] == 1
        assert int(matrix["X"].sum()) == 1
        results.append((matrix, dropped))
    for matrix, dropped in results[1:]:
        pd.testing.assert_frame_equal(matrix, results[0][0])
        pd.testing.assert_frame_equal(dropped, results[0][1])


def test_multiallelic_position_then_neighbour():
    """43 C>T, 43 C>A, 44 C>T: 43a/43b dropped, 44 kept, no DBS event."""
    maf = _snvs_alt([("X", "chr1", 43, "T"), ("X", "chr1", 43, "A"), ("X", "chr1", 44, "T")])
    ann = annotate_sbs96(maf)
    assert ann["drop_reason"].tolist() == ["multiallelic_position", "multiallelic_position", None]
    _, dropped = build_sbs96_matrix(maf, return_dropped=True)
    assert dropped.loc["X", "adjacent_snv_dbs"] == 0
    assert dropped.loc["X", "dbs_events"] == 0
    # 44/45 still form an ordinary DBS
    maf = pd.concat([maf, _snvs_alt([("X", "chr1", 45, "T")])], ignore_index=True)
    _, dropped = build_sbs96_matrix(maf, return_dropped=True)
    assert dropped.loc["X", "multiallelic_position"] == 2
    assert dropped.loc["X", "adjacent_snv_dbs"] == 2
    assert dropped.loc["X", "dbs_events"] == 1
    assert dropped.loc["X", "mnv_events"] == 0


def test_isolated_multiallelic_position_dropped():
    maf = _snvs_alt([("X", "chr1", 43, "T"), ("X", "chr1", 43, "A")])
    ann = annotate_sbs96(maf)
    assert ann["drop_reason"].tolist() == ["multiallelic_position"] * 2
    assert ann["sbs96_channel"].isna().all()


def test_three_alts_at_one_position_all_dropped():
    maf = _snvs_alt([("X", "chr1", 43, "T"), ("X", "chr1", 43, "A"), ("X", "chr1", 43, "G")])
    _, dropped = build_sbs96_matrix(maf, return_dropped=True)
    assert dropped.loc["X", "multiallelic_position"] == 3
    assert dropped.loc["X", "kept"] == 0


def test_multiallelic_with_exact_duplicate():
    """43 C>T twice + 43 C>A: one duplicate, then the two distinct alts are multiallelic."""
    maf = _snvs_alt([("X", "chr1", 43, "T"), ("X", "chr1", 43, "T"), ("X", "chr1", 43, "A")])
    ann = annotate_sbs96(maf)
    assert ann["drop_reason"].tolist() == [
        "multiallelic_position",
        "duplicate",
        "multiallelic_position",
    ]


def test_exact_duplicate_alone_is_not_multiallelic():
    maf = _snvs_alt([("X", "chr1", 43, "T"), ("X", "chr1", 43, "T")])
    assert annotate_sbs96(maf)["drop_reason"].tolist() == [None, "duplicate"]


def test_same_position_different_samples_not_multiallelic():
    maf = _snvs_alt([("X", "chr1", 43, "T"), ("Y", "chr1", 43, "A")])
    ann = annotate_sbs96(maf)
    assert ann["drop_reason"].isna().all()


def test_whitespace_in_sample_and_chrom_normalised():
    maf = _snvs([("S1", "chr1", 100), (" S1 ", " chr1", 101), ("S1", "chr1 ", 100)])
    ann = annotate_sbs96(maf)
    # " S1 "/" chr1" is the same sample/chrom: 100 & 101 adjacent; 3rd row is a duplicate
    assert ann["drop_reason"].tolist() == ["adjacent_snv_dbs", "adjacent_snv_dbs", "duplicate"]
    assert ann["Tumor_Sample_Barcode"].tolist() == ["S1", "S1", "S1"]
    assert ann["Chromosome"].tolist() == ["chr1", "chr1", "chr1"]
    matrix, dropped = build_sbs96_matrix(maf, return_dropped=True)
    assert list(matrix.columns) == ["S1"]
    assert dropped.loc["S1", "total"] == 3
    assert dropped.loc["S1", "dbs_events"] == 1


def test_dropped_neighbour_is_ignored():
    """Documented choice: only valid SNVs take part in adjacency."""
    maf = _snvs([("X", "chr1", 100), ("X", "chr1", 101), ("X", "chr1", 102)])
    maf.loc[0, "CONTEXT"] = "TTTTAAGTTTT"  # context_mismatch
    maf.loc[2, "Variant_Type"] = "DNP"  # not_snp
    ann = annotate_sbs96(maf)
    assert list(ann["drop_reason"]) == ["context_mismatch", None, "not_snp"]
    assert ann.loc[1, "sbs96_channel"] == "A[C>T]G"


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
    assert len(df) == N_FIXTURE_ROWS
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

    assert matrix.sum().to_dict() == {S1: 7, S2: 9, S3: 0}
    assert (matrix[S3] == 0).all()  # indel-only sample still present


def test_full_fixture_drop_summary_exact():
    _, dropped = build_sbs96_matrix(FIXTURE, return_dropped=True)
    assert list(dropped.columns) == ["kept", *DROP_REASONS, "total", *EVENT_COLUMNS]
    assert list(dropped.index) == [S1, S2, S3]
    for sample, counts in EXPECTED_DROPS.items():
        full = {c: 0 for c in dropped.columns} | counts
        assert dropped.loc[sample].to_dict() == full, sample
    # every input row is accounted for
    assert int(dropped["total"].sum()) == N_FIXTURE_ROWS
    row_cols = ["kept", *DROP_REASONS]
    assert (dropped[row_cols].sum(axis=1) == dropped["total"]).all()
    # kept matches the matrix column sums
    matrix = build_sbs96_matrix(FIXTURE)
    assert (matrix.sum() == dropped["kept"]).all()


def test_count_as_sbs_full_fixture_exact():
    """Adjacent SNVs kept as SBS: S1 +2 (chr19:500-501), S2 +3 (chr21:100-102)."""
    matrix, dropped = build_sbs96_matrix(FIXTURE, return_dropped=True, adjacent_snvs="count_as_sbs")
    cells = {s: dict(c) for s, c in EXPECTED_CELLS.items()}
    # chr19:500/501 C>T, plus-strand contexts ACC / CCA -> A[C>T]C and C[C>T]A
    cells[S1]["A[C>T]C"] = 1
    cells[S1]["C[C>T]A"] = 1
    # chr21:100-102 C>A, contexts TCC / CCC / CCT
    cells[S2]["T[C>A]C"] = 1
    cells[S2]["C[C>A]C"] = 1
    cells[S2]["C[C>A]T"] = 1
    expected = pd.DataFrame(0, index=SBS96_CHANNELS, columns=[S1, S2, S3], dtype="int64")
    for sample, sample_cells in cells.items():
        for ch, n in sample_cells.items():
            expected.loc[ch, sample] = n
    pd.testing.assert_frame_equal(matrix, expected, check_names=False)
    assert matrix.sum().to_dict() == {S1: 9, S2: 12, S3: 0}

    assert (dropped["adjacent_snv_dbs"] == 0).all()
    assert dropped["kept"].to_dict() == {S1: 9, S2: 12, S3: 0}
    assert int(dropped["total"].sum()) == N_FIXTURE_ROWS
    # Event counts identical to the default (exclude) mode.
    _, dropped_excl = build_sbs96_matrix(FIXTURE, return_dropped=True)
    pd.testing.assert_frame_equal(dropped[EVENT_COLUMNS], dropped_excl[EVENT_COLUMNS])
    assert dropped.loc[S1, "dbs_events"] == 1 and dropped.loc[S2, "mnv_events"] == 1
    # All other drop reasons unchanged.
    others = [r for r in DROP_REASONS if r != "adjacent_snv_dbs"]
    pd.testing.assert_frame_equal(dropped[others], dropped_excl[others])


def test_count_as_sbs_keeps_run_annotation():
    maf = _snvs([("X", "chr1", 100), ("X", "chr1", 101)])
    ann = annotate_sbs96(maf, adjacent_snvs="count_as_sbs")
    assert ann["drop_reason"].isna().all()
    assert list(ann["adjacent_run_len"]) == [2, 2]


def test_adjacent_snvs_default_is_exclude():
    import inspect

    for fn in (build_sbs96_matrix, annotate_sbs96):
        assert inspect.signature(fn).parameters["adjacent_snvs"].default == "exclude"
    pd.testing.assert_frame_equal(
        build_sbs96_matrix(FIXTURE), build_sbs96_matrix(FIXTURE, adjacent_snvs="exclude")
    )


@pytest.mark.parametrize("bad", ["merge", "EXCLUDE", "", None])
def test_adjacent_snvs_invalid_value_raises(bad):
    with pytest.raises(ValueError, match="adjacent_snvs must be one of"):
        build_sbs96_matrix(FIXTURE, adjacent_snvs=bad)
    with pytest.raises(ValueError, match="adjacent_snvs must be one of"):
        annotate_sbs96(read_maf(FIXTURE), adjacent_snvs=bad)


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
    assert "16 SNVs kept" in msg
    assert "multiallelic_position=2" in msg
    assert "tri_allelic=1" in msg and "adjacent_snv_dbs=5" in msg
    assert "1 DBS (2 rows), 1 MNV (3+ rows)" in msg
    assert "not_snp=4" in msg and "duplicate=1" in msg
    assert "missing_position=0" in msg and "non_plus_strand=0" in msg
