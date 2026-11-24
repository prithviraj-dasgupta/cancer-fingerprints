#!/usr/bin/env python
"""Generate the SYNTHETIC example MAF for the app (no real patient data).

Mutation counts are sampled (multinomial, fixed seed) from a chosen mix of COSMIC v3.4 exome
signatures read from ``data/raw/signatures``. Each sampled channel becomes one SNV with a
random plus-strand strand choice, random flanking bases (11-mer ``CONTEXT``), a random
chromosome and a random position (spaced >= 100 bp apart, so none are adjacent). Gene names
are dummy (``Unknown``) and the sample is ``SYNTHETIC-0001``.

The committed output contains sampled mutations only. It does not contain any COSMIC matrix.

Usage::

    uv run python scripts/make_example_maf.py [--output app/example/synthetic_melanoma.maf]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fingerprints import appcore, fit  # noqa: E402
from fingerprints.profile import SBS96_CHANNELS  # noqa: E402

SEED = 20261123
N_SNVS = 400
SAMPLE = "SYNTHETIC-0001"
#: Planted mix: UV-dominated, melanoma-like (fractions of mutations; sum to 1).
MIX: dict[str, float] = {"SBS7a": 0.45, "SBS7b": 0.25, "SBS7d": 0.05, "SBS1": 0.10, "SBS5": 0.15}
CHROMS = [f"chr{i}" for i in range(1, 23)]
HEADER = [
    "#SYNTHETIC EXAMPLE: simulated mutations, no real patient data.",
    "#Sampled from a planted signature mix with a fixed random seed (scripts/make_example_maf.py).",
    "#Positions, flanking sequence and sample name are random; Hugo_Symbol is a dummy.",
]
COLUMNS = [
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Strand",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
    "CONTEXT",
]
_COMP = str.maketrans("ACGT", "TGCA")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def channel_counts(profile: pd.Series, n: int, rng: np.random.Generator) -> pd.Series:
    """Multinomial draw of ``n`` mutations over the 96 channels."""
    probs = (profile / profile.sum()).reindex(SBS96_CHANNELS).to_numpy()
    return pd.Series(rng.multinomial(n, probs), index=SBS96_CHANNELS)


def snv_row(channel: str, rng: np.random.Generator) -> tuple[str, str, str]:
    """(ref, alt, 11-mer plus-strand context) for a channel, strand chosen at random."""
    five, ref, alt, three = channel[0], channel[2], channel[4], channel[6]
    flank = "".join(rng.choice(list("ACGT"), size=8))
    context = flank[:4] + five + ref + three + flank[4:]
    if rng.random() < 0.5:  # mutation on the opposite strand: report plus-strand view
        context, ref, alt = _revcomp(context), _revcomp(ref), _revcomp(alt)
    return ref, alt, context


def build_maf(counts: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    channels = [c for c, k in counts.items() for _ in range(int(k))]
    slots = rng.choice(len(CHROMS) * 20_000, size=len(channels), replace=False)
    rows = []
    for channel, slot in zip(channels, slots, strict=True):
        chrom, idx = CHROMS[int(slot) // 20_000], int(slot) % 20_000
        pos = idx * 100 + int(rng.integers(1, 90))  # >= 100 bp apart: never adjacent
        ref, alt, context = snv_row(channel, rng)
        rows.append(
            ["Unknown", chrom, pos, pos, "+", "Missense_Mutation", "SNP", ref, ref, alt, SAMPLE]
            + [context]
        )
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["Chromosome", "Start_Position"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "app/example/synthetic_melanoma.maf")
    args = parser.parse_args(argv)
    config = appcore.load_config()
    reference = fit.load_reference(appcore.reference_path(config))
    sub = reference[list(MIX)]
    sub = sub / sub.sum(axis=0)
    profile = sub @ pd.Series(MIX)
    rng = np.random.default_rng(SEED)
    maf = build_maf(channel_counts(profile, N_SNVS, rng), rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(HEADER) + "\n")
        maf.to_csv(fh, sep="\t", index=False)
    print(f"Wrote {len(maf)} synthetic SNVs to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
