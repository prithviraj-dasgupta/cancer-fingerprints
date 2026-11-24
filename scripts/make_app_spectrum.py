#!/usr/bin/env python
"""Draw the SBS96 spectrum of the bundled synthetic melanoma example for the README.

Uses the app's own data preparation (``fingerprints.plots.spectrum_frame`` and colours) and
profile code, drawn with matplotlib so no browser or kaleido is needed.

    uv run python scripts/make_app_spectrum.py

Output: docs/img/app-spectrum.png. No COSMIC files and no real patient data are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fingerprints import plots, profile  # noqa: E402
from fingerprints.profile import SUBSTITUTIONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maf", type=Path, default=ROOT / "app/example/synthetic_melanoma.maf")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/img/app-spectrum.png")
    args = parser.parse_args()

    maf = profile.read_maf(args.maf)
    matrix, _dropped = profile.build_sbs96_matrix(maf, return_dropped=True)
    sample = matrix.columns[0]
    counts = matrix[sample]
    frame = plots.spectrum_frame(counts)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    for sub in SUBSTITUTIONS:
        part = frame[frame["substitution"] == sub]
        ax.bar(part["channel"], part["count"], color=plots.SUBSTITUTION_COLOURS[sub], label=sub)
    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels(frame["context"], rotation=90, fontsize=5)
    ax.set_xlim(-0.8, len(frame) - 0.2)
    ax.set_ylabel("Number of SNVs")
    ax.set_xlabel("Trinucleotide context (5' [ref>alt] 3')")
    ax.set_title(f"SBS96 mutation spectrum: {sample} (synthetic example, {int(counts.sum())} SNVs)")
    ax.legend(title="Substitution", ncol=6, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
