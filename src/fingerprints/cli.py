"""Command-line interface for Cancer Fingerprints.

Subcommands:
    profile  MAF -> 96-channel SBS matrix (TSV)
    fit      fit a profile to COSMIC signatures (Phase 2)
    explain  plain-English report from exposures (Phase 2)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from fingerprints import __version__

EXIT_OK = 0
EXIT_ERROR = 1
# Distinct from argparse's usage-error code (2).
EXIT_NOT_IMPLEMENTED = 3
DEFAULT_SAMPLE_COL = "Tumor_Sample_Barcode"


def _error(message: str) -> int:
    """Print a one-line error to stderr and return the error exit code."""
    one_line = " ".join(message.split())
    print(f"error: {one_line}", file=sys.stderr)
    return EXIT_ERROR


def _cmd_profile(args: argparse.Namespace) -> int:
    """Build the SBS96 matrix from a MAF file and write it as TSV."""
    if not args.maf.is_file():
        return _error(f"MAF file not found: {args.maf}")

    # Imported lazily so the CLI (and --help) works without the profile module.
    from fingerprints import profile

    try:
        matrix = profile.build_sbs96_matrix(args.maf, sample_col=args.sample_col)
    except (FileNotFoundError, ValueError) as exc:
        return _error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output, sep="\t")
    print(
        f"Wrote {matrix.shape[0]} channels x {matrix.shape[1]} samples to {args.output}",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    """Placeholder for subcommands arriving in a later phase."""
    print(f"fingerprints {args.command}: not implemented yet (Phase 2)", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="fingerprints",
        description=(
            "Explain a tumour's somatic mutations using COSMIC mutational signatures. "
            "Research and education only, not for diagnosis."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_profile = sub.add_parser("profile", help="Build a 96-channel SBS matrix from a MAF file.")
    p_profile.add_argument("maf", type=Path, help="Path to a MAF file (optionally gzipped).")
    p_profile.add_argument(
        "-o", "--output", type=Path, required=True, help="Output TSV path (channels x samples)."
    )
    p_profile.add_argument(
        "--sample-col",
        default=DEFAULT_SAMPLE_COL,
        help=f"MAF column holding the sample ID (default: {DEFAULT_SAMPLE_COL}).",
    )
    p_profile.set_defaults(func=_cmd_profile)

    p_fit = sub.add_parser("fit", help="Fit profiles to COSMIC signatures (Phase 2).")
    p_fit.set_defaults(func=_cmd_not_implemented)

    p_explain = sub.add_parser("explain", help="Plain-English report from exposures (Phase 2).")
    p_explain.set_defaults(func=_cmd_not_implemented)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``fingerprints`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
