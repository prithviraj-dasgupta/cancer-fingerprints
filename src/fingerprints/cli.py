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
# Kept in sync with fingerprints.profile.ADJACENT_SNV_MODES (not imported here so that
# --help works without the profile module; a test checks the two match).
ADJACENT_SNV_CHOICES = ("exclude", "count_as_sbs")
DEFAULT_ADJACENT_SNVS = "exclude"
DEFAULT_CONFIG = Path("config/config.yaml")


class ConfigError(ValueError):
    """A config file exists but holds an unusable value."""


def _resolve_adjacent_snvs(flag: str | None, config_arg: Path | None) -> tuple[str, str]:
    """Pick the adjacent-SNV mode and say where it came from.

    Precedence: explicit ``--adjacent-snvs`` > ``profile.adjacent_snvs`` in the
    config > default ("exclude"). Returns ``(mode, source)``, where ``source``
    is a short human-readable description for the log line.

    ``config_arg`` is ``None`` when ``--config`` was not given. Then the default
    ``config/config.yaml`` is used if it exists, and a missing default file is
    fine. An explicitly given ``--config`` that does not exist is an error
    (ConfigError), even if ``--adjacent-snvs`` is also given. An unparsable file
    or an invalid value is also a ConfigError.
    """
    explicit = config_arg is not None
    config_path = config_arg if explicit else DEFAULT_CONFIG
    if explicit and not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    if flag is not None:
        return flag, "from --adjacent-snvs"
    if not config_path.is_file():
        return DEFAULT_ADJACENT_SNVS, "default; no config found"

    import yaml  # lazy: keeps --help fast and independent of optional imports

    try:
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read config {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError(f"config {config_path} is not a YAML mapping")
    section = config.get("profile")
    if section is not None and not isinstance(section, dict):
        raise ConfigError(f"config {config_path}: 'profile' must be a mapping")
    value = (section or {}).get("adjacent_snvs")
    if value is None:
        return DEFAULT_ADJACENT_SNVS, f"default; not set in config {config_path}"
    if value not in ADJACENT_SNV_CHOICES:
        raise ConfigError(
            f"config {config_path}: profile.adjacent_snvs must be one of "
            f"{list(ADJACENT_SNV_CHOICES)}, got {value!r}"
        )
    return value, f"from config {config_path}"


def _error(message: str) -> int:
    """Print a one-line error to stderr and return the error exit code."""
    one_line = " ".join(message.split())
    print(f"error: {one_line}", file=sys.stderr)
    return EXIT_ERROR


def _cmd_profile(args: argparse.Namespace) -> int:
    """Build the SBS96 matrix from a MAF file and write it as TSV."""
    if not args.maf.is_file():
        return _error(f"MAF file not found: {args.maf}")

    try:
        adjacent_snvs, source = _resolve_adjacent_snvs(args.adjacent_snvs, args.config)
    except ConfigError as exc:
        return _error(str(exc))

    # Imported lazily so the CLI (and --help) works without the profile module.
    from fingerprints import profile

    try:
        matrix = profile.build_sbs96_matrix(
            args.maf, sample_col=args.sample_col, adjacent_snvs=adjacent_snvs
        )
    except (FileNotFoundError, ValueError) as exc:
        return _error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output, sep="\t")
    print(
        f"Wrote {matrix.shape[0]} channels x {matrix.shape[1]} samples to {args.output}",
        file=sys.stderr,
    )
    print(f"adjacent_snvs={adjacent_snvs} ({source})", file=sys.stderr)
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
    p_profile.add_argument(
        "--adjacent-snvs",
        choices=ADJACENT_SNV_CHOICES,
        default=None,  # None = not given: fall back to --config, then "exclude"
        help=(
            "How to treat same-sample SNVs at adjacent positions (likely doublets): "
            "'exclude' leaves them out of SBS96, 'count_as_sbs' counts them as ordinary SBS, "
            "as SigProfilerMatrixGenerator 1.6.0 does. Overrides profile.adjacent_snvs in "
            "--config; if neither is set, 'exclude'. See docs/decisions.md D5."
        ),
    )
    p_profile.add_argument(
        "--config",
        type=Path,
        default=None,  # None = not given: use DEFAULT_CONFIG if it exists
        help=(
            f"Project config YAML (default: {DEFAULT_CONFIG}, relative to the current "
            "directory; skipped if absent). Only profile.adjacent_snvs is read. A file "
            "given explicitly must exist."
        ),
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
