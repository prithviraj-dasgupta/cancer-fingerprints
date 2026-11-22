#!/usr/bin/env python
"""Find, check and copy the COSMIC SBS reference signature matrix.

The COSMIC SBS v3.4 matrix comes from the copy bundled inside the SigProfilerAssignment
pip package. The script does not contact the COSMIC website. Steps:

1. find the configured file in the installed package via ``importlib.resources``
   (default ``data/Reference_Signatures/GRCh38/COSMIC_v3.4_SBS_GRCh38_exome.txt``);
2. check it: 96 rows whose labels match ``fingerprints.profile.SBS96_CHANNELS`` as a set,
   columns named SBS*, numeric and non-negative values, each column summing to about 1, and
   a sha256 that matches the one pinned in ``config/config.yaml``;
3. copy it to ``data/raw/signatures/`` (gitignored) and write ``<file>.provenance.json``
   next to it, recording the package version, sha256 and licence.

COSMIC licence: free for academic / non-commercial internal research. No redistribution.
Never commit the file. See ``signatures.licence`` in config/config.yaml.

Usage::

    uv sync --extra signatures
    uv run python scripts/download_signatures.py
    uv run python scripts/download_signatures.py --check   # check the local copy only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"
PACKAGE = "SigProfilerAssignment"
SIGNATURE_COLUMN = re.compile(r"^SBS\d+[a-z]?$")
INSTALL_HINT = (
    f"{PACKAGE} is not installed. Install the optional extra with:\n"
    "    uv sync --extra signatures\n"
    "then re-run this script."
)


class SignatureSourceError(RuntimeError):
    """The bundled signature file could not be found."""


@dataclass
class ValidationResult:
    ok: bool
    n_channels: int = 0
    n_signatures: int = 0
    errors: list[str] = field(default_factory=list)


def _channels() -> list[str]:
    from fingerprints.profile import SBS96_CHANNELS

    return list(SBS96_CHANNELS)


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_signature_matrix(
    path: Path,
    expected_channels: list[str] | None = None,
    sum_tolerance: float = 0.01,
) -> ValidationResult:
    """Check that a tab-separated signature matrix looks like a COSMIC SBS96 file.

    The first column must hold the channel labels (e.g. ``A[C>A]A``). Row order does not
    matter: COSMIC files are sorted lexicographically, not in SBS96_CHANNELS order.
    """
    import pandas as pd

    expected = list(expected_channels) if expected_channels is not None else _channels()
    errors: list[str] = []
    try:
        df = pd.read_csv(path, sep="\t", index_col=0)
    except Exception as exc:  # report any parse failure
        return ValidationResult(False, errors=[f"could not parse {path}: {exc}"])

    if df.shape[0] != 96:
        errors.append(f"expected 96 rows, found {df.shape[0]}")
    labels = [str(x) for x in df.index]
    dupes = sorted({x for x in labels if labels.count(x) > 1})
    if dupes:
        errors.append(f"duplicate channel labels: {dupes[:5]}")
    missing = sorted(set(expected) - set(labels))
    extra = sorted(set(labels) - set(expected))
    if missing:
        errors.append(f"{len(missing)} expected channels missing, e.g. {missing[:3]}")
    if extra:
        errors.append(f"{len(extra)} unexpected channel labels, e.g. {extra[:3]}")

    if df.shape[1] == 0:
        errors.append("no signature columns")
    bad_cols = [c for c in df.columns if not SIGNATURE_COLUMN.match(str(c))]
    if bad_cols:
        errors.append(f"columns not named SBS<n>: {bad_cols[:5]}")

    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        errors.append("non-numeric or missing values present")
    elif (numeric < 0).any().any():
        errors.append("negative values present")
    else:
        sums = numeric.sum(axis=0)
        off = sums[(sums - 1.0).abs() > sum_tolerance]
        if len(off):
            errors.append(
                f"{len(off)} columns do not sum to 1 (+/- {sum_tolerance}): "
                + ", ".join(f"{k}={v:.4f}" for k, v in off.head(5).items())
            )

    return ValidationResult(not errors, df.shape[0], df.shape[1], errors)


def package_version(package: str = PACKAGE) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def locate_bundled_file(package_path: str, file_name: str, package: str = PACKAGE) -> Path:
    """Return the filesystem path of a data file inside an installed package."""
    import importlib.resources as resources
    import importlib.util

    if importlib.util.find_spec(package) is None:
        raise SignatureSourceError(INSTALL_HINT)
    resource = resources.files(package).joinpath(*Path(package_path).parts, file_name)
    if not resource.is_file():
        raise SignatureSourceError(
            f"{file_name} not found in {package} {package_version(package)} under "
            f"{package_path}. Check signatures.package_version in config/config.yaml "
            "matches the installed package."
        )
    return Path(str(resource))


def destination_path(out_root: Path, raw_signatures: str, file_name: str) -> Path:
    return out_root / raw_signatures / file_name


def is_up_to_date(dest: Path, expected_sha256: str | None) -> bool:
    return (
        dest.is_file()
        and provenance_path(dest).is_file()
        and bool(expected_sha256)
        and sha256_of(dest) == expected_sha256
    )


def build_provenance(sig_cfg: dict[str, Any], sha256: str, pkg_version: str | None) -> dict:
    return {
        "file": sig_cfg["file"],
        "sha256": sha256,
        "source": sig_cfg["source"],
        "package_version": pkg_version,
        "package_path": sig_cfg.get("package_path"),
        "cosmic_version": sig_cfg.get("cosmic_version"),
        "genome_build": sig_cfg.get("genome_build"),
        "assay": sig_cfg.get("assay"),
        "copied_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "licence": sig_cfg.get("licence", {}).get("summary"),
        "licence_terms_url": sig_cfg.get("licence", {}).get("terms_url"),
        "citation": sig_cfg.get("citation", []),
        "note": "COSMIC data: do not commit or redistribute this file.",
    }


def install_signatures(
    src: Path, dest: Path, sig_cfg: dict[str, Any], pkg_version: str | None
) -> Path:
    """Validate ``src``, check its sha256, copy to ``dest`` and write provenance JSON."""
    result = validate_signature_matrix(src)
    if not result.ok:
        raise ValueError(f"{src} failed validation: " + "; ".join(result.errors))
    expected = required_sha256(sig_cfg)
    digest = sha256_of(src)
    if digest != expected:
        raise ValueError(
            f"sha256 mismatch for {src.name}: got {digest}, config pins {expected}. "
            "The package may have changed the file. Inspect it before updating the config."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(dest)
    provenance_path(dest).write_text(
        json.dumps(build_provenance(sig_cfg, digest, pkg_version), indent=2) + "\n"
    )
    return dest


def required_sha256(sig_cfg: dict[str, Any]) -> str:
    """Return the pinned sha256 from config, or raise ValueError if it is missing."""
    sha = str(sig_cfg.get("sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError(
            "signatures.sha256 is missing or not a 64-character hex digest in "
            "config/config.yaml. Pin the hash of the bundled file before copying it."
        )
    return sha


def provenance_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".provenance.json")


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, default=REPO_ROOT, help="Root for config paths")
    p.add_argument("--check", action="store_true", help="Only validate the local copy")
    p.add_argument("--force", action="store_true", help="Copy even if up to date")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    sig = cfg["signatures"]
    dest = destination_path(args.out, cfg["paths"]["raw_signatures"], sig["file"])
    try:
        expected_sha = required_sha256(sig)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.check:
        if not dest.is_file():
            print(f"ERROR: {dest} not found. Run without --check first.", file=sys.stderr)
            return 1
        res = validate_signature_matrix(dest)
        sha_ok = sha256_of(dest) == expected_sha
        prov_ok = provenance_path(dest).is_file()
        print(f"{dest}: {res.n_channels} channels x {res.n_signatures} signatures")
        for e in res.errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        if not sha_ok:
            print("  ERROR: sha256 does not match config", file=sys.stderr)
        if not prov_ok:
            print(
                f"  ERROR: {provenance_path(dest).name} missing; re-run with --force",
                file=sys.stderr,
            )
        return 0 if res.ok and sha_ok and prov_ok else 1

    if not args.force and is_up_to_date(dest, expected_sha):
        print(f"Up to date: {dest}")
        return 0

    try:
        src = locate_bundled_file(sig["package_path"], sig["file"])
    except SignatureSourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    installed = package_version()
    if installed != str(sig.get("package_version")):
        print(
            f"WARNING: installed {PACKAGE} {installed} differs from pinned "
            f"{sig.get('package_version')}; the sha256 check will catch content changes.",
            file=sys.stderr,
        )
    try:
        install_signatures(src, dest, sig, installed)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Copied {sig['file']} ({PACKAGE} {installed}) -> {dest}")
    print("COSMIC licence: academic/non-commercial use only. Do not commit or redistribute.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
