#!/usr/bin/env bash
# Install the app + COSMIC reference in a Codespace, retrying flaky PyPI downloads.
set -uo pipefail
pip install --user --quiet uv
export UV_HTTP_TIMEOUT=180 UV_CONCURRENT_DOWNLOADS=4
ok=0
for i in 1 2 3 4 5; do
  if uv sync --locked --extra app --extra signatures; then ok=1; break; fi
  echo "uv sync failed (attempt $i of 5), retrying in 10s..."; sleep 10
done
[ "$ok" = 1 ] || { echo "Install failed: PyPI unreachable. Rebuild the codespace in another region."; exit 1; }
uv run --no-sync python scripts/download_signatures.py
