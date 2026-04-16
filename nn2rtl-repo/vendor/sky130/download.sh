#!/usr/bin/env bash
# Download the Sky130 standard cell liberty file used by run_yosys.
# Safe to re-run; verifies SHA-256 before overwriting.
set -euo pipefail

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_FILE="${DEST_DIR}/sky130_fd_sc_hd__tt_025C_1v80.lib"
URL="https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/raw/master/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
EXPECTED_SHA256="ec0e1067a35c8bf20b11e58d1e8ac53326067e4dac84a125cc1b917a3518d0d9"

if [ -f "${DEST_FILE}" ]; then
  actual="$(sha256sum "${DEST_FILE}" | awk '{print $1}')"
  if [ "${actual}" = "${EXPECTED_SHA256}" ]; then
    echo "Sky130 library already present and SHA-256 matches."
    exit 0
  fi
  echo "Existing file has wrong SHA-256 (${actual}); re-downloading."
fi

echo "Downloading Sky130 library..."
curl -fsSL "${URL}" -o "${DEST_FILE}.tmp"
actual="$(sha256sum "${DEST_FILE}.tmp" | awk '{print $1}')"
if [ "${actual}" != "${EXPECTED_SHA256}" ]; then
  rm -f "${DEST_FILE}.tmp"
  echo "SHA-256 mismatch: got ${actual}, expected ${EXPECTED_SHA256}" >&2
  exit 1
fi
mv "${DEST_FILE}.tmp" "${DEST_FILE}"
echo "Sky130 library installed at ${DEST_FILE}"
