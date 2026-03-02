#!/usr/bin/env bash
set -euo pipefail

BIN_PATH="${1:-./build/nms_golden_test}"
OUT_DIR="${2:-tests/golden}"

mkdir -p "${OUT_DIR}"

"${BIN_PATH}" --dump nms_case1 > "${OUT_DIR}/nms_case1.json"
"${BIN_PATH}" --dump nms_case2 > "${OUT_DIR}/nms_case2.json"

echo "Golden files generated under ${OUT_DIR}"
