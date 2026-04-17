#!/usr/bin/env bash
# End-to-end bench driver: build hnswz (ReleaseFast), ensure Python deps,
# ensure dataset is present, run both benchmarks, print the comparison.
#
# Usage:
#   bench/run.sh [siftsmall|sift1m] [hnswz_extra_args...]
#
# Example:
#   bench/run.sh siftsmall
#   bench/run.sh sift1m --num-vectors 200000 --num-queries 2000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_NAME="${1:-siftsmall}"
shift || true

DATASET_DIR="${SCRIPT_DIR}/data/${DATASET_NAME}"
OUT_DIR="${SCRIPT_DIR}/results"
mkdir -p "${OUT_DIR}"

# 1. dataset
if [[ ! -d "${DATASET_DIR}" ]]; then
    echo ">> dataset missing, downloading ${DATASET_NAME}"
    "${SCRIPT_DIR}/download_sift.sh" "${DATASET_NAME}"
fi

# 2. build hnswz in ReleaseFast
echo ">> building hnswz (ReleaseFast)"
(cd "${REPO}" && zig build -Doptimize=ReleaseFast) >/dev/null

# 3. ensure python deps. Homebrew python3 is PEP-668-locked, so we isolate
#    the deps in a venv under bench/.venv.
VENV_DIR="${SCRIPT_DIR}/.venv"
VENV_PY="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PY}" ]] || ! "${VENV_PY}" -c 'import hnswlib, numpy' 2>/dev/null; then
    echo ">> creating venv and installing hnswlib + numpy"
    python3 -m venv "${VENV_DIR}"
    "${VENV_PY}" -m pip install --quiet --upgrade pip hnswlib numpy
fi
PY="${VENV_PY}"

# 4. param set — shared between both. Tune via env if you want.
: "${M:=16}"
: "${EF_CONS:=200}"
: "${EF_SEARCH:=100}"
: "${TOP_K:=10}"
: "${WARMUP:=50}"
: "${SEED:=42}"

HNSWZ_JSON="${OUT_DIR}/${DATASET_NAME}_hnswz.json"
HNSWLIB_JSON="${OUT_DIR}/${DATASET_NAME}_hnswlib.json"

# Default num-vectors / num-queries: let each side consume the whole
# dataset unless the caller overrides. siftsmall has 10k/100; sift1m has
# 1M/10k. Passing 0 to the Python side = "use all"; the Zig side needs an
# explicit cap or it uses whatever the CLI default is (10_000 base,
# 1_000 query). So we introspect the file size once and set both.
#
# We do this in Python via a tiny inline snippet to avoid reimplementing
# fvecs parsing in bash.
read -r N Q DIM < <("${PY}" - <<'EOF' "${DATASET_DIR}"
import os, sys
from pathlib import Path
d = Path(sys.argv[1])
def one(suffix):
    for p in sorted(d.iterdir()):
        if p.name.endswith(suffix): return p
    raise SystemExit(f"missing *{suffix} in {d}")
def count(path):
    import numpy as np
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    stride = dim + 1
    return raw.size // stride, dim
base_n, dim = count(one("base.fvecs"))
q_n, _ = count(one("query.fvecs"))
print(base_n, q_n, dim)
EOF
)

echo ">> dataset ${DATASET_NAME}: n=${N} q=${Q} dim=${DIM}"

# 5. run hnswz
echo ">> running hnswz benchmark (ReleaseFast, in-process)"
(cd "${REPO}" && ./zig-out/bin/hnswz benchmark \
    --dataset "${DATASET_DIR}" \
    --num-vectors "${N}" \
    --num-queries "${Q}" \
    --ef-construction "${EF_CONS}" \
    --ef-search "${EF_SEARCH}" \
    --top-k "${TOP_K}" \
    --warmup "${WARMUP}" \
    --seed "${SEED}" \
    --validate --json "$@") > "${HNSWZ_JSON}"

# 6. run hnswlib
echo ">> running hnswlib benchmark (single-threaded)"
"${PY}" "${SCRIPT_DIR}/hnswlib_bench.py" \
    --dataset "${DATASET_DIR}" \
    --num-vectors "${N}" \
    --num-queries "${Q}" \
    --M "${M}" \
    --ef-construction "${EF_CONS}" \
    --ef-search "${EF_SEARCH}" \
    --top-k "${TOP_K}" \
    --warmup "${WARMUP}" \
    --seed "${SEED}" \
    --threads 1 \
    --validate --json > "${HNSWLIB_JSON}"

# 7. compare
echo
"${PY}" "${SCRIPT_DIR}/compare.py" "${HNSWZ_JSON}" "${HNSWLIB_JSON}"
echo
echo "raw JSONs: ${HNSWZ_JSON}  |  ${HNSWLIB_JSON}"
