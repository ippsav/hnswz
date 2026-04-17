#!/usr/bin/env bash
# Fetch SIFT ANN-benchmark corpora into `bench/data/<name>`.
#
# Supports:
#   siftsmall   - 10k base, 100 queries, dim=128  (~9 MB, for smoke tests)
#   sift1m      - 1M base, 10k queries, dim=128   (~500 MB download, headline result)
#
# Usage:
#   bench/download_sift.sh siftsmall
#   bench/download_sift.sh sift1m
#   bench/download_sift.sh all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${SCRIPT_DIR}/data"

fetch() {
    local name="$1"; local url="$2"; local tar_name="$3"; local inner_dir="$4"

    local dest="${DATA_ROOT}/${name}"
    if [[ -f "${dest}"/*base.fvecs ]] 2>/dev/null; then
        # shellcheck disable=SC2144
        echo "[${name}] already present at ${dest}, skipping"
        return 0
    fi

    mkdir -p "${DATA_ROOT}"
    local tarball="${DATA_ROOT}/${tar_name}"

    if [[ ! -f "${tarball}" ]]; then
        echo "[${name}] downloading ${url}"
        curl -fSL -o "${tarball}.part" "${url}"
        mv "${tarball}.part" "${tarball}"
    else
        echo "[${name}] tarball already downloaded: ${tarball}"
    fi

    echo "[${name}] extracting"
    tar -xzf "${tarball}" -C "${DATA_ROOT}"

    # Normalize to `${DATA_ROOT}/${name}/` regardless of inner dir name.
    if [[ "${inner_dir}" != "${name}" ]]; then
        rm -rf "${dest}"
        mv "${DATA_ROOT}/${inner_dir}" "${dest}"
    fi

    # List contents so the user sees what shipped.
    echo "[${name}] contents:"
    ls -lh "${dest}"
}

main() {
    local which="${1:-}"
    case "${which}" in
        siftsmall)
            fetch siftsmall \
                "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz" \
                "siftsmall.tar.gz" "siftsmall"
            ;;
        sift1m)
            fetch sift1m \
                "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" \
                "sift1m.tar.gz" "sift"
            ;;
        all)
            fetch siftsmall \
                "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz" \
                "siftsmall.tar.gz" "siftsmall"
            fetch sift1m \
                "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" \
                "sift1m.tar.gz" "sift"
            ;;
        *)
            cat >&2 <<EOF
usage: $0 {siftsmall|sift1m|all}

  siftsmall  ~9 MB   10k base, 100 queries, dim 128 (smoke)
  sift1m     ~500 MB 1M base, 10k queries, dim 128 (headline)
EOF
            exit 2
            ;;
    esac
}

main "$@"
