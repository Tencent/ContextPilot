#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-${INFER_DIR}/.venv}"
ES_VERSION="${ES_VERSION:-7.17.29}"
RUNTIME_DIR="${CONTEXT_PILOT_RUNTIME_DIR:-${INFER_DIR}/.runtime}"
ES_HOME="${ES_HOME:-${RUNTIME_DIR}/elasticsearch-${ES_VERSION}}"

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
    echo "[ERROR] The locked environment currently supports Linux x86_64 only." >&2
    exit 2
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "[ERROR] Python 3.12.13 is required; ${PYTHON_BIN} was not found." >&2
    exit 2
}
"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:3] != (3, 12, 13):
    raise SystemExit(f"Python 3.12.13 is required, found {sys.version.split()[0]}")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade "pip==26.1.2"

REQUIREMENTS="${INFER_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS}"
"${VENV_DIR}/bin/python" -m pip check

if [[ "${SKIP_ELASTICSEARCH_DOWNLOAD:-0}" != "1" && ! -x "${ES_HOME}/bin/elasticsearch" ]]; then
    command -v curl >/dev/null 2>&1 || {
        echo "[ERROR] curl is required to download Elasticsearch." >&2
        exit 2
    }
    command -v tar >/dev/null 2>&1 || {
        echo "[ERROR] tar is required to unpack Elasticsearch." >&2
        exit 2
    }
    command -v sha512sum >/dev/null 2>&1 || {
        echo "[ERROR] sha512sum is required to verify Elasticsearch." >&2
        exit 2
    }
    mkdir -p "${RUNTIME_DIR}"
    ARCHIVE="elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"
    URL="https://artifacts.elastic.co/downloads/elasticsearch/${ARCHIVE}"
    curl --fail --location --retry 3 --output "${RUNTIME_DIR}/${ARCHIVE}" "${URL}"
    curl --fail --location --retry 3 --output "${RUNTIME_DIR}/${ARCHIVE}.sha512" "${URL}.sha512"
    (
        cd "${RUNTIME_DIR}"
        sha512sum --check "${ARCHIVE}.sha512"
        tar -xzf "${ARCHIVE}"
    )
fi

"${VENV_DIR}/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys

print(f"[OK] Python: {sys.version.split()[0]}")
for package in ("torch", "vllm", "transformers", "datasets", "openai"):
    print(f"[OK] {package}: {metadata.version(package)}")
PY
echo "[OK] Virtual environment: ${VENV_DIR}"
if [[ -x "${ES_HOME}/bin/elasticsearch" ]]; then
    echo "[OK] Elasticsearch: ${ES_HOME}"
else
    echo "[SKIP] Elasticsearch download was disabled; expected path: ${ES_HOME}"
fi
echo "Activate with: source ${VENV_DIR}/bin/activate"
