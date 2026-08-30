#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ES_VERSION="${ES_VERSION:-7.17.29}"
RUNTIME_DIR="${CONTEXT_PILOT_RUNTIME_DIR:-${INFER_DIR}/.runtime}"
ES_HOME="${ES_HOME:-${RUNTIME_DIR}/elasticsearch-${ES_VERSION}}"
ES_DATA_DIR="${ES_DATA_DIR:-${RUNTIME_DIR}/elasticsearch-data}"
ES_PID_FILE="${ES_PID_FILE:-${ES_DATA_DIR}/elasticsearch.pid}"
ES_PORT="${ES_PORT:-9200}"

if curl -s "http://localhost:${ES_PORT}" >/dev/null 2>&1; then
    echo "[INFO] Elasticsearch is already running on port ${ES_PORT}"
    curl -s "http://localhost:${ES_PORT}"
    exit 0
fi

if [[ ! -x "${ES_HOME}/bin/elasticsearch" ]]; then
    echo "[ERROR] Elasticsearch ${ES_VERSION} is not installed at ${ES_HOME}." >&2
    echo "Run: bash ${INFER_DIR}/scripts/setup_environment.sh" >&2
    exit 2
fi

CURRENT_MAP_COUNT=$(cat /proc/sys/vm/max_map_count)
if [[ "$CURRENT_MAP_COUNT" -lt 262144 ]]; then
    if [[ "$(id -u)" -eq 0 ]]; then
        echo "[INFO] Setting vm.max_map_count from ${CURRENT_MAP_COUNT} to 262144"
        sysctl -w vm.max_map_count=262144
    else
        echo "[ERROR] vm.max_map_count=${CURRENT_MAP_COUNT}; Elasticsearch requires at least 262144." >&2
        echo "Run as root: sysctl -w vm.max_map_count=262144" >&2
        exit 2
    fi
else
    echo "[INFO] vm.max_map_count is already $CURRENT_MAP_COUNT (>= 262144)"
fi

mkdir -p "${ES_DATA_DIR}/data" "${ES_DATA_DIR}/logs" "$(dirname "${ES_PID_FILE}")"

echo "[INFO] Starting Elasticsearch from $ES_HOME ..."
ES_ARGS=(
    -E"http.port=${ES_PORT}"
    -E"path.data=${ES_DATA_DIR}/data"
    -E"path.logs=${ES_DATA_DIR}/logs"
    -d -p "${ES_PID_FILE}"
)
if [[ "$(id -u)" -eq 0 ]]; then
    ES_USER="${ES_USER:-contextpilot-es}"
    if ! id "${ES_USER}" >/dev/null 2>&1; then
        useradd -m -s /bin/bash "${ES_USER}"
    fi
    chown -R "${ES_USER}:${ES_USER}" "${ES_HOME}" "${ES_DATA_DIR}"
    runuser -u "${ES_USER}" -- env HOME="$(getent passwd "${ES_USER}" | cut -d: -f6)" \
        "${ES_HOME}/bin/elasticsearch" "${ES_ARGS[@]}"
else
    "${ES_HOME}/bin/elasticsearch" "${ES_ARGS[@]}"
fi

echo "[INFO] Waiting for Elasticsearch to start on port ${ES_PORT} ..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:${ES_PORT}" >/dev/null 2>&1; then
        echo "[OK] Elasticsearch is up and running!"
        curl -s "http://localhost:${ES_PORT}"
        exit 0
    fi
    sleep 2
done

echo "[ERROR] Elasticsearch did not start within 60 seconds"
exit 1
