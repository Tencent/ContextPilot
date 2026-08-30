#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_DIR="$(realpath "${1:?usage: $0 MODEL_DIR [RUN_ID]}")"
RUN_ID="${2:-$(basename "${MODEL_DIR}")}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${INFER_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${INFER_DIR}/.venv/bin/python"
  else
    PYTHON_BIN=python
  fi
fi
LOG_DIR="${LOG_DIR:-${INFER_DIR}/results/logs/${RUN_ID}}"
TASKS="${TASKS:-infbench novelqa longmemeval bc_plus}"

export TOKENIZER_PATH="${MODEL_DIR}"
export RUN_ID
export N_PROC="${N_PROC:-64}"
export CONTEXTPILOT_DETERMINISTIC_SEED="${CONTEXTPILOT_DETERMINISTIC_SEED:-20260810}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES}"
[[ "${#GPUS[@]}" -eq 8 ]] || { echo "DP=8 requires eight visible GPUs" >&2; exit 2; }
[[ -s "${MODEL_DIR}/config.json" ]] || { echo "invalid model directory: ${MODEL_DIR}" >&2; exit 2; }
mkdir -p "${LOG_DIR}"

MODEL_TYPE="$(${PYTHON_BIN} - "${MODEL_DIR}/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
print(config.get("model_type") or config.get("text_config", {}).get("model_type") or "")
PY
)"

case "${MODEL_TYPE}" in
  gemma4)
    export OPENAI_FILE="${OPENAI_FILE:-${INFER_DIR}/configs/openai_endpoint_1x_gemma4.json}"
    ;;
  *)
    export OPENAI_FILE="${OPENAI_FILE:-${INFER_DIR}/configs/openai_endpoint_1x_nonthinking.json}"
    ;;
esac

bash "${INFER_DIR}/scripts/start_elasticsearch.sh" >"${LOG_DIR}/elasticsearch.log" 2>&1

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

setsid env MODEL_TYPE="${MODEL_TYPE}" bash "${SCRIPT_DIR}/serve_vllm.sh" "${MODEL_DIR}" \
  >"${LOG_DIR}/vllm.log" 2>&1 &
SERVER_PID="$!"

ready=false
for _ in $(seq 1 300); do
  if curl -fsS http://127.0.0.1:8080/v1/models >/dev/null; then ready=true; break; fi
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 100 "${LOG_DIR}/vllm.log"; exit 1; }
  sleep 2
done
[[ "${ready}" == true ]] || { echo "vLLM did not become ready" >&2; exit 1; }

: >"${LOG_DIR}/scores.txt"
for task in ${TASKS}; do
  bash "${INFER_DIR}/scripts/eval_task.sh" "${task}" 2>&1 | tee "${LOG_DIR}/${task}.log"
  sed -n 's/^SCORE_FILE=//p' "${LOG_DIR}/${task}.log" | tail -n 1 >>"${LOG_DIR}/scores.txt"
done

echo "Score manifest: ${LOG_DIR}/scores.txt"
