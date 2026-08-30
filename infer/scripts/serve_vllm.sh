#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_DIR="$(realpath "${1:?usage: $0 MODEL_DIR}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${INFER_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${INFER_DIR}/.venv/bin/python"
  else
    PYTHON_BIN=python
  fi
fi
if [[ -z "${VLLM_BIN:-}" ]]; then
  if [[ -x "${INFER_DIR}/.venv/bin/vllm" ]]; then
    VLLM_BIN="${INFER_DIR}/.venv/bin/vllm"
  else
    VLLM_BIN=vllm
  fi
fi
VLLM_EXEC="$(command -v "${VLLM_BIN}")"
export PATH="$(dirname "${VLLM_EXEC}"):${PATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES}"
[[ "${#GPUS[@]}" -eq 8 ]] || { echo "DP=8 requires eight visible GPUs" >&2; exit 2; }
[[ -s "${MODEL_DIR}/config.json" ]] || { echo "invalid model directory: ${MODEL_DIR}" >&2; exit 2; }

MODEL_TYPE="${MODEL_TYPE:-$(${PYTHON_BIN} - "${MODEL_DIR}/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
print(config.get("model_type") or config.get("text_config", {}).get("model_type") or "")
PY
)}"

MODEL_ARGS=()
case "${MODEL_TYPE}" in
  gemma4)
    MODEL_ARGS+=(--tool-call-parser gemma4 --reasoning-parser gemma4)
    ;;
  *)
    MODEL_ARGS+=(--tool-call-parser hermes --chat-template "${INFER_DIR}/scripts/qwen3.jinja")
    ;;
esac

if [[ -z "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  case "$(basename "${MODEL_DIR}")" in
    *8B*|*8b*) VLLM_MAX_MODEL_LEN=32768 ;;
    *14B*|*14b*) VLLM_MAX_MODEL_LEN=40960 ;;
    *) VLLM_MAX_MODEL_LEN=40960 ;;
  esac
fi

exec "${VLLM_EXEC}" serve "${MODEL_DIR}" \
  --host "${VLLM_HOST:-127.0.0.1}" --port "${VLLM_PORT:-8080}" \
  --tensor-parallel-size 1 --data-parallel-size 8 \
  --dtype bfloat16 --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --served-model-name ContextPilot \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --enable-auto-tool-choice "${MODEL_ARGS[@]}"
