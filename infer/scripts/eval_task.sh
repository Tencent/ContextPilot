#!/usr/bin/env bash
set -Eeuo pipefail

TASK="${1:?usage: $0 TASK  # TASK=infbench|novelqa|longmemeval|bc_plus}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INFER_DIR="${PROJECT_ROOT}/infer"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CONTEXTPILOT_DETERMINISTIC_SEED="${CONTEXTPILOT_DETERMINISTIC_SEED:-20260810}"
export CONTEXTPILOT_SEED_MODE="${CONTEXTPILOT_SEED_MODE:-global}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${INFER_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${INFER_DIR}/.venv/bin/python"
  else
    PYTHON_BIN=python
  fi
fi
TOKENIZER_PATH="${TOKENIZER_PATH:?set TOKENIZER_PATH to the evaluated model}"
RUN_ID="${RUN_ID:-ContextPilot}"
OPENAI_FILE="${OPENAI_FILE:-${INFER_DIR}/configs/openai_endpoint_1x_nonthinking.json}"
SAVE_DIR="${SAVE_DIR:-${INFER_DIR}/results}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-${INFER_DIR}/tools/context-shaper_tools.json}"
SYSTEM_PROMPT_NAME="${SYSTEM_PROMPT_NAME:-FSM_PLAN_BM25_MC_PROMPT}"

N_PROC="${N_PROC:-64}"
MAX_ITEMS="${MAX_ITEMS:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-4000}"
OVERLAP="${OVERLAP:-0}"
BOUNDARY_BACKTRACK="${BOUNDARY_BACKTRACK:-400}"
MAX_TOKEN_WINDOW="${MAX_TOKEN_WINDOW:-}"
MAX_CONTEXT="${MAX_CONTEXT:-32000}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2048}"
MAX_SEARCH_CALLS="${MAX_SEARCH_CALLS:-15}"
MAX_CONTEXT_EXP="${MAX_CONTEXT_EXP:-32000}"
SCHEDULING="${SCHEDULING:-queue}"
DEFER_JUDGE="${DEFER_JUDGE:-false}"

TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"

if [[ -z "${MAX_TOKEN_WINDOW}" ]]; then
  case "$(basename "${TOKENIZER_PATH}")" in
    *8B*|*8b*) MAX_TOKEN_WINDOW=29800 ;;
    *) MAX_TOKEN_WINDOW=30000 ;;
  esac
fi

case "${TASK}" in
  infbench)
    DATASET_NAME="${INFBENCH_DATASET:-lindsay21/InfiniteBench}"
    DATASET_SPLIT="longbook_choice_eng"
    PREFIX="infinitebench_longbook_choice_eng"
    ITEM_META=""
    MAX_TURNS="${MAX_TURNS:-50}"
    ;;
  novelqa)
    NOVELQA_LOCAL_DATA="${INFER_DIR}/data/NovelQA/CopyrightProtected.jsonl"
    if [[ -n "${NOVELQA_DATA:-}" ]]; then
      DATASET_NAME="${NOVELQA_DATA}"
    elif [[ -f "${NOVELQA_LOCAL_DATA}" ]]; then
      DATASET_NAME="${NOVELQA_LOCAL_DATA}"
    else
      cat >&2 <<EOF
NovelQA requires the prepared CopyrightProtected JSONL with gold answers and
access to the corresponding copyrighted book texts. The official question-only
gated file is not sufficient. This private evaluation bundle is not distributed
with the repository; set NOVELQA_DATA and NOVELQA_CONTEXT_ROOT after obtaining
the required access from https://huggingface.co/datasets/NovelQA/NovelQA.
EOF
      exit 2
    fi
    [[ -f "${DATASET_NAME}" ]] || {
      echo "NovelQA data file does not exist: ${DATASET_NAME}" >&2
      exit 2
    }
    "${PYTHON_BIN}" - "${DATASET_NAME}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
valid = all(
    row.get("question")
    and row.get("options")
    and (row.get("gold") or row.get("answer"))
    and (row.get("context") or row.get("context_path"))
    for row in rows
)
if len(rows) != 757 or not valid:
    raise SystemExit(
        "NovelQA data validation failed: expected 757 prepared rows with "
        "question, options, gold answer, and context/context_path. The "
        "official question-only gated file is not sufficient."
    )
PY
    DATASET_SPLIT="local"
    PREFIX="novelqa"
    ITEM_META="${INFER_DIR}/src/hf_process_fns.py:novelqa_i2meta"
    MAX_TURNS="${MAX_TURNS:-50}"
    ;;
  longmemeval)
    DATASET_NAME="${LONGMEMEVAL_DATA:?set LONGMEMEVAL_DATA to the benchmark JSON/JSONL}"
    DATASET_SPLIT="local"
    PREFIX="longmemevals"
    ITEM_META="${INFER_DIR}/src/hf_process_fns.py:longmemevals_i2meta"
    MAX_TURNS="${MAX_TURNS:-60}"
    ;;
  bc_plus)
    DATASET_NAME="${BROWSECOMP_PLUS_DATA:?set BROWSECOMP_PLUS_DATA to the decrypted benchmark JSONL}"
    DATASET_SPLIT="local"
    PREFIX="bc_plus"
    ITEM_META="${INFER_DIR}/src/hf_process_fns.py:bc_plus_i2meta"
    MAX_TURNS="${MAX_TURNS:-60}"
    ;;
  *) echo "unknown task: ${TASK}" >&2; exit 2 ;;
esac

STAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${SAVE_DIR}/${TASK}/${RUN_ID}/${STAMP}"
OUTPUT_FP="${OUT_DIR}/generations.jsonl"
SCORE_FP="${OUT_DIR}/score.txt"
ANNOTATED_FP="${OUT_DIR}/generations_scored.jsonl"
mkdir -p "${OUT_DIR}/trajectories" "${OUT_DIR}/artifacts"

ARGS=(
  -m infer.src.hf_test_runner eval_hfds_contextpilot
  --vllm_cfg "${OPENAI_FILE}"
  --temperature "${TEMP}" --top_p "${TOP_P}" --top_k "${TOP_K}"
  --max_turns_exp "${MAX_TURNS}" --max_turns_to_fail "$((MAX_TURNS + 10))"
  --max_context_exp "${MAX_CONTEXT_EXP}" --max_context "${MAX_CONTEXT}"
  --max_output_tokens "${MAX_OUTPUT_TOKENS}"
  --tool_config_path "${TOOL_CONFIG_PATH}"
  --system_prompt_name "${SYSTEM_PROMPT_NAME}"
  --dataset_name "${DATASET_NAME}" --dataset_split "${DATASET_SPLIT}"
  --item_to_question "${INFER_DIR}/src/hf_process_fns.py:${PREFIX}_i2q"
  --item_to_context "${INFER_DIR}/src/hf_process_fns.py:${PREFIX}_i2c"
  --item_to_answer "${INFER_DIR}/src/hf_process_fns.py:${PREFIX}_i2a"
  --trajectory_dir "${OUT_DIR}/trajectories" --result_dir "${OUT_DIR}/artifacts"
  --output_fp "${OUTPUT_FP}" --tokenizer_path "${TOKENIZER_PATH}"
  --n_proc "${N_PROC}" --scheduling "${SCHEDULING}"
  --resume False --version fsm
  --allow_text_tool_call_fallback True
  --max_search_calls "${MAX_SEARCH_CALLS}" --max_token_window "${MAX_TOKEN_WINDOW}"
  --chunk_size "${CHUNK_SIZE}" --overlap "${OVERLAP}"
  --boundary_backtrack "${BOUNDARY_BACKTRACK}"
  --highlight_fragment_size "${HIGHLIGHT_FRAGMENT_SIZE:-512}"
  --highlight_num_fragments "${HIGHLIGHT_NUM_FRAGMENTS:-2}"
  --highlight_no_match_size "${HIGHLIGHT_NO_MATCH_SIZE:-200}"
  --search_engine_max_results "${SEARCH_ENGINE_MAX_RESULTS:-10}"
)
[[ -z "${ITEM_META}" ]] || ARGS+=(--item_to_meta "${ITEM_META}")
[[ "${MAX_ITEMS}" == 0 ]] || ARGS+=(--max_items "${MAX_ITEMS}")
if [[ -n "${FAILED_SAMPLES_FILE:-}" ]]; then
  ARGS+=(--failed_samples_file "${FAILED_SAMPLES_FILE}")
  ARGS+=(--retry_with_hint "${RETRY_WITH_HINT:-False}")
fi
"${PYTHON_BIN}" "${ARGS[@]}"

case "${TASK}" in
  infbench|novelqa)
    TASK_NAME="longbook_choice_eng"
    [[ "${TASK}" != novelqa ]] || TASK_NAME="CopyrightProtected"
    "${PYTHON_BIN}" "${INFER_DIR}/src/hf_score_fns.py" evaluate_choice_file \
      --file_path "${OUTPUT_FP}" \
      --model_name "${RUN_ID}" --task_name "${TASK_NAME}" \
      --label_key correct_answer --pred_key final_answer \
      --results_output "${SCORE_FP}"
    ;;
  longmemeval)
    if [[ "${DEFER_JUDGE}" == true ]]; then
      printf 'Score: DEFERRED\n--------------------\n' >"${SCORE_FP}"
    else
      : "${JUDGE_OPENAI_FILE:?set JUDGE_OPENAI_FILE to the judge endpoint config}"
      "${PYTHON_BIN}" "${INFER_DIR}/src/hf_score_fns.py" annotate_longmemeval_file \
        --preds_path "${OUTPUT_FP}" --ref_path "${DATASET_NAME}" \
        --output_path "${ANNOTATED_FP}" --results_output "${SCORE_FP}" \
        --pred_key final_answer --judge_model endpoint-config \
        --endpoint_config "${JUDGE_OPENAI_FILE}" --max_judge_tokens 4096 \
        --num_workers "${JUDGE_NUM_WORKERS:-32}"
    fi
    ;;
  bc_plus)
    if [[ "${DEFER_JUDGE}" == true ]]; then
      printf 'Score: DEFERRED\n--------------------\n' >"${SCORE_FP}"
    else
      : "${JUDGE_OPENAI_FILE:?set JUDGE_OPENAI_FILE to the judge endpoint config}"
      "${PYTHON_BIN}" "${INFER_DIR}/src/hf_score_fns.py" annotate_browsecomp_plus_file \
        --preds_path "${OUTPUT_FP}" --ref_path "${DATASET_NAME}" \
        --output_path "${ANNOTATED_FP}" --results_output "${SCORE_FP}" \
        --endpoint_config "${JUDGE_OPENAI_FILE}" --num_workers "${JUDGE_NUM_WORKERS:-32}"
    fi
    ;;
esac

echo "OUTPUT_FILE=${OUTPUT_FP}"
echo "SCORE_FILE=${SCORE_FP}"
echo "JUDGE_DEFERRED=${DEFER_JUDGE}"
