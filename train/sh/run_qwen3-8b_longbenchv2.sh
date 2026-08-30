#!/usr/bin/env bash

set -euo pipefail

ulimit -n 65535

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_DIR=$(cd "$PROJECT_DIR/.." && pwd)

CONTEXTPILOT_VENV_DIR=${CONTEXTPILOT_VENV_DIR:-$PROJECT_DIR/.venv/contextpilot-agent-rl}
if [[ -x "$CONTEXTPILOT_VENV_DIR/bin/python" ]]; then
    export VIRTUAL_ENV=$CONTEXTPILOT_VENV_DIR
    export PATH=$CONTEXTPILOT_VENV_DIR/bin:$PATH
    hash -r
fi

ENV_FILE=${ENV_FILE:-$REPO_DIR/.env}
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi
CONFIG_ROOT=$PROJECT_DIR/examples/sglang_multiturn/config
TOOLS_CONFIG=$CONFIG_ROOT/tool_config/statelm_tool_optimized_config.yaml

DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data/rl_data/longbench_v2_transformed_rl}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/data/train-00000-of-00001.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/data/val-00000-of-00001.parquet}

WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-contextpilot}
DATE=$(date +%Y%m%d)
MODEL_SIZE=${MODEL_SIZE:-8B}
MODEL_TAG=$(printf '%s' "$MODEL_SIZE" | tr '[:upper:]' '[:lower:]')
WANDB_EXPERIMENT_NAME=${WANDB_EXPERIMENT_NAME:-"contextpilot-${MODEL_TAG}-rl-lbv2-${DATE}"}

MODEL_PATH=${MODEL_PATH:-}
SAVE_DIR=${SAVE_DIR:-$PROJECT_DIR/runs/contextpilot_longbenchv2/checkpoints}
MONITOR_DIR=${MONITOR_DIR:-$PROJECT_DIR/runs/contextpilot_longbenchv2/monitor}

LOG_TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$MONITOR_DIR/valid_logs/${WANDB_EXPERIMENT_NAME}_${LOG_TS}.log
TRAJECTORIES_DIR=$MONITOR_DIR/trajectories/${WANDB_EXPERIMENT_NAME}_${LOG_TS}
VAL_DATA_DIR=$MONITOR_DIR/validation_data/${WANDB_EXPERIMENT_NAME}

mkdir -p "$SAVE_DIR" "$VAL_DATA_DIR" "$(dirname "$LOG_FILE")" "$TRAJECTORIES_DIR"
echo "Project dir: $PROJECT_DIR"
echo "Logging to: $LOG_FILE"
echo "Trajectories directory: $TRAJECTORIES_DIR"
echo "Validation data directory: $VAL_DATA_DIR"

NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
N_ROLLOUT=${N_ROLLOUT:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-128}
MAX_SAMPLES_PER_TRAJECTORY=${MAX_SAMPLES_PER_TRAJECTORY:-8}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-100}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-28672}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-12288}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-40960}
SINGLE_TURN_MAX_TOKENS=${SINGLE_TURN_MAX_TOKENS:-2048}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.8}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:-20}
ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-false}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-40960}
CONTEXTPILOT_SNAPSHOT_BUDGET=${CONTEXTPILOT_SNAPSHOT_BUDGET:-128}
CONTEXTPILOT_CONTEXT_WEIGHT=${CONTEXTPILOT_CONTEXT_WEIGHT:-1.0}
CONTEXTPILOT_ENTROPY_WEIGHT=${CONTEXTPILOT_ENTROPY_WEIGHT:-1.0}
CONTEXTPILOT_ENTROPY_TOP_K=${CONTEXTPILOT_ENTROPY_TOP_K:-10}
CONTEXTPILOT_ENTROPY_TOKEN_WINDOW=${CONTEXTPILOT_ENTROPY_TOKEN_WINDOW:-20}
CONTEXTPILOT_MAX_CONCURRENT_BRANCHES=${CONTEXTPILOT_MAX_CONCURRENT_BRANCHES:-64}
CONTEXTPILOT_BUDGET_TOKEN_LIMIT=${CONTEXTPILOT_BUDGET_TOKEN_LIMIT:-26000}
CONTEXTPILOT_AUTO_DELETE_TOKEN_LIMIT=${CONTEXTPILOT_AUTO_DELETE_TOKEN_LIMIT:-24000}
AGENT_STICKY_MAX_LOAD_SKEW=${AGENT_STICKY_MAX_LOAD_SKEW:-2}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true}
ROLLOUT_SEED=${ROLLOUT_SEED:-20260810}
CONTEXTPILOT_SYSTEM_PROMPT_FILE=${CONTEXTPILOT_SYSTEM_PROMPT_FILE:-$PROJECT_DIR/configs/contextpilot_system_prompt.txt}
if (( ROLLOUT_PROMPT_LENGTH + ROLLOUT_RESPONSE_LENGTH != ROLLOUT_MAX_MODEL_LEN )); then
    echo "ROLLOUT_PROMPT_LENGTH + ROLLOUT_RESPONSE_LENGTH must equal ROLLOUT_MAX_MODEL_LEN." >&2
    exit 1
fi
if (( MAX_TOKEN_LEN_PER_GPU < ROLLOUT_MAX_MODEL_LEN )); then
    echo "MAX_TOKEN_LEN_PER_GPU must be at least ROLLOUT_MAX_MODEL_LEN." >&2
    exit 1
fi
if (( CONTEXTPILOT_AUTO_DELETE_TOKEN_LIMIT >= CONTEXTPILOT_BUDGET_TOKEN_LIMIT )); then
    echo "CONTEXTPILOT_AUTO_DELETE_TOKEN_LIMIT must be below CONTEXTPILOT_BUDGET_TOKEN_LIMIT." >&2
    exit 1
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    DEFAULT_TRAINER_LOGGER='["console", "wandb"]'
else
    DEFAULT_TRAINER_LOGGER='["console"]'
fi
TRAINER_LOGGER=${TRAINER_LOGGER:-$DEFAULT_TRAINER_LOGGER}
SAVE_FREQ=${SAVE_FREQ:-8}
TEST_FREQ=${TEST_FREQ:-8}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
CHECKPOINT_SAVE_CONTENTS=${CHECKPOINT_SAVE_CONTENTS:-'[model,extra]'}
CHECKPOINT_LOAD_CONTENTS=${CHECKPOINT_LOAD_CONTENTS:-$CHECKPOINT_SAVE_CONTENTS}

DUMP_TRAJECTORIES=${DUMP_TRAJECTORIES:-false}
DUMP_TRAJECTORIES_DIR=${DUMP_TRAJECTORIES_DIR:-"$TRAJECTORIES_DIR"}
DUMP_TRAJECTORIES_FREQ=${DUMP_TRAJECTORIES_FREQ:-80}

export OPENAI_API_KEY=${OPENAI_API_KEY:-}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-}
export OPENAI_MODEL=${OPENAI_MODEL:-}
export WANDB_API_KEY=${WANDB_API_KEY:-""}
export WANDB_MODE=${WANDB_MODE:-online}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-1}
DEFAULT_ROUTE_IFACE=$(ip route show default 2>/dev/null | awk 'NR == 1 {print $5}')
if [[ -z "${NCCL_SOCKET_IFNAME:-}" && -n "$DEFAULT_ROUTE_IFACE" ]]; then
    export NCCL_SOCKET_IFNAME=$DEFAULT_ROUTE_IFACE
fi
export LM_HARNESS_CACHE_PATH=${LM_HARNESS_CACHE_PATH:-cache}
export PYTHONUNBUFFERED=1
export CONTEXTPILOT_ROLLOUT_SEED=$ROLLOUT_SEED
export CONTEXTPILOT_SYSTEM_PROMPT_FILE
export VERL_AGENT_STICKY_MAX_LOAD_SKEW=$AGENT_STICKY_MAX_LOAD_SKEW

cd "$PROJECT_DIR"

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
    echo "LongBench-v2 parquet files were not found. Set TRAIN_FILE/VAL_FILE or prepare $DATA_DIR first." >&2
    exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
    echo "MODEL_PATH is required. Set it to a local model directory or Hugging Face checkpoint ID." >&2
    exit 1
fi

if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "OPENAI_API_KEY is required to score open-ended LongBench-v2 examples." >&2
    echo "Create $REPO_DIR/.env from $REPO_DIR/.env.example or set ENV_FILE." >&2
    exit 1
fi
if [[ -z "$OPENAI_BASE_URL" || -z "$OPENAI_MODEL" ]]; then
    echo "OPENAI_BASE_URL and OPENAI_MODEL are required for the open-ended judge." >&2
    echo "Set them in the local $REPO_DIR/.env file or through the environment." >&2
    exit 1
fi

if [[ ! -s "$CONTEXTPILOT_SYSTEM_PROMPT_FILE" ]]; then
    echo "ContextPilot system prompt was not found: $CONTEXTPILOT_SYSTEM_PROMPT_FILE" >&2
    exit 1
fi

if [[ "$ATTN_IMPLEMENTATION" == "flash_attention_2" ]]; then
    python3 -c 'import flash_attn; print(f"[OK] FlashAttention {flash_attn.__version__} is available")'
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_ROOT" \
        --config-name="statelm_tool_agent_grpo" \
        ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS} \
        actor_rollout_ref.rollout.name=vllm \
        algorithm.adv_estimator=contextpilot_grpo \
        data.train_batch_size=${TRAIN_BATCH_SIZE} \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT_LENGTH} \
        actor_rollout_ref.rollout.response_length=${ROLLOUT_RESPONSE_LENGTH} \
        +actor_rollout_ref.rollout.multi_turn.single_turn_max_tokens=${SINGLE_TURN_MAX_TOKENS} \
        +actor_rollout_ref.rollout.multi_turn.max_samples_per_trajectory=${MAX_SAMPLES_PER_TRAJECTORY} \
        +actor_rollout_ref.rollout.multi_turn.downsample_mode=random \
        +actor_rollout_ref.rollout.multi_turn.model_type=qwen3 \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS} \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=null \
        actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
        +actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION \
        actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
        actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
        actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
        actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_USE_DYNAMIC_BSZ} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.rollout.val_kwargs.top_k=${ROLLOUT_TOP_K} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${ROLLOUT_TOP_P} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${ROLLOUT_TEMPERATURE} \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
        actor_rollout_ref.rollout.n=${N_ROLLOUT} \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.default_local_dir=$SAVE_DIR/$WANDB_EXPERIMENT_NAME \
        trainer.project_name="$WANDB_PROJECT_NAME" \
        trainer.experiment_name="$WANDB_EXPERIMENT_NAME" \
        trainer.device=cuda \
        trainer.n_gpus_per_node=${GPUS_PER_NODE} \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.test_freq=${TEST_FREQ} \
        trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
        "actor_rollout_ref.actor.checkpoint.save_contents=${CHECKPOINT_SAVE_CONTENTS}" \
        "actor_rollout_ref.actor.checkpoint.load_contents=${CHECKPOINT_LOAD_CONTENTS}" \
        trainer.val_before_train=${VAL_BEFORE_TRAIN} \
        trainer.validation_data_dir="$VAL_DATA_DIR" \
        trainer.logger="$TRAINER_LOGGER" \
        custom_reward_function.path=verl/utils/reward_score/statelm_qa.py \
        custom_reward_function.name=compute_score \
        data.train_files="$TRAIN_FILE" \
        data.val_files="$VAL_FILE" \
        trainer.total_epochs=$TOTAL_EPOCHS \
        trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
        actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 \
        actor_rollout_ref.rollout.trace.token2text=False \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOLS_CONFIG" \
        actor_rollout_ref.rollout.free_cache_engine=True \
        +actor_rollout_ref.rollout.multi_turn.exceed_length_penalty=-0.5 \
        +actor_rollout_ref.rollout.multi_turn.dump_trajectories_enabled=$DUMP_TRAJECTORIES \
        +actor_rollout_ref.rollout.multi_turn.dump_trajectories_dir="$DUMP_TRAJECTORIES_DIR" \
        +actor_rollout_ref.rollout.multi_turn.dump_trajectories_freq=$DUMP_TRAJECTORIES_FREQ \
        +actor_rollout_ref.rollout.multi_turn.contextpilot.enable=True \
        +actor_rollout_ref.rollout.multi_turn.contextpilot.auto_delete_token_limit=${CONTEXTPILOT_AUTO_DELETE_TOKEN_LIMIT} \
        +actor_rollout_ref.rollout.multi_turn.contextpilot.budget_token_limit=${CONTEXTPILOT_BUDGET_TOKEN_LIMIT} \
        +actor_rollout_ref.rollout.multi_turn.contextpilot.partial_rollout.enable=True \
        +actor_rollout_ref.rollout.multi_turn.contextpilot.partial_rollout.snapshot_budget=${CONTEXTPILOT_SNAPSHOT_BUDGET} \
        +actor_rollout_ref.rollout.multi_turn.contextpilot.partial_rollout.context_weight=${CONTEXTPILOT_CONTEXT_WEIGHT} \
        +actor_rollout_ref.rollout.multi_turn.contextpilot.partial_rollout.entropy_weight=${CONTEXTPILOT_ENTROPY_WEIGHT} \
        +actor_rollout_ref.rollout.multi_turn.contextpilot.partial_rollout.entropy_top_k=${CONTEXTPILOT_ENTROPY_TOP_K} \
        +actor_rollout_ref.rollout.multi_turn.contextpilot.partial_rollout.entropy_token_window=${CONTEXTPILOT_ENTROPY_TOKEN_WINDOW} \
        +actor_rollout_ref.rollout.multi_turn.contextpilot.partial_rollout.max_concurrent_branches=${CONTEXTPILOT_MAX_CONCURRENT_BRANCHES} \
    2>&1 | tee -a "$LOG_FILE"
