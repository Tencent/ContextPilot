#!/usr/bin/env bash

set -xeuo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading model to ${MODEL_PATH}..."
else
    echo "Model directory ${MODEL_PATH} already exists, skip downloading."
fi


BATCH_SIZE=16
EXP_NAME="qwen2.5_0.5b_grpo_lora"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/verl_grpo_example_gsm8k/${EXP_NAME}/global_step_1/actor/ \
    --target_dir checkpoints/verl_grpo_example_gsm8k/${EXP_NAME}/global_step_1/actor/hf

file_path="checkpoints/verl_grpo_example_gsm8k/${EXP_NAME}/global_step_1/actor/hf/lora_adapter/adapter_model.safetensors"

if [ ! -f "$file_path" ]; then
    echo "Error: File $file_path does not exist!"
    exit 1
fi

file_size=$(stat -c %s "$file_path")

min_size_mb=1
min_size=$((min_size_mb * 1024 * 1024))  # 1MB = 1048576 bytes

if [ "$file_size" -lt "$min_size" ]; then
    echo "Error: File $file_path is too small! Current size: $((file_size/1024))KB, Required: ${min_size_mb}MB"
    exit 1
fi

echo "Check passed: File exists and size is $(($file_size/1024/1024))MB"
exit 0
