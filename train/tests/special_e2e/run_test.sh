#!/bin/bash
set -xeuo pipefail

CONFIG_NAME="$1"
ENGINE="${2:-vllm}"


python3 -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" "$@" 