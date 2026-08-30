#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MODEL_SIZE=14B
export MODEL_PATH=${MODEL_PATH:-}
exec bash "$SCRIPT_DIR/run_qwen3-8b_longbenchv2.sh"
