#!/usr/bin/env bash
set -Eeuo pipefail
export TASKS=novelqa
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_full_pipeline.sh" "$@"
