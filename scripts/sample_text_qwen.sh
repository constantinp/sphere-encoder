#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPHERE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

JOB_DIR="${JOB_DIR:-}"
NUM_GEN_SAMPLES="${NUM_GEN_SAMPLES:-16}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-8}"
PROMPT="${PROMPT:-portrait photo of a woman, soft natural light, detailed skin, 35mm}"

if [[ -z "$JOB_DIR" ]]; then
  echo "Set JOB_DIR to the experiment directory name under workspace/experiments." >&2
  exit 1
fi

cd "$SPHERE_ROOT"

if [[ -n "${PROMPTS_FILE:-}" ]]; then
  exec ./run.sh sample.py \
    --job_dir "$JOB_DIR" \
    --num_gen_samples "$NUM_GEN_SAMPLES" \
    --batch_size_per_rank "$BATCH_SIZE_PER_RANK" \
    --prompts_file "$PROMPTS_FILE" \
    "$@"
fi

exec ./run.sh sample.py \
  --job_dir "$JOB_DIR" \
  --num_gen_samples "$NUM_GEN_SAMPLES" \
  --batch_size_per_rank "$BATCH_SIZE_PER_RANK" \
  --prompt "$PROMPT" \
  "$@"
