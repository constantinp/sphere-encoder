#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPHERE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SPHERE_ROOT/../.." && pwd)"

IMG_DIR="${IMG_DIR:-$REPO_ROOT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SPHERE_ROOT/workspace/datasets/list}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
SEED="${SEED:-99}"

exec python -u "$REPO_ROOT/scripts/prepare_sphere_dataset.py" \
  --img_dir "$IMG_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --val_split "$VAL_SPLIT" \
  --seed "$SEED" \
  "$@"
