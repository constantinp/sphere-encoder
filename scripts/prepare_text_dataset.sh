#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPHERE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMG_DIR="${IMG_DIR:-$SPHERE_ROOT/workspace/raw_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SPHERE_ROOT/workspace/datasets/list}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
SEED="${SEED:-99}"

exec python -u "$SPHERE_ROOT/prepare_dataset.py" \
  --img_dir "$IMG_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --val_split "$VAL_SPLIT" \
  --seed "$SEED" \
  "$@"
