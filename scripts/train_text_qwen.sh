#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPHERE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEV_DIR="${DEV_DIR:-workspace}"
DATA_DIR="${DATA_DIR:-datasets}"
DATASET_NAME="${DATASET_NAME:-list}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
EPOCHS="${EPOCHS:-800}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TEXT_ENCODER_MODEL="${TEXT_ENCODER_MODEL:-Qwen/Qwen3.5-0.8B-Base}"

cd "$SPHERE_ROOT"

exec ./run.sh train.py \
  --dev_dir "$DEV_DIR" \
  --data_dir "$DATA_DIR" \
  --dataset_name "$DATASET_NAME" \
  --conditioning_mode embedding \
  --text_encoder_model "$TEXT_ENCODER_MODEL" \
  --text_encoder_extraction_layers 24 \
  --image_size "$IMAGE_SIZE" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --epochs "$EPOCHS" \
  --batch_size_per_rank "$BATCH_SIZE_PER_RANK" \
  --batch_size "$BATCH_SIZE" \
  --compression_ratio 3.0 \
  --noise_sigma_max_angle 85 \
  --cond_generator False \
  --vit_enc_model_size base \
  --vit_dec_model_size base \
  --vit_enc_latent_mlp_mixer_depth 4 \
  --vit_dec_latent_mlp_mixer_depth 4 \
  --affine_latent_mlp_mixer True \
  --pixel_head_type linear \
  --ckpt_save_interval 100 \
  --out_dir experiments \
  --lat_con_loss_weight 0.1 \
  --pix_recon_dist_loss_weight 50 \
  --pix_recon_perc_loss_weight 1.0 \
  --pix_con_dist_loss_weight 25.0 \
  --pix_con_perc_loss_weight 1.0 \
  "$@"
