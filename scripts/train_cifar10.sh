# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

#!/usr/bin/env bash

./run.sh train.py \
  --dataset_name cifar-10 \
  --image_size 32 \
  --warmup_epochs 10 \
  --epochs 5000 \
  --compression_ratio 1.5 \
  --noise_sigma_max_angle 80 \
  --vit_enc_model_size base \
  --vit_dec_model_size base \
  --vit_enc_latent_mlp_mixer_depth 2 \
  --vit_dec_latent_mlp_mixer_depth 2 \
  --affine_latent_mlp_mixer True \
  --pixel_head_type conv \
  --ckpt_save_interval 100 \
  --out_dir experiments \
  --lat_con_loss_weight 0.1 \
  --pix_recon_dist_loss_weight 1.0 \
  --pix_recon_perc_loss_weight 1.0 \
  --pix_con_dist_loss_weight 0.5 \
  --pix_con_perc_loss_weight 0.5
