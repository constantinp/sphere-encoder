# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import glob
import json
import os
import os.path as osp
import shutil
import logging
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import sphere.rng as rng
from sphere.model import G
from sphere.ema import SimpleEMA
from sphere.utils import (
    load_ckpt,
    save_image,
    save_tensors_to_images,
    nn_concat_all_gather,
)
from cli_utils import str2bool
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generation")
# --- directory
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--out_dir", type=str, default="visualization")
parser.add_argument("--job_dir", type=str, default=None)
# --- generation
parser.add_argument("--ckpt_fname", type=str, default=None)
parser.add_argument("--num_gen_samples", type=int, default=64)
parser.add_argument("--batch_size_per_rank", type=int, default=16)
parser.add_argument("--class_of_interests", type=int, nargs="+", default=[100])
parser.add_argument("--num_trials", type=int, default=2)
parser.add_argument("--compile_model", type=str2bool, default=True)
parser.add_argument("--random_sample_classes", type=str2bool, default=False)
parser.add_argument("--forward_steps", type=int, nargs="+", default=[4])
parser.add_argument("--use_sampling_scheduler", type=str2bool, default=False)
parser.add_argument("--cache_sampling_noise", type=str2bool, default=True)
parser.add_argument("--seed_sampling", type=str2bool, default=False)
parser.add_argument("--use_ema_model", type=str2bool, default=False)
# --- guidance
parser.add_argument("--use_cfg", type=str2bool, default=False)
parser.add_argument("--cfg_min", type=float, default=1.4)
parser.add_argument("--cfg_max", type=float, default=1.4)
parser.add_argument("--cfg_gap", type=float, default=0.2)
parser.add_argument("--cfg_position", type=str, default="combo")
# --- saving
parser.add_argument("--save_grid_images", type=str2bool, default=True)
parser.add_argument("--grid_nrow", type=int, default=8)
cli_args = parser.parse_args()
# -----------------------------------------------------------------------------


def main(cli_args):
    # setup dirs
    exp_dir = osp.join(cli_args.dev_dir, "experiments", cli_args.job_dir)

    # prepare to merge config
    cli_args_dict = vars(cli_args)

    # load config from exp folder
    logger.info(f"load cfg from {exp_dir}")
    config_path = os.path.join(exp_dir, "cfg.json")
    with open(config_path, "r") as fio:
        cfg_args = json.load(fio)

    # let cli args override config file args
    cfg_args.update(cli_args_dict)

    # convert to namespace for easy access
    args = SimpleNamespace(**cfg_args)
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")

    # various inits, derived attributes, I/O setup
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{ddp_local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=datetime.timedelta(hours=2),
    )
    ddp_rank0 = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed

    # seed
    if args.seed_sampling:
        seed = 99
        torch.manual_seed(seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    # create output folders
    out_dir = osp.join(args.dev_dir, args.out_dir, args.job_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"create output folder: {out_dir}")

    if args.dataset_name in ["cifar-10", "flowers-102", "animal-faces"]:
        args.random_sample_classes = True
        logger.info(f"enable random sampling classes for {args.dataset_name}")

    if args.random_sample_classes:
        args.class_of_interests = [0]  # dummy for one loop

    if args.dataset_name in ["flowers-102"]:
        args.cache_sampling_noise = False
        logger.info(f"disable caching sampling noise for {args.dataset_name}")

    # build model
    model = G(
        input_size=args.image_size,
        patch_size=args.patch_size,
        vit_enc_model_size=args.vit_enc_model_size,
        vit_dec_model_size=args.vit_dec_model_size,
        token_channels=args.token_channels,
        num_classes=args.num_classes if args.cond_generator else 0,
        halve_model_size=args.halve_model_size,
        spherify_model=args.spherify_model,
        pixel_head_type=args.pixel_head_type,
        in_context_size=args.in_context_size,
        noise_sigma_max_angle=args.noise_sigma_max_angle,
        vit_enc_latent_mlp_mixer_depth=args.vit_enc_latent_mlp_mixer_depth,
        vit_dec_latent_mlp_mixer_depth=args.vit_dec_latent_mlp_mixer_depth,
        affine_latent_mlp_mixer=args.affine_latent_mlp_mixer,
    )
    model.to(dtype=ptdtype, device=device, memory_format=torch.channels_last)
    logger.info(model)

    ema_model = SimpleEMA(model)

    # load ckpt path
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if len(ckpts) == 0:
        raise ValueError("no checkpoints to eval")
    ckpts = sorted(ckpts)

    # optionally load from a specific ckpt
    load_from = (
        ckpts[-1]
        if args.ckpt_fname is None
        else os.path.join(ckpt_dir, args.ckpt_fname)
    )
    assert os.path.exists(load_from)
    logger.info(f"find the latest ckpt: {load_from}")
    ckpt_epoch = osp.basename(load_from).replace(".pth", "")

    # load ckpt
    load_ckpt(
        model,
        ckpt_path=load_from,
        ema_model=ema_model,
        strict=True,
        override_model_with_ema=args.use_ema_model,
        verbose=True,
    )

    if args.compile_model:
        logger.info("compiling the model...")
        model = torch.compile(model)

    model.eval().requires_grad_(False)

    args.batch_size_per_rank = min(
        args.batch_size_per_rank, args.num_gen_samples // ddp_world_size
    )
    assert args.num_gen_samples % (ddp_world_size * args.batch_size_per_rank) == 0
    num_batches_per_rank = int(
        args.num_gen_samples / ddp_world_size / args.batch_size_per_rank
    )

    cfg_vals = [
        round(v, 1) for v in list(np.arange(args.cfg_min, args.cfg_max, args.cfg_gap))
    ] + [args.cfg_max]

    # stack loops
    for try_id in range(args.num_trials):
        for cfg in cfg_vals:
            for fwd_step in args.forward_steps:
                for clss in args.class_of_interests:
                    # create the sub folder to save images
                    save_name = (
                        f"imgs"
                        f"_try={try_id:02d}"
                        f"_clss={clss:05d}"
                        f"_pth={ckpt_epoch}"
                        f"_cfg={cfg}-{args.cfg_position}"
                        f"_steps={fwd_step}"
                        f"_sched={args.use_sampling_scheduler}"
                        f"_cache={args.cache_sampling_noise}"
                    )
                    if not args.save_grid_images:
                        # save images in this folder
                        sub_dir = osp.join(out_dir, save_name)
                        if osp.exists(sub_dir):
                            shutil.rmtree(sub_dir)
                        os.makedirs(sub_dir, exist_ok=True)
                    else:
                        # save grid image in this path
                        save_path = osp.join(out_dir, save_name + ".png")

                    # prepare class conditioning if needed
                    clss = torch.tensor(
                        [clss] * args.batch_size_per_rank,
                        dtype=torch.long,
                        device=device,
                    )
                    if args.save_grid_images:
                        gen_images = []

                    dist.barrier()
                    logger.info("start sampling images 🍭")

                    pbar = tqdm(range(num_batches_per_rank), total=num_batches_per_rank)
                    for batch_idx in pbar:

                        with (
                            torch.random.fork_rng(devices=[device])
                            if args.seed_sampling
                            else nullcontext()
                        ):
                            if args.seed_sampling:
                                torch.manual_seed(
                                    rng.fold_in(seed, ddp_rank, batch_idx)
                                )

                            with torch.autocast(device_type="cuda", dtype=ptdtype):
                                x_gen_1_step, x_gen_n_step = model.generate(
                                    batch_size=args.batch_size_per_rank,
                                    y=None if args.random_sample_classes else clss,
                                    cfg=cfg,
                                    cfg_position=args.cfg_position,
                                    forward_steps=fwd_step,
                                    use_sampling_scheduler=args.use_sampling_scheduler,
                                    cache_sampling_noise=args.cache_sampling_noise,
                                    device=device,
                                )

                        if args.save_grid_images:
                            gen_images.append(x_gen_n_step)
                        else:
                            save_image(
                                x=x_gen_n_step,
                                batch_idx=batch_idx,
                                ddp_rank=ddp_rank,
                                save_dir=sub_dir,
                            )

                    dist.barrier()
                    if args.save_grid_images:
                        gen_images = torch.cat(gen_images, dim=0)
                        gen_images = nn_concat_all_gather(gen_images)

                        # cifar image in the paper, max 32 in a row
                        save_tensors_to_images(
                            gen_images,
                            path=save_path,
                            nrow=args.grid_nrow,
                            max_nimgs=args.num_gen_samples,
                        )

    dist.destroy_process_group()
    return


if __name__ == "__main__":
    main(cli_args)
