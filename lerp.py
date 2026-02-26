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
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from sphere.model import G
from sphere.ema import SimpleEMA
from sphere.utils import load_ckpt, save_tensors_to_images
from cli_utils import str2bool
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Interpolation")
# --- directory
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--out_dir", type=str, default="interpolation")
parser.add_argument("--job_dir", type=str, default=None)
# --- interpolation
parser.add_argument(
    "--interp_mode",
    type=str,
    default="lerp",
    choices=[
        "lerp",  # simple linear interpolation
        "slerp",  # spherical interpolation
        "blerp",  # bilinear interpolation (for 4 images)
    ],
)
# --- generation
parser.add_argument("--ckpt_fname", type=str, default=None)
parser.add_argument("--batch_size_per_rank", type=int, default=2)
parser.add_argument("--num_trials", type=int, default=2)
parser.add_argument("--compile_model", type=str2bool, default=True)
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
parser.add_argument("--grid_nrow", type=int, default=16)
parser.add_argument("--grid_ncol", type=int, default=8)
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

    args.random_sample_classes = True
    class_of_interests = [0]  # dummy for one loop
    if args.cond_generator:
        args.random_sample_classes = False
        class_of_interests = list(range(args.num_classes))

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

    # prepare variables
    cfg_vals = [
        round(v, 1) for v in list(np.arange(args.cfg_min, args.cfg_max, args.cfg_gap))
    ] + [args.cfg_max]

    assert args.save_grid_images is True

    if args.interp_mode in ["blerp"]:
        # blerp only supports batch size of 1 for now
        args.batch_size_per_rank = 1

    latent_shape = model.latent_shape  # [B, ...]

    # stack loops
    for try_id in range(args.num_trials):
        for cfg in cfg_vals:
            N = args.batch_size_per_rank
            a = torch.randn(N, *latent_shape[1:], device=device)
            b = torch.randn(N, *latent_shape[1:], device=device)

            if args.cond_generator:
                clss_a = np.random.choice(class_of_interests, size=N)
                clss_b = np.random.choice(class_of_interests, size=N)
                clss_a = torch.tensor(np.array(clss_a), device=device).long()
                clss_b = torch.tensor(np.array(clss_b), device=device).long()
                clss_enc_embed_a = model.encoder.y_embedder(clss_a, False)
                clss_enc_embed_b = model.encoder.y_embedder(clss_b, False)
                clss_dec_embed_a = model.decoder.y_embedder(clss_a, False)
                clss_dec_embed_b = model.decoder.y_embedder(clss_b, False)
            else:
                clss_enc_embed_a = None
                clss_enc_embed_b = None
                clss_dec_embed_a = None
                clss_dec_embed_b = None

            for fwd_step in args.forward_steps:
                # create the sub folder to save images
                save_name = (
                    f"imgs"
                    f"_try={try_id:02d}"
                    f"_pth={ckpt_epoch}"
                    f"_cfg={cfg}-{args.cfg_position}"
                    f"_steps={fwd_step}"
                    f"_sched={args.use_sampling_scheduler}"
                    f"_cache={args.cache_sampling_noise}"
                    f"_interp={args.interp_mode}"
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

                if args.save_grid_images:
                    gen_images = []

                dist.barrier()
                logger.info("start interpolating latents 🍡")

                if args.interp_mode in ["lerp", "slerp"]:
                    grid_nrow = args.grid_nrow
                    grid_ncol = 1

                    embed_interp_func = lerp
                    noise_interp_func = lerp if args.interp_mode == "lerp" else slerp

                    for t in tqdm(
                        np.linspace(0, 1, grid_nrow * grid_ncol),
                        total=grid_nrow * grid_ncol,
                        desc=args.interp_mode,
                    ):
                        noise = noise_interp_func(a, b, t)

                        if args.cond_generator:
                            cls_enc_embed = embed_interp_func(
                                clss_enc_embed_a, clss_enc_embed_b, t
                            )
                            cls_dec_embed = embed_interp_func(
                                clss_dec_embed_a, clss_dec_embed_b, t
                            )
                        else:
                            cls_enc_embed = None
                            cls_dec_embed = None

                        with torch.autocast(device_type="cuda", dtype=ptdtype):
                            x_gen_1_step, x_gen_n_step = model.edit(
                                batch_size=N,
                                y=None,
                                cfg=cfg,
                                cfg_position=args.cfg_position,
                                forward_steps=fwd_step,
                                use_sampling_scheduler=args.use_sampling_scheduler,
                                cache_sampling_noise=args.cache_sampling_noise,
                                y_enc_embed=cls_enc_embed,
                                y_dec_embed=cls_dec_embed,
                                input_noise=noise,
                                device=device,
                            )
                        if args.save_grid_images:
                            gen_images.append(x_gen_n_step)

                if args.interp_mode in ["blerp"]:
                    grid_nrow = args.grid_nrow
                    grid_ncol = args.grid_ncol

                    c = torch.randn(N, *latent_shape[1:], device=device)
                    d = torch.randn(N, *latent_shape[1:], device=device)

                    a = a.reshape(-1)
                    b = b.reshape(-1)
                    c = c.reshape(-1)
                    d = d.reshape(-1)

                    if args.cond_generator:
                        pass

                    noise_grid = blerp(a, b, c, d, grid_nrow, grid_ncol)
                    pbar = tqdm(
                        range(grid_nrow * grid_ncol),
                        total=grid_nrow * grid_ncol,
                        desc=args.interp_mode,
                    )
                    for idx in pbar:
                        row = idx // grid_ncol
                        col = idx % grid_ncol
                        noise = noise_grid[row, col].reshape(1, *latent_shape[1:])
                        with torch.autocast(device_type="cuda", dtype=ptdtype):
                            x_gen_1_step, x_gen_n_step = model.edit(
                                batch_size=N,
                                y=None,
                                cfg=cfg,
                                cfg_position=args.cfg_position,
                                forward_steps=fwd_step,
                                use_sampling_scheduler=args.use_sampling_scheduler,
                                input_noise=noise,
                                device=device,
                            )
                        if args.save_grid_images:
                            gen_images.append(x_gen_n_step)

                if args.save_grid_images:
                    save_tensors_to_images(
                        gen_images, path=save_path, nrow=grid_nrow, max_nimgs=256
                    )

    dist.destroy_process_group()
    return


def lerp(a, b, t):
    return a * (1 - t) + b * t


def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


def blerp(z_tl, z_tr, z_bl, z_br, nrows, ncols):
    top = torch.stack([z_tl, z_tr])  # (2, dim)
    bot = torch.stack([z_bl, z_br])  # (2, dim)
    grid_2x2 = torch.stack([top, bot])  # (2, 2, dim)
    g = grid_2x2.permute(2, 0, 1).unsqueeze(0)  # (1, dim, 2, 2)
    g = torch.nn.functional.interpolate(
        g,
        size=(nrows, ncols),
        mode="bilinear",
        align_corners=True,
    )
    return g.squeeze(0).permute(1, 2, 0)


if __name__ == "__main__":
    main(cli_args)
