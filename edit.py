# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from copy import deepcopy
import datetime
import glob
import json
import os
import os.path as osp
import shutil
import logging
from types import SimpleNamespace
from torchvision import datasets

import numpy as np
import torch
import torch.distributed as dist
from sphere.loader import center_crop_arr
from sphere.model import G
from sphere.ema import SimpleEMA
from sphere.utils import load_ckpt, save_tensors_to_images
from cli_utils import str2bool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Image Editing")
# --- directory
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--out_dir", type=str, default="image_editing")
parser.add_argument("--job_dir", type=str, default=None)
# --- image editing
parser.add_argument("--input_image", type=str, default=None)
parser.add_argument("--extra_image", type=str, default=None)
parser.add_argument(
    "--edit_mode",
    type=str,
    default="reconstruction",
    choices=[
        "condition",  # manipulate one image with conditioning on a class label
        "crossover",  # smooth stitched images
        "reconstruction",  # just reconstruct input image
    ],
)
parser.add_argument(
    "--class_of_interests",
    type=int,
    nargs="+",
    default=[269, 270, 271, 272, 248, 249, 250, 985],
)
parser.add_argument("--stitch_mode", type=str, default="horizontal")
parser.add_argument("--stitch_swap", type=str2bool, default=False)
parser.add_argument("--noise_strength_scaler", type=float, default=1.0)
# --- generation
parser.add_argument("--ckpt_fname", type=str, default=None)
parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--compile_model", type=str2bool, default=True)
parser.add_argument("--forward_steps", type=int, nargs="+", default=[10])
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

    if not args.cond_generator:
        args.class_of_interests = [None]  # dummy class for one loop

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
    grid_nrow = 0

    # always load the input image
    input_image1 = load_image_to_tensor(args.input_image, args.image_size)
    input_image1 = input_image1.to(device=device).unsqueeze(0)
    gen_images = [input_image1 * 0.5 + 0.5]
    grid_nrow += 1

    if args.edit_mode == "reconstruction":
        with torch.autocast(device_type="cuda", dtype=ptdtype):
            x_rec = model.reconstruct(input_image1)
        gen_images.append(x_rec)
        grid_nrow += 1

        save_name = f"imgs" f"_pth={ckpt_epoch}" f"_edit_mode={args.edit_mode}"
        if not args.save_grid_images:
            # save images in this folder
            sub_dir = osp.join(out_dir, save_name)
            if osp.exists(sub_dir):
                shutil.rmtree(sub_dir)
            os.makedirs(sub_dir, exist_ok=True)
        else:
            # save grid image in this path
            save_path = osp.join(out_dir, save_name + ".png")

        save_tensors_to_images(
            gen_images,
            path=save_path,
            nrow=grid_nrow,
            max_nimgs=256,
        )

        dist.destroy_process_group()
        return

    elif args.edit_mode == "crossover":
        input_image2 = load_image_to_tensor(args.extra_image, args.image_size)
        input_image2 = input_image2.to(device=device).unsqueeze(0)
        gen_images.append(input_image2 * 0.5 + 0.5)
        grid_nrow += 1

        # stitch two images to create the input image for editing
        input_image1 = stitch(
            input_image1,
            input_image2,
            stitch_mode=args.stitch_mode,
            swap=args.stitch_swap,
        )
        gen_images.append(input_image1 * 0.5 + 0.5)
        grid_nrow += 1

        # for conditional model, use null class embedding
        if args.cond_generator:
            args.class_of_interests = [args.num_classes]  # null class

        cache_sampling_noise_stack = [False, True]
        use_sampling_scheduler_stack = [False, True]

    elif args.edit_mode == "condition":
        cache_sampling_noise_stack = [args.cache_sampling_noise]
        use_sampling_scheduler_stack = [args.use_sampling_scheduler]

    clean_gen_images = deepcopy(gen_images)

    # stack loops
    for try_id in range(args.num_trials):
        for cache_sampling_noise in cache_sampling_noise_stack:
            for use_sampling_scheduler in use_sampling_scheduler_stack:
                for fwd_step in args.forward_steps:
                    for clss in args.class_of_interests:
                        gen_images = deepcopy(clean_gen_images)

                        clss_str = f"_clss={clss:05d}" if clss is not None else ""

                        # create the sub folder to save images
                        save_name = (
                            f"imgs"
                            f"_try={try_id:02d}"
                            f"{clss_str}"
                            f"_pth={ckpt_epoch}"
                            f"_steps={fwd_step}"
                            f"_strength={args.noise_strength_scaler}"
                            f"_sched={use_sampling_scheduler}"
                            f"_cache={cache_sampling_noise}"
                            f"_edit_mode={args.edit_mode}"
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

                        N = input_image1.shape[0]
                        if clss is not None:
                            clss = torch.full(
                                (N,), clss, dtype=torch.long, device=device
                            )
                            clss_enc_embed = model.encoder.y_embedder(clss, False)
                            clss_dec_embed = model.decoder.y_embedder(clss, False)
                        else:
                            clss_enc_embed = None
                            clss_dec_embed = None

                        with torch.autocast(device_type="cuda", dtype=ptdtype):
                            x_gen_1_step, x_gen_n_step = model.edit(
                                batch_size=N,
                                y=clss if args.cond_generator else None,
                                cfg_position=args.cfg_position,
                                forward_steps=fwd_step,
                                use_sampling_scheduler=use_sampling_scheduler,
                                cache_sampling_noise=cache_sampling_noise,
                                y_enc_embed=clss_enc_embed,
                                y_dec_embed=clss_dec_embed,
                                x_enc_image=input_image1,
                                return_step_images=True,
                                device=device,
                            )

                        if isinstance(x_gen_n_step, list):
                            gen_images.extend(x_gen_n_step)
                            grid_nrow += len(x_gen_n_step)
                        else:
                            gen_images.append(x_gen_n_step)
                            grid_nrow += 1

                        save_tensors_to_images(
                            gen_images,
                            path=save_path,
                            nrow=grid_nrow,
                            max_nimgs=256,
                        )

    dist.destroy_process_group()
    return


def load_image_to_tensor(path, image_size):
    x = datasets.folder.pil_loader(path)
    x = np.array(center_crop_arr(x, image_size))
    x = x.astype(np.float32) / 255.0  # [0, 1]
    x = torch.from_numpy(x).permute(2, 0, 1)  # [3, H, W]
    x = x * 2 - 1  # [-1, 1]
    return x


def stitch(x1, x2, stitch_mode="horizontal", swap=False):
    assert stitch_mode in ["vertical", "horizontal", "tri_backward", "tri_forward"]

    h, w = x1.shape[-2], x1.shape[-1]
    assert h == w

    canvas = torch.zeros((1, *list(x1.shape[1:])), dtype=x1.dtype, device=x1.device)
    s = x1.shape[2] // 2

    if swap:
        x1, x2 = x2, x1

    # |
    if stitch_mode == "vertical":
        canvas[:, :, :s, :] = x1[:, :, :s, :]
        canvas[:, :, s:, :] = x2[:, :, s:, :]

    # --
    if stitch_mode == "horizontal":
        canvas[:, :, :, :s] = x1[:, :, :, :s]
        canvas[:, :, :, s:] = x2[:, :, :, s:]

    # \
    elif stitch_mode == "tri_backward":
        mask = torch.ones((h, w), device=x1.device, dtype=x1.dtype)
        mask = torch.triu(mask)
        mask = mask.unsqueeze(0).unsqueeze(0)
        canvas = x1 * mask + x2 * (1 - mask)

    # /
    elif stitch_mode == "tri_forward":
        mask = torch.ones((h, w), device=x1.device, dtype=x1.dtype)
        mask = torch.triu(mask)
        mask = torch.flip(mask, dims=[-1])
        mask = mask.unsqueeze(0).unsqueeze(0)
        canvas = x1 * mask + x2 * (1 - mask)

    return canvas


if __name__ == "__main__":
    main(cli_args)
