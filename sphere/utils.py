# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import math
import os
from contextlib import nullcontext
from typing import List

import numpy as np
import PIL as pil
import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torchvision
from sphere.loader import resize_arr
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

logger = logging.getLogger(__name__)


def nn_concat_all_gather(tensor, gather_dim=0):
    if dist.get_world_size() == 1:
        return tensor
    output = dist_nn.functional.all_gather(tensor)
    return torch.cat(output, dim=gather_dim)


def concat_all_gather(tensor):
    if dist.get_world_size() == 1:
        return tensor
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def cosine_scheduler(
    max_value,
    min_value,
    current_step,
    warmup_steps=0,
    decay=False,
    decay_steps=0,
):
    # 1) linear warmup for warmup_steps steps
    if current_step < warmup_steps:
        return max_value * (current_step + 1) / (warmup_steps + 1)
    # 2) if not decay lr
    if not decay:
        return max_value
    # 3) if current_step > decay_steps, return min learning rate
    if current_step > decay_steps:
        return min_value
    # 4) in between, use cosine decay down to min learning rate
    decay_ratio = (current_step - warmup_steps) / (decay_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_value + coeff * (max_value - min_value)


def save_tensors_to_images(
    tensors: torch.Tensor | List[torch.Tensor],
    path: str = None,
    nrow_mult: int = 2,
    max_nimgs: int = 48,
    nrow: int = 0,
):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    assert path is not None

    g = len(tensors)
    x = torch.cat(tensors, dim=-1)
    x = x[:max_nimgs].cpu()
    b, c, h, w = x.shape
    x = (
        x.clamp(min=0.0, max=1.0)
        .reshape(b, c, h, g, w // g)
        .permute(0, 3, 1, 2, 4)
        .reshape(b * g, c, h, w // g)
    )
    if x.shape[0] // g <= 16:
        nrow_mult = 1
    grid = torchvision.utils.make_grid(
        x,
        nrow=nrow if nrow else nrow_mult * g,
        padding=2 if h <= 64 else 8,
        pad_value=1,
    )
    torchvision.utils.save_image(grid, path)
    logger.info(f"grid saved to the image: {path}")


@torch.inference_mode()
def save_image(x, batch_idx, ddp_rank, save_dir, force_image_size=-1):
    assert isinstance(x, torch.Tensor)
    x = x * 255.0
    x = torch.floor(x).to(torch.uint8)
    x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
    x = x.cpu().numpy()

    for i, img in enumerate(x):
        image_name = f"rank={ddp_rank:05d}_ord={batch_idx:05d}_idx={i:05d}.png"
        image_path = os.path.join(save_dir, image_name)
        image = pil.Image.fromarray(img)

        if force_image_size > 0:
            image = resize_arr(image, image_size=force_image_size)

        image.save(image_path, format="PNG", compress_level=0)


@torch.inference_mode()
def visualize(
    vis_loader,
    model_without_ddp,
    ddp_rank,
    epoch,
    cfg=1.0,
    cfg_position="combo",
    forward_steps=1,
    use_sampling_scheduler=False,
    class_of_interest=None,
    ema_model=None,
    use_ema_model=False,
    save_dir=None,
    gather_all_tensors=False,
    device="cuda",
    ctx=nullcontext(),
    fsdp_mode=False,
):
    assert save_dir is not None

    model_without_ddp.eval()

    if fsdp_mode:
        fsdp_ctx = FSDP.summon_full_params(
            model_without_ddp, writeback=False, rank0_only=False
        )
    else:
        fsdp_ctx = nullcontext()

    if use_ema_model and ema_model is not None:
        ema_model.store(model_without_ddp)
        ema_model.copy_to(model_without_ddp)
    else:
        logger.info("no ema_model to use")
        use_ema_model = False

    imgs, clss = next(vis_loader)[:2]
    N = 1 if gather_all_tensors else len(imgs)

    imgs = imgs[:N].to(device, non_blocking=True)
    clss = clss[:N].to(device, non_blocking=True)

    if class_of_interest is not None:
        gen_clss = np.random.choice(class_of_interest, size=N)
        gen_clss = torch.tensor(gen_clss, dtype=torch.long, device=device)
    else:
        gen_clss = clss

    with fsdp_ctx, ctx:
        # reconstruction
        rec_imgs = model_without_ddp.reconstruct(imgs, clss, sampling=False)
        rec_imgs_with_noise_small = model_without_ddp.reconstruct(
            imgs, clss, noise_scaler=0.5, sampling=True
        )
        rec_imgs_with_noise_large = model_without_ddp.reconstruct(
            imgs, clss, noise_scaler=1.0, sampling=True
        )

        # generation
        gen_rand_1_step, gen_rand_n_step = model_without_ddp.generate(
            batch_size=N,
            y=gen_clss if model_without_ddp.use_modulation else None,
            cfg=cfg,
            cfg_position=cfg_position,
            forward_steps=forward_steps,
            use_sampling_scheduler=use_sampling_scheduler,
            device=device,
        )

        gen_clss_1_step, gen_clss_n_step = model_without_ddp.generate(
            batch_size=N,
            y=gen_clss if model_without_ddp.use_modulation else None,
            cfg=cfg,
            cfg_position=cfg_position,
            forward_steps=forward_steps,
            use_sampling_scheduler=use_sampling_scheduler,
            device=device,
        )

    ori_imgs = imgs * 0.5 + 0.5

    to_zip = [
        ori_imgs,
        rec_imgs,
        rec_imgs_with_noise_small,
        rec_imgs_with_noise_large,
        gen_rand_1_step,
        gen_clss_1_step,
    ]
    if forward_steps > 1:
        to_zip.append(gen_rand_n_step)
        to_zip[-1], to_zip[-2] = to_zip[-2], to_zip[-1]
        to_zip.append(gen_clss_n_step)

    if ddp_rank == 0:
        img_path = (
            f"imgs_ep{epoch:04d}"
            f"_ema={use_ema_model}"
            f"_cfg={cfg}-{cfg_position}"
            f"_steps={forward_steps}"
            f"_sched={use_sampling_scheduler}"
            f".png"
        )
        save_tensors_to_images(
            to_zip,
            path=os.path.join(save_dir, img_path),
            nrow_mult=2,
        )

    if use_ema_model and ema_model is not None:
        ema_model.restore(model_without_ddp)

    dist.barrier()


def save_ckpt(
    model_without_ddp,
    optimizer=None,
    loss_scaler=None,
    epoch=0,
    ema_model=None,
    discriminator_optimizer=None,
    discriminator_loss_scaler=None,
    ckpt_dir=None,
    ddp_rank0=False,
):
    assert ckpt_dir is not None

    dist.barrier()

    if ddp_rank0:
        ckpt = {
            "model": model_without_ddp.state_dict(),
            "epoch": epoch,
        }
        if ema_model is not None:
            ckpt["ema_model"] = ema_model.state_dict()
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if loss_scaler is not None:
            ckpt["loss_scaler"] = loss_scaler.state_dict()
        if discriminator_optimizer is not None:
            ckpt["discriminator_optimizer"] = discriminator_optimizer.state_dict()
        if discriminator_loss_scaler is not None:
            ckpt["discriminator_loss_scaler"] = discriminator_loss_scaler.state_dict()
        ckpt_path = os.path.join(ckpt_dir, f"ep{epoch:04d}.pth")
        torch.save(ckpt, ckpt_path)
        logger.info(f"checkpoint saved to {ckpt_path}")

        organize_ckpt(
            ckpt_dir,
            keep_num=10,
            milestone_interval=100,
            cleanup_checkpoints=True,
        )

    dist.barrier()


def load_ckpt(
    model_without_ddp,
    ckpt_path=None,
    ema_model=None,
    strict=False,
    override_model_with_ema=False,
    verbose=False,
    return_ckpt=False,
):
    if ckpt_path is None:
        return

    dist.barrier()

    logger.info(f"loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # FIXME: this is temporary fix
    state_dict = ckpt["model"]
    for k, v in state_dict.items():
        if "alpha" in k:
            v = torch.tensor([v]).reshape(1)
        state_dict[k] = v

    msgs = model_without_ddp.load_state_dict(state_dict, strict=strict)
    if strict:
        assert len(msgs.missing_keys) == 0, f"{msgs.missing_keys}"
    else:
        logger.warning(f"missing keys: {msgs.missing_keys}")

    if verbose:
        logger.info(msgs)

    if ema_model is not None:
        if "ema_model" in ckpt:

            # FIXME: this is temporary fix
            ema_state_dict = ckpt["ema_model"]
            for k, v in ema_state_dict.items():
                if "alpha" in k:
                    v = torch.tensor([v]).reshape(1)
                ema_state_dict[k] = v

            ema_model.load_state_dict(ema_state_dict, strict=strict)

            if override_model_with_ema:
                ema_model.copy_to(model_without_ddp)
                logger.info("copy ema model to model")

        else:
            ema_model.load_state_dict(state_dict, strict=False)
            logger.info("no ema_model state_dict, load ema model from model")

            if override_model_with_ema:
                logger.warning(
                    "override_model_with_ema is True, but no ema_model to override"
                )

    dist.barrier()

    if return_ckpt:
        return ckpt

    ckpt = None  # free up memory


def organize_ckpt(
    ckpt_dir: str,
    keep_num: int = 5,
    milestone_interval: int = 5,
    cleanup_checkpoints: bool = False,
):
    """
    Clean up older checkpoint files in `ckpt_dir` while keeping the latest `keep_num` checkpoints by epoch number.

    Parameters
    ----------
    ckpt_dir : str
        The directory where checkpoint .pth files are stored.
    keep_num : int, optional
        The number of most recent checkpoints to keep (default=5).
    milestone_interval : int, optional
        The interval used to decide if a checkpoint is a "milestone."
        If (epoch_num + 1) % milestone_interval == 0, it is kept (default=50).
    cleanup_checkpoints : bool, optional
        Whether to delete the checkpoints that are not kept (default=False).
    """

    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    ckpts = [ckpt for ckpt in ckpts if "latest" not in ckpt and "best" not in ckpt]

    def get_ckpt_num(path):
        """
        Extract the epoch number from a checkpoint filename.
        """
        filename = os.path.basename(path)
        # expecting something like 'epoch_049.pth'
        # we'll parse out the part after the last underscore and before '.pth'
        try:
            return int(filename.rsplit("_", 1)[-1].split(".")[0])
        except ValueError:
            return None

    # sort checkpoints by epoch number
    ckpts.sort(key=lambda x: (get_ckpt_num(x) is None, get_ckpt_num(x)))

    # filter out any that failed to parse an integer epoch (get_ckpt_num == None)
    ckpts = [ckpt for ckpt in ckpts if get_ckpt_num(ckpt) is not None]

    if not ckpts:
        # if no checkpoints remain, nothing to do
        return

    # determine which checkpoints to keep:
    # 1. the newest `keep_num` by epoch number.
    # 2. any milestone checkpoints.
    #    (epoch_num + 1) % milestone_interval == 0
    newest_keep = set(ckpts[-keep_num:])  # handle if keep_num > number of ckpts
    milestone_keep = set(
        ckpt for ckpt in ckpts if ((get_ckpt_num(ckpt) + 1) % milestone_interval == 0)
    )

    # union of both sets
    keep_set = newest_keep.union(milestone_keep)

    # remove anything not in keep_set
    for ckpt in ckpts:
        if ckpt not in keep_set and cleanup_checkpoints:
            os.remove(ckpt)
            logger.info(f"Removed checkpoint: {ckpt}")

    # recreate the 'latest.pth' symlink to the newest checkpoint
    if keep_set:
        # we need the absolute newest based on epoch number
        # sort again from keep_set only
        remaining_ckpts_sorted = sorted(
            keep_set, key=lambda x: (get_ckpt_num(x) is None, get_ckpt_num(x))
        )
        newest_ckpt = os.path.abspath(remaining_ckpts_sorted[-1])
        latest_symlink = os.path.join(ckpt_dir, "latest.pth")

        # remove the old symlink if it exists
        try:
            os.remove(latest_symlink)
            logger.info(f"Removed old symlink: {latest_symlink}")
        except FileNotFoundError:
            pass

        # create a new symlink
        os.symlink(newest_ckpt, latest_symlink)
        logger.info(f"Created symlink: {latest_symlink} -> {newest_ckpt}")


@torch.no_grad()
def compute_psnr_torch_batch(
    input: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
):
    """
    computes psnr for a batch of images using pytorch operations
    """
    mse_per_sample = torch.nn.functional.mse_loss(input, target, reduction="none").mean(
        dim=[1, 2, 3]
    )
    psnr_per_sample = 10.0 * torch.log10(data_range**2 / mse_per_sample)
    return psnr_per_sample


@torch.no_grad()
def vector_compute_magnitude(x):
    assert x.ndim >= 2
    reduce_dims = tuple(range(1, x.ndim))
    mag = x.square().sum(dim=reduce_dims, keepdim=True).sqrt()
    return mag


@torch.no_grad()
def vector_compute_angle(x, y):
    assert x.ndim >= 2
    assert x.shape == y.shape
    reduce_dims = tuple(range(1, x.ndim))
    dot = (x * y).sum(dim=reduce_dims, keepdim=True)
    mag = vector_compute_magnitude(x) * vector_compute_magnitude(y)
    mag = torch.clamp(mag, min=1e-6)
    cos_sim = torch.clamp(dot / mag, min=-1.0, max=1.0)
    angle = torch.acos(cos_sim) / math.pi * 180.0  # rad to deg
    return angle


def save_fsdp_ckpt(
    model_without_fsdp,
    epoch=0,
    ckpt_dir=None,
    ddp_rank0=False,
    **kwargs,
):
    assert ckpt_dir is not None

    dist.barrier()

    # in fsdp_mode, we need to call the state_dict on each rank
    # then stream the overall states on the master rank to save
    with FSDP.state_dict_type(
        model_without_fsdp,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state_dict = model_without_fsdp.state_dict()

    if ddp_rank0:
        ckpt = {
            "model": model_state_dict,
            "epoch": epoch,
        }
        ckpt_path = os.path.join(ckpt_dir, f"ep{epoch:04d}.pth")
        torch.save(ckpt, ckpt_path)
        logger.info(f"checkpoint saved to {ckpt_path}")

    dist.barrier()
