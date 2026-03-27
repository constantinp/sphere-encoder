# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import glob
import json
import argparse
import os
import os.path as osp
import time
import logging
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
import wandb
from cli_utils import str2bool
from sphere.ema import SimpleEMA
from sphere.loader import create_loader, ListDataset
from sphere.logger import append_log, setup_logging
from sphere.loss import ReconstructionLoss
from sphere.model import G
from sphere.text import QwenTextEmbedder
from sphere.utils import cosine_scheduler, load_ckpt, save_ckpt, visualize
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Training")
# --- directory
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--out_dir", type=str, default="experiments")
parser.add_argument("--data_dir", type=str, default="datasets")
# --- logging
parser.add_argument("--log_interval", type=int, default=100, help="in iterations")
parser.add_argument("--vis_interval", type=int, default=2, help="in epochs")
parser.add_argument("--ckpt_save_interval", type=int, default=10, help="in epochs")
# --- visualization
parser.add_argument(
    "--class_of_interest", type=int, nargs="+", default=None, help="for visualization"
)
parser.add_argument("--forward_steps", type=int, default=2)
parser.add_argument(
    "--cfg_position", type=str, default="combo", choices=["enc", "dec", "combo"]
)
parser.add_argument("--cfg", type=float, default=1.0)
# --- wandb
parser.add_argument("--use_wandb", type=str2bool, default=False)
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_key", type=str, default=None)
# --- dataset
parser.add_argument(
    "--dataset_name",
    type=str,
    default="cifar-10",
    choices=[
        "cifar-10",
        "cifar-100",
        "food-101",
        "flowers-102",
        "animal-faces",
        "imagenet",
        "list",
    ],
)
parser.add_argument(
    "--conditioning_mode",
    type=str,
    default="class",
    choices=["class", "embedding"],
)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument(
    "--crop_mode",
    type=str,
    default="center",
    choices=["center", "random", "center_adm", "random_adm"],
)
parser.add_argument("--flip_image", type=str2bool, default=True)
parser.add_argument("--extra_padding", type=str2bool, default=False)
parser.add_argument("--rot_degrees", type=int, default=0)
parser.add_argument(
    "--interp_mode", type=str, default="bicubic", choices=["bicubic", "nearest"]
)
parser.add_argument("--concat_train_val_splits", type=str2bool, default=False)
parser.add_argument("--load_from_zip", type=str2bool, default=False)
parser.add_argument(
    "--max_samples", type=int, default=-1, help="for using partial data"
)
parser.add_argument("--caption_style", type=str, default=None)
parser.add_argument(
    "--text_encoder_model",
    type=str,
    default="Qwen/Qwen3.5-0.8B-Base",
)
parser.add_argument(
    "--text_encoder_extraction_layers",
    type=int,
    nargs="+",
    default=[24],
)
parser.add_argument("--text_encoder_max_length", type=int, default=512)
# --- optimizer
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--batch_size_per_rank", type=int, default=16)
parser.add_argument("--warmup_epochs", type=int, default=5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--grad_clip", type=float, default=1.0)
# --- scheduler
parser.add_argument("--epochs", type=int, default=800)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--encoder_lr_scaler", type=float, default=0.1)
parser.add_argument("--decay_lr", type=str2bool, default=True)
# --- latent
parser.add_argument("--compression_ratio", type=float, default=3.0)
parser.add_argument(
    "--latent_resolution", type=str, default="high", choices=["low", "high"]
)
# --- noise
parser.add_argument("--noise_sigma_max_angle", type=int, default=85)
parser.add_argument("--mix_hard_cases", type=str2bool, default=True)
parser.add_argument("--mix_hard_cases_prob", type=float, default=0.1)
# --- model
parser.add_argument("--vit_enc_model_size", type=str, default="base")
parser.add_argument("--vit_dec_model_size", type=str, default="base")
parser.add_argument("--vit_enc_latent_mlp_mixer_depth", type=int, default=4)
parser.add_argument("--vit_dec_latent_mlp_mixer_depth", type=int, default=4)
parser.add_argument("--affine_latent_mlp_mixer", type=str2bool, default=True)
parser.add_argument("--cond_generator", type=str2bool, default=True)
parser.add_argument(
    "--pixel_head_type", type=str, default="linear", choices=["linear", "conv"]
)
# --- ema
parser.add_argument("--use_ema", type=str2bool, default=True)
parser.add_argument("--ema_model_decay", type=float, default=0.9997)
# --- other model settings that are not applied in the paper
parser.add_argument("--spherify_model", type=str2bool, default=False)
parser.add_argument("--in_context_size", type=int, default=0)
parser.add_argument("--halve_model_size", type=str2bool, default=False)
# --- training
parser.add_argument(
    "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"]
)
parser.add_argument("--device_type", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--compile_model", type=str2bool, default=False)
parser.add_argument("--force_to_bf16", type=str2bool, default=False)
parser.add_argument("--use_activation_checkpointing", type=str2bool, default=False)
# --- loss
parser.add_argument(
    "--distance_loss_type",
    type=str,
    default="l1",
    choices=["l1", "l2", "l1+l2", "l2+l1"],
)
parser.add_argument("--pix_recon_dist_loss_weight", type=float, default=1.0)
parser.add_argument("--pix_recon_perc_loss_weight", type=float, default=1.0)
parser.add_argument("--pix_con_dist_loss_weight", type=float, default=1.0)
parser.add_argument("--pix_con_perc_loss_weight", type=float, default=1.0)
parser.add_argument("--lat_con_loss_weight", type=float, default=0.1)
# --- resume
parser.add_argument(
    "--load_from", type=str, default=None, help="just load model weights"
)
parser.add_argument("--resume_from", type=str, default=None)
parser.add_argument(
    "--init_from", type=str, default="scratch", choices=["scratch", "resume"]
)
parser.add_argument("--auto_resume", type=str2bool, default=True)
parser.add_argument("--override_model_with_ema", type=str2bool, default=False)
parser.add_argument("--override_ema_with_model", type=str2bool, default=False)
# --- lpips
parser.add_argument("--perceptual_ckpt_path", type=str, default="pretrained/lpips")
cli_args = parser.parse_args()
# -----------------------------------------------------------------------------


def set_exp_name(args):
    exp_name = "sphere"
    exp_name += f"-{args.vit_enc_model_size}"
    exp_name += f"-{args.vit_dec_model_size}"
    exp_name += f"-{args.dataset_name}"
    exp_name += f"-{args.conditioning_mode}"
    exp_name += f"-{args.image_size}px"
    if args.use_wandb:
        exp_name += f"-{os.urandom(6).hex()[:6]}"
    return exp_name


def build_text_embedder(args, device, ptdtype):
    return QwenTextEmbedder(
        model_name=args.text_encoder_model,
        extraction_layers=args.text_encoder_extraction_layers,
        max_length=args.text_encoder_max_length,
        dtype=ptdtype,
        device=device,
    )


def main(args):
    # setup dirs
    job_dir = set_exp_name(args)
    exp_dir = osp.join(args.dev_dir, args.out_dir, job_dir)

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
    torch.manual_seed(99 + seed_offset)  # number 99
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    ctx = (
        torch.amp.autocast(device_type=args.device_type, dtype=ptdtype)
        if args.device_type == "cuda"
        else nullcontext()
    )

    # directories
    if ddp_rank0:
        os.makedirs(exp_dir, exist_ok=True)
    args.vis_dir = os.path.join(exp_dir, "vis")  # visualizations
    args.ckpt_dir = os.path.join(exp_dir, "ckpt")  # checkpoints
    for d in [args.vis_dir, args.ckpt_dir]:
        if not ddp_rank0:
            continue
        os.makedirs(d, exist_ok=True)
    dist.barrier()

    setup_logging(output_path=exp_dir, rank=ddp_rank)
    logger.info(f"logging to {exp_dir}")

    # for detailed logging
    log_training_path = os.path.join(exp_dir, "log.jsonl")

    # prepare various components before training starts
    if args.conditioning_mode == "embedding":
        args.num_classes = 0
        args.cond_generator = False
        args.class_of_interest = None
    else:
        args.num_classes = {
            "food-101": 101,
            "flowers-102": 102,
            "animal-faces": 3,
            "cifar-10": 10,
            "cifar-100": 100,
            "imagenet": 1000,
        }[args.dataset_name]

    if args.conditioning_mode == "class" and args.dataset_name == "imagenet":
        args.class_of_interest = [
            913,  # wreck
            283,  # persian cat
            17,  # jay
            607,  # jack-o-lantern
            933,  # cheeseburger
            148,  # killer whale
            207,  # golden retriever
            387,  # lesser panda
            88,  # macaw
            979,  # valley
            417,  # balloon
            279,  # arctic fox
            928,  # ice cream
            953,  # pineapple
            936,  # head cabbage
            866,  # tractor
            107,  # jellyfish
            574,  # golf ball
            393,  # anemone fish
            269,  # timber wolf
            815,  # spider web
            950,  # orange
            644,  # matchstick
            852,  # tennis ball
            130,  # flamingo
        ]

    if args.dataset_name == "cifar-10":
        dataset_cls = datasets.CIFAR10
    elif args.dataset_name == "cifar-100":
        dataset_cls = datasets.CIFAR100
    else:
        # load images from a list of file paths and labels
        dataset_cls = ListDataset

    if args.dataset_name == "list" and args.conditioning_mode == "class":
        raise ValueError("dataset_name='list' currently supports conditioning_mode='embedding' only")

    if args.image_size <= 64:
        args.interp_mode = "nearest"

    interp_mode = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "nearest": transforms.InterpolationMode.NEAREST,
    }[args.interp_mode]

    # default patch size (for the case of latent_resolution = "low")
    if args.image_size in [32, 64]:
        args.patch_size = 4
    elif args.image_size in [128]:
        args.patch_size = 8
    elif args.image_size in [256]:
        args.patch_size = 16
    elif args.image_size in [512]:
        args.patch_size = 32

    if args.latent_resolution == "low":
        args.patch_size = args.patch_size // 1
    elif args.latent_resolution == "high":
        args.patch_size = args.patch_size // 2
    elif args.latent_resolution == "super_high":
        args.patch_size = args.patch_size // 4
    else:
        raise ValueError(f"unknown latent resolution: {args.latent_resolution}")

    # compute token channels
    args.latent_resolution = args.image_size // args.patch_size
    args.token_channels = int(
        3 * args.image_size**2 / args.latent_resolution**2 / args.compression_ratio
    )

    # mix hard cases settings
    args.mix_hard_cases_max_angle = min(args.noise_sigma_max_angle + 5, 89)

    text_embedder = None
    args.cond_dim = 0
    if args.conditioning_mode == "embedding":
        text_embedder = build_text_embedder(args, device=device, ptdtype=ptdtype)
        args.cond_dim = text_embedder.output_dim
        logger.info(
            "embedding conditioning enabled with %s, layers=%s, cond_dim=%s",
            args.text_encoder_model,
            args.text_encoder_extraction_layers,
            args.cond_dim,
        )

    # logger
    if args.use_wandb and ddp_rank0:
        wandb.login(key=args.wandb_key, host="https://api.wandb.ai", relogin=True)
        wandb.init(
            name=job_dir,
            project=args.wandb_project,
            config=vars(args),
            entity=args.wandb_entity,
        )

    # dump config
    config_path = os.path.join(exp_dir, "cfg.json")
    if ddp_rank0:
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)

    # build data loader and sampler
    train_loader, test_loader, vis_loader, train_sampler = create_loader(
        dataset_cls,
        osp.join(args.dev_dir, args.data_dir, args.dataset_name),
        args.image_size,
        args.patch_size,
        max_samples=args.max_samples,
        interp_mode=interp_mode,
        rot_degrees=args.rot_degrees,
        crop_mode=args.crop_mode,
        flip_image=args.flip_image,
        extra_padding=args.extra_padding,
        concat_train_val_splits=args.concat_train_val_splits,
        ddp_world_size=ddp_world_size,
        ddp_rank=ddp_rank,
        batch_size_per_rank=args.batch_size_per_rank,
        num_workers=args.num_workers,
        load_from_zip=args.load_from_zip,
        conditioning_mode=args.conditioning_mode,
        caption_style=args.caption_style,
    )
    logger.info(f"training dataset: {train_loader.dataset}")

    # learning rate decay scheduler (cosine with warmup)
    effective_batch_size = args.batch_size_per_rank * ddp_world_size

    total_steps = int(
        len(train_loader)
        * ddp_world_size
        * args.batch_size_per_rank
        * args.epochs
        / effective_batch_size
    )

    # --- warmup and decay
    warmup_steps_ratio = args.warmup_epochs / args.epochs
    lr_decay_steps_ratio = 1 - warmup_steps_ratio

    warmup_steps = int(warmup_steps_ratio * total_steps)
    lr_decay_steps = int(lr_decay_steps_ratio * total_steps)

    # rescale lr if needed
    if effective_batch_size != args.batch_size:
        rescaler = effective_batch_size / args.batch_size
        args.learning_rate *= rescaler
        args.min_lr *= rescaler
        logger.info(
            f"learning rate is rescaled to {args.learning_rate} (min_lr = {args.min_lr})"
        )

    # init these up here, can override later if we want to resume from a checkpoint
    cur_epoch = 0

    # build loss
    loss_cls = ReconstructionLoss(
        perceptual_ckpt_path=osp.join(args.dev_dir, args.perceptual_ckpt_path),
        perceptual_weight=args.pix_recon_perc_loss_weight,
        distance_loss_type=args.distance_loss_type,
        distance_weight=args.pix_recon_dist_loss_weight,
        pixel_consistency_distance_weight=args.pix_con_dist_loss_weight,
        pixel_consistency_perceptual_weight=args.pix_con_perc_loss_weight,
        latent_consistency_weight=args.lat_con_loss_weight,
    )
    loss_cls.to(device=device)
    logger.info(loss_cls)

    # build model
    model = G(
        # model args
        input_size=args.image_size,
        patch_size=args.patch_size,
        vit_enc_model_size=args.vit_enc_model_size,
        vit_dec_model_size=args.vit_dec_model_size,
        token_channels=args.token_channels,
        num_classes=args.num_classes if args.cond_generator else 0,
        cond_dim=args.cond_dim,
        halve_model_size=args.halve_model_size,
        in_context_size=args.in_context_size,
        pixel_head_type=args.pixel_head_type,
        spherify_model=args.spherify_model,
        # latent args
        use_pixel_consistency=args.pix_con_dist_loss_weight > 0
        or args.pix_con_perc_loss_weight > 0,
        use_latent_consistency=args.lat_con_loss_weight > 0,
        # noise args
        noise_sigma_max_angle=args.noise_sigma_max_angle,
        mix_hard_cases=args.mix_hard_cases,
        mix_hard_cases_prob=args.mix_hard_cases_prob,
        mix_hard_cases_max_angle=args.mix_hard_cases_max_angle,
        # mixer args
        vit_enc_latent_mlp_mixer_depth=args.vit_enc_latent_mlp_mixer_depth,
        vit_dec_latent_mlp_mixer_depth=args.vit_dec_latent_mlp_mixer_depth,
        affine_latent_mlp_mixer=args.affine_latent_mlp_mixer,
    )
    model.to(device=device, memory_format=torch.channels_last)
    if args.force_to_bf16:
        model.to(ptdtype)
    logger.info(model)

    ema_model = None
    if args.use_ema:
        ema_model = SimpleEMA(model, decay=args.ema_model_decay)

    if args.auto_resume:
        ckpts = glob.glob(os.path.join(args.ckpt_dir, "*.pth"))
        if len(ckpts) > 0:
            ckpts = sorted(ckpts)  # 0 to N epochs
            args.init_from = "resume"
            args.resume_from = ckpts[-1]
            logger.info(f"auto resume from {args.resume_from}")

    ckpt_path = args.load_from
    if args.init_from == "resume":
        ckpt_path = args.resume_from

    checkpoint = load_ckpt(
        model,
        ckpt_path=ckpt_path,
        ema_model=ema_model,
        strict=False,
        override_model_with_ema=args.override_model_with_ema,
        verbose=True,
        return_ckpt=True,
    )
    if args.init_from == "resume":
        cur_epoch = checkpoint["epoch"]
        if args.use_ema and args.override_ema_with_model:
            ema_model = SimpleEMA(model, decay=args.ema_model_decay)

    # build optimizer
    params_enc_w_decay = []
    params_dec_w_decay = []
    params_enc = []  # w/o decay
    params_dec = []  # w/o decay

    exclude = lambda name, p: (
        p.ndim < 2
        or any(
            keyword in name
            for keyword in [
                "ln",
                "bias",
                "embedding",
                "norm",
                "embed",
                "token",
                "decoder",
            ]
        )
    )

    max_n_len = max([len(n) for n, _ in model.named_parameters()])
    for n, p in model.named_parameters():
        if not p.requires_grad:
            logger.info(
                f"p.requires_grad: {str(p.requires_grad):<5}, "
                f"param: {n:<{max_n_len}}, "
                f"{p.shape}"
            )
            continue
        else:
            p.requires_grad = True

        if exclude(n, p):
            with_decay = False
            if "encoder" in n:
                params_enc.append(p)
            elif "decoder" in n:
                params_dec.append(p)
            else:
                raise ValueError(f"unknown param: {n}")
        else:
            with_decay = True
            if "encoder" in n:
                params_enc_w_decay.append(p)
            elif "decoder" in n:
                params_dec_w_decay.append(p)
            else:
                raise ValueError(f"unknown param: {n}")

        log_str = (
            f"p.requires_grad: {str(p.requires_grad):<5}, "
            f"param: {n:<{max_n_len}}, "
            f"decay: {str(with_decay):<5}, "
            f"{str(p.shape):<30}"
        )

        logger.info(log_str)

    optim_params = [
        {
            "params": params_enc,
            "weight_decay": 0.0,
            "pg_name": "encoder",
        },
        {
            "params": params_enc_w_decay,
            "weight_decay": args.weight_decay,
            "pg_name": "encoder",
        },
        {
            "params": params_dec,
            "weight_decay": 0.0,
            "pg_name": "decoder",
        },
        {
            "params": params_dec_w_decay,
            "weight_decay": args.weight_decay,
            "pg_name": "decoder",
        },
    ]

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.learning_rate,
        betas=(0.90, 0.95),
        weight_decay=0.0,
        fused=True,  # speed up step training
    )

    # free up memory
    if checkpoint is not None:
        checkpoint = None

    # count params
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"parameters of model: {total_params/1e6:.2f}M")
    logger.info(f"parameters to train: {train_params/1e6:.2f}M")

    # compile the model
    if args.compile_model:
        logger.info("compiling the model...")
        torch._dynamo.config.optimize_ddp = True
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model)

    # wrap the model in DDP
    model = DDP(model, device_ids=[ddp_local_rank])

    # apply activation checkpointing to transformer blocks for memory efficiency
    if args.use_activation_checkpointing:
        from sphere.layers import Block

        check_fn = lambda submodule: isinstance(submodule, Block)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
        logger.info("activation checkpointing enabled for transformer layers")

    # training loop
    model_without_ddp = model.module
    global_step = 0

    if args.init_from == "resume":
        global_step = int(
            len(train_loader)
            * ddp_world_size
            * args.batch_size_per_rank
            * cur_epoch
            / effective_batch_size
        )

    visualize(
        vis_loader,
        model_without_ddp,
        ddp_rank,
        epoch=cur_epoch if args.init_from == "resume" else 0,
        cfg=args.cfg,
        cfg_position=args.cfg_position,
        class_of_interest=args.class_of_interest,
        forward_steps=args.forward_steps,
        use_ema_model=False,
        ema_model=ema_model,
        save_dir=args.vis_dir,
        device=device,
        ctx=ctx,
        conditioning_mode=args.conditioning_mode,
        text_embedder=text_embedder,
    )

    # scheduler for lr
    get_lr = partial(
        cosine_scheduler,
        warmup_steps=warmup_steps,
        decay=args.decay_lr,
        decay_steps=lr_decay_steps,
    )

    logger.info(
        f"training starts at epoch {cur_epoch} and global step {global_step} 🚀"
    )
    for epoch in range(cur_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        for local_step, data in enumerate(train_loader):
            dd = time.perf_counter() - t0

            lr = get_lr(args.learning_rate, args.min_lr, global_step)
            for pg in optimizer.param_groups:
                if "encoder" in pg["pg_name"]:
                    pg["lr"] = lr * args.encoder_lr_scaler
                else:
                    pg["lr"] = lr

            imgs, cond_input = data[:2]  # class ids or caption strings
            imgs = imgs.to(device, non_blocking=True)  # [-1, 1]
            clss = None
            cond_embed = None
            if args.conditioning_mode == "embedding":
                cond_embed = text_embedder.encode_pooled(list(cond_input))
            else:
                clss = cond_input.to(device, non_blocking=True)

            # forward
            with ctx:
                rec_imgs, extra_loss, z_noisy, z_clean = model(
                    imgs,
                    clss,
                    cond_embed=cond_embed,
                )
                loss = loss_cls(
                    input=rec_imgs * 0.5 + 0.5,  # [0, 1]
                    target=imgs * 0.5 + 0.5,
                    epoch=epoch,
                    extra_loss=extra_loss,
                    noisy_latent=z_noisy,
                    clean_latent=z_clean,
                )

            loss.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            # update ema model
            if args.use_ema:
                ema_model.step(model)

            # update logging
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1
            global_step += 1

            if (
                local_step == 0 or (local_step + 1) % args.log_interval == 0
            ) and ddp_rank0:
                lossf = loss.item()
                log_str = (
                    f"epoch {epoch:4d} | "
                    f"iter {local_step:7d} [/{len(train_loader)}] | "
                    f"step {global_step:5d} [/{total_steps}]: "
                    f"lr {lr:.2e}, "
                    f"loss {lossf:.5f}, "
                    f"time {dt*1000:.2f}ms, "
                    f"data {dd*1000:.2f}ms"
                )
                for k, v in loss_cls.log_dict.items():
                    log_str += f", {k} {v:.5f}"

                if args.grad_clip > 0:
                    log_str += f", grad_norm {grad_norm:.5f}"

                log_dict = {
                    "loss": lossf,
                    "lr": lr,
                    "epoch": epoch,
                    "step": global_step,
                    **loss_cls.log_dict,
                    **model_without_ddp.log_dict,
                }
                if args.grad_clip > 0:
                    log_dict["grad_norm"] = grad_norm

                logger.info(log_str)

                if args.use_wandb:
                    wandb.log(log_dict)

                append_log(file_path=log_training_path, entry=log_dict)

            # NOTE: break after the first iteration for debugging
            # if local_step == 1:
            #     break

        if epoch == 0 or (epoch + 1) % args.vis_interval == 0:
            visualize(
                vis_loader,
                model_without_ddp,
                ddp_rank,
                epoch,
                class_of_interest=args.class_of_interest,
                forward_steps=args.forward_steps,
                use_ema_model=False,
                ema_model=ema_model,
                save_dir=args.vis_dir,
                device=device,
                ctx=ctx,
                conditioning_mode=args.conditioning_mode,
                text_embedder=text_embedder,
            )

        if (epoch + 1) % args.ckpt_save_interval == 0:
            save_ckpt(
                model_without_ddp,
                epoch=epoch,
                ema_model=ema_model,
                ckpt_dir=args.ckpt_dir,
                ddp_rank0=ddp_rank0,
            )

    # save the final checkpoint
    save_ckpt(
        model_without_ddp,
        epoch=epoch,
        ema_model=ema_model,
        ckpt_dir=args.ckpt_dir,
        ddp_rank0=ddp_rank0,
    )

    # end of training
    dist.destroy_process_group()


if __name__ == "__main__":
    main(cli_args)
