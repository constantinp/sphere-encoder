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
import shutil
import logging
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import sphere.rng as rng
import torch
import torch.distributed as dist
import torch_fidelity
from sphere.model import G
from sphere.ema import SimpleEMA
from sphere.loader import create_dataset, cycle, resize_arr
from sphere.utils import load_ckpt, save_image, save_tensors_to_images
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cli_utils import str2bool
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluation")
# --- directory
parser.add_argument("--dev_dir", type=str, default="workspace")
parser.add_argument("--out_dir", type=str, default="evaluation")
parser.add_argument("--data_dir", type=str, default="datasets")
parser.add_argument("--job_dir", type=str, default=None)
# --- generation
parser.add_argument("--ckpt_fname", type=str, default=None)
parser.add_argument("--num_eval_samples", type=int, default=50 * 1000)
parser.add_argument("--batch_size_per_rank", type=int, default=25)
parser.add_argument("--forward_steps", type=int, nargs="+", default=[1, 4])
parser.add_argument("--use_sampling_scheduler", type=str2bool, default=False)
parser.add_argument("--cache_sampling_noise", type=str2bool, default=True)
parser.add_argument("--seed_sampling", type=str2bool, default=False)
parser.add_argument("--use_ema_model", type=str2bool, default=False)
parser.add_argument("--compile_model", type=str2bool, default=True)
# --- guidance
parser.add_argument("--use_cfg", type=str2bool, default=False)
parser.add_argument("--cfg_min", type=float, default=1.0)
parser.add_argument("--cfg_max", type=float, default=2.0)
parser.add_argument("--cfg_gap", type=float, default=0.2)
parser.add_argument("--cfg_position", type=str, default="combo")
# --- metrics
parser.add_argument(
    "--fid_stats_used_from",
    type=str,
    default="rand-50k",
    choices=["jit", "adm", "extr", "rand-50k"],
)
parser.add_argument("--fid_stats_dir", type=str, default="fid_stats")
parser.add_argument("--fid_ref_dir", type=str, default="fid_refs")
parser.add_argument("--report_fid", type=str, nargs="+", default=["rfid", "gfid"])
parser.add_argument("--report_precision_recall", type=str2bool, default=False)
# --- flops
parser.add_argument("--report_flops", type=str2bool, default=False)
parser.add_argument("--flops_steps", type=int, default=1)
# --- saving
parser.add_argument("--save_grid_images", type=str2bool, default=True)
parser.add_argument("--num_snapshot_samples", type=int, default=256)
parser.add_argument("--rm_folder_after_eval", type=str2bool, default=False)
cli_args = parser.parse_args()
# -----------------------------------------------------------------------------


def main(cli_args):
    # setup dirs
    exp_dir = osp.join(cli_args.dev_dir, "jobs", cli_args.job_dir)

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
    seed = None
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

    # save generated images to this folder
    out_dir = osp.join(args.dev_dir, args.out_dir, args.job_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"output dir for images: {out_dir}")

    # save summary table to this folder
    tabl_dir = osp.join(exp_dir, "eval")
    os.makedirs(tabl_dir, exist_ok=True)
    logger.info(f"output dir for tables: {tabl_dir}")

    if args.dataset_name in ["cifar-10", "cifar-100", "animal-faces", "flowers-102"]:
        args.fid_stats_used_from = "extr"

    ref_imgs_fold_name = f"ref_images_{args.dataset_name}_{args.image_size}px"
    args.fid_ref_dir = osp.join(args.dev_dir, args.fid_ref_dir)
    if "rfid" in args.report_fid and args.dataset_name not in ["cifar-10", "cifar-100"]:
        assert osp.exists(osp.join(args.fid_ref_dir, ref_imgs_fold_name)), (
            f"reference images for reconstruction FID not found at: "
            f"{osp.join(args.fid_ref_dir, ref_imgs_fold_name)}"
        )
        logger.info(
            f"found reference images for reconstruction FID at: {osp.join(args.fid_ref_dir, ref_imgs_fold_name)}"
        )

    fid_stats_file_path = osp.join(
        args.dev_dir,
        args.fid_stats_dir,
        f"fid_stats_{args.fid_stats_used_from}_{args.dataset_name}_{args.image_size}px.npz",
    )
    assert osp.exists(
        fid_stats_file_path
    ), f"FID stats not found at: {fid_stats_file_path}"

    # save snapshot images to this folder
    snapshot_save_dir = None
    if args.save_grid_images:
        snapshot_save_dir = osp.join(exp_dir, "eval_snapshot")
        os.makedirs(snapshot_save_dir, exist_ok=True)
        logger.info(f"output snapshot dir: {snapshot_save_dir}")

    # build loader for reconstruction task
    if "rfid" in args.report_fid:
        Ts = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: resize_arr(pil_image, args.image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        if args.dataset_name in ["cifar-10", "cifar-100"]:
            dataset_cls = datasets.__dict__[args.dataset_name.upper().replace("-", "")]
            ds = create_dataset(
                dataset_cls,
                root=osp.join(args.dev_dir, args.data_dir, args.dataset_name),
                split="train",
                download=True,
                transform=Ts,
            )
        else:
            dataset_cls = datasets.ImageFolder
            ds = dataset_cls(
                root=osp.join(args.fid_ref_dir, ref_imgs_fold_name), transform=Ts
            )

        sampler = torch.utils.data.DistributedSampler(
            ds, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
        )

        loader = DataLoader(
            ds,
            batch_size=args.batch_size_per_rank,
            sampler=sampler,
            num_workers=8,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        logger.info(
            f"{len(loader)} batches in the loader of reference images on rank {ddp_rank}"
        )
        loader = cycle(loader)
    else:
        loader = None

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

    if args.report_flops and ddp_rank0:
        run_to_meaure_flops(model, device=device, ptdtype=ptdtype)
        dist.barrier()
        dist.destroy_process_group()
        return

    ema_model = SimpleEMA(model)

    # load ckpt path
    ckpt_dir = osp.join(exp_dir, "ckpt")
    ckpts = sorted(glob.glob(osp.join(ckpt_dir, "*.pth")))
    if not ckpts:
        raise ValueError("no checkpoints to eval")

    # optionally load from a specific ckpt
    load_from = (
        ckpts[-1] if args.ckpt_fname is None else osp.join(ckpt_dir, args.ckpt_fname)
    )
    assert osp.exists(load_from), f"ckpt not found: {load_from}"
    logger.info(f"load checkpoint from: {load_from}")
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
        model = torch.compile(model)
        logger.info("model is compiled")

    model.eval().requires_grad_(False)

    # kwargs for evaluation calls
    eval_kwargs = dict(
        model=model,
        loader=loader,
        use_ema=args.use_ema_model,
        image_size=args.image_size,
        dataset_name=args.dataset_name,
        num_classes=args.num_classes if args.cond_generator else 0,
        save_dir=out_dir,
        tabl_dir=tabl_dir,
        fid_stats_file_path=fid_stats_file_path,
        fid_stats_used_from=args.fid_stats_used_from,
        fid_ref_dir=args.fid_ref_dir,
        ckpt_epoch=ckpt_epoch,
        save_snapshot=args.save_grid_images,
        num_snapshot_samples=args.num_snapshot_samples,
        snapshot_save_dir=snapshot_save_dir,
        report_prc=args.report_precision_recall,
        seed_sampling=args.seed_sampling,
        seed=seed,
        ptdtype=ptdtype,
        device=device,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
    )

    if "rfid" in args.report_fid:
        # run reconstruction eval
        evaluate(args, task_mode="reconstruction", **eval_kwargs)
        dist.barrier()

    if "gfid" in args.report_fid:
        cfg_vals = [
            round(v, 1)
            for v in list(np.arange(args.cfg_min, args.cfg_max, args.cfg_gap))
        ] + [args.cfg_max]

        if not args.use_cfg:
            cfg_vals = [1.0]

        # run generation eval
        for cfg in cfg_vals:
            for step in args.forward_steps:
                evaluate(
                    args,
                    task_mode="generation",
                    forward_steps=step,
                    cfg=cfg,
                    cfg_position=args.cfg_position,
                    use_sampling_scheduler=args.use_sampling_scheduler,
                    cache_sampling_noise=args.cache_sampling_noise,
                    **eval_kwargs,
                )

    # end of eval
    dist.destroy_process_group()
    return


@torch.inference_mode()
def evaluate(
    args,
    task_mode,  # 'generation' or 'reconstruction'
    model,
    loader=None,
    image_size=256,
    use_ema=False,
    forward_steps=1,
    num_classes=0,
    use_sampling_scheduler=False,
    cache_sampling_noise=False,
    cfg=1.0,
    cfg_position="combo",
    save_dir=None,
    tabl_dir=None,
    fid_stats_file_path=None,
    fid_stats_used_from="jit",
    fid_ref_dir=None,
    ckpt_epoch=None,
    save_snapshot=False,
    num_snapshot_samples=128,
    snapshot_save_dir=None,
    seed_sampling=False,
    report_prc=False,
    dataset_name=None,
    device="cuda",
    ptdtype=torch.bfloat16,
    seed=99,
    ddp_rank=0,
    ddp_local_rank=0,
    ddp_world_size=1,
):
    """
    unified evaluation function for both generation and reconstruction
    """
    assert task_mode in ["generation", "reconstruction"]
    assert save_dir is not None
    assert dataset_name is not None

    is_rec = task_mode == "reconstruction"
    sub_fold_name = "recs" if is_rec else "gens"
    suffix_key = "_rec.png" if is_rec else "_gen.png"
    icon = "🥨" if is_rec else "🍺"

    # sub folder name
    save_fold_name = (
        f"imgs"
        f"_px={image_size}"
        f"_pth={ckpt_epoch}"
        f"_ema={use_ema}"
        f"_cfg={cfg}-{cfg_position}"
        f"_steps={forward_steps}"
        f"_sched={use_sampling_scheduler}"
        f"_cache={cache_sampling_noise}"
    )

    if save_snapshot:
        snapshot_root = snapshot_save_dir if snapshot_save_dir else save_dir
        snapshot_img_path = osp.join(snapshot_root, save_fold_name + suffix_key)

    save_dir = osp.join(save_dir, save_fold_name)
    gen_imgs_dir = osp.join(save_dir, sub_fold_name)
    if osp.exists(gen_imgs_dir):
        shutil.rmtree(gen_imgs_dir, ignore_errors=True)
    os.makedirs(gen_imgs_dir, exist_ok=True)
    logger.info(f"save output images to: {gen_imgs_dir}")

    assert args.num_eval_samples % (ddp_world_size * args.batch_size_per_rank) == 0, (
        f"got num_eval_samples={args.num_eval_samples}, "
        f"world_size={ddp_world_size}, "
        f"batch_size_per_rank={args.batch_size_per_rank}"
    )
    num_batches_per_rank = int(
        args.num_eval_samples / ddp_world_size / args.batch_size_per_rank
    )

    class_ids = None
    if num_classes > 0:
        assert args.num_eval_samples % num_classes == 0
        assert model.use_modulation is True
        num_eval_samples_per_class = args.num_eval_samples // num_classes
        class_ids = np.arange(0, num_classes).repeat(num_eval_samples_per_class)
        logger.info(
            f"total classes: {len(class_ids)}, "
            f"samples per class: {num_eval_samples_per_class}"
        )

    dist.barrier()
    logger.info(f"start eval for {task_mode} on {dataset_name} {icon}")

    cnt = 0
    clss = None
    pbar = tqdm(range(num_batches_per_rank), total=num_batches_per_rank)

    for batch_idx in pbar:

        start_idx = (
            batch_idx * args.batch_size_per_rank * ddp_world_size
            + ddp_local_rank * args.batch_size_per_rank
        )
        end_idx = start_idx + args.batch_size_per_rank

        with torch.autocast(device_type="cuda", dtype=ptdtype):
            if is_rec:
                if num_classes > 0:
                    clss = torch.full(
                        (args.batch_size_per_rank,),
                        num_classes,  # empty class
                        dtype=torch.long,
                        device=device,
                    )
                imgs = next(loader)[0].to(device)
                outs = model.reconstruct(imgs, clss, sampling=False)
            else:
                with (
                    torch.random.fork_rng(devices=[device])
                    if seed_sampling
                    else nullcontext()
                ):
                    if seed_sampling:
                        torch.manual_seed(rng.fold_in(seed, ddp_rank, batch_idx))

                    if num_classes > 0:
                        clss = torch.tensor(class_ids[start_idx:end_idx]).to(
                            device=device, dtype=torch.long
                        )

                    # output tuple: gen_imgs_1_step, gen_imgs_n_step
                    _, outs = model.generate(
                        batch_size=args.batch_size_per_rank,
                        y=clss,
                        cfg=cfg,
                        cfg_position=cfg_position,
                        forward_steps=forward_steps,
                        use_sampling_scheduler=use_sampling_scheduler,
                        cache_sampling_noise=cache_sampling_noise,
                        device=device,
                    )

        cnt += outs.shape[0]
        pbar.set_description(f"{task_mode}: generated {cnt} images on rank {ddp_rank}")

        # save individual batch images
        save_image(
            x=outs,
            batch_idx=batch_idx,
            ddp_rank=ddp_rank,
            save_dir=gen_imgs_dir,
            force_image_size=args.image_size,
        )
        torch.cuda.empty_cache()

    # wait for all ranks to finish
    dist.barrier()

    calc_metrics(
        task_mode=task_mode,
        dataset_name=dataset_name,
        image_size=image_size,
        num_eval_samples=args.num_eval_samples,
        gen_imgs_dir=gen_imgs_dir,
        tabl_dir=tabl_dir,
        fid_stats_file_path=fid_stats_file_path,
        fid_stats_used_from=fid_stats_used_from,
        fid_ref_dir=fid_ref_dir,
        ckpt_epoch=ckpt_epoch,
        forward_steps=forward_steps,
        seed_sampling=seed_sampling,
        use_sampling_scheduler=use_sampling_scheduler,
        cache_sampling_noise=cache_sampling_noise,
        use_ema=use_ema,
        report_prc=report_prc,
        cfg=cfg,
        cfg_position=cfg_position,
        ddp_rank0=ddp_rank == 0,
    )

    dist.barrier()

    # save snapshot grid
    if save_snapshot and ddp_rank == 0:
        imgs = glob.glob(osp.join(gen_imgs_dir, "*.png"))
        if len(imgs) > 0:
            imgs = np.random.choice(
                imgs, min(len(imgs), num_snapshot_samples), replace=False
            )
            imgs = [datasets.folder.pil_loader(img) for img in imgs]
            imgs = [torch.from_numpy(np.array(img)) for img in imgs]
            imgs = torch.stack(imgs, dim=0).permute(0, 3, 1, 2) / 255.0  # [B, C, H, W]

            save_tensors_to_images(
                imgs,
                path=snapshot_img_path,
                nrow=max(8, int(num_snapshot_samples / 128 * 8)),
                max_nimgs=num_snapshot_samples,
            )
            logger.info(f"save snapshot image to {snapshot_img_path}")

    dist.barrier()

    if args.rm_folder_after_eval:
        shutil.rmtree(gen_imgs_dir, ignore_errors=True)
        logger.info(f"removed generated images folder: {gen_imgs_dir}")

    dist.barrier()
    return


def calc_metrics(
    task_mode,
    dataset_name,
    image_size,
    num_eval_samples,
    ckpt_epoch,
    gen_imgs_dir,
    tabl_dir,
    use_ema=False,
    report_prc=False,
    fid_stats_file_path=None,
    fid_stats_used_from="jit",
    fid_ref_dir=None,
    forward_steps=1,
    seed_sampling=False,
    use_sampling_scheduler=False,
    cache_sampling_noise=False,
    cfg=1.0,
    cfg_position="combo",
    ddp_rank0=False,
):
    if not ddp_rank0:
        return

    img_fnames = os.listdir(gen_imgs_dir)
    num_imgs = len(img_fnames)
    assert num_imgs == num_eval_samples
    logger.info(f"total number of images to eval: {num_imgs}")

    report_prc = fid_ref_dir is not None and report_prc
    if report_prc:
        ref_imgs_dir = osp.join(
            fid_ref_dir, f"ref_images_{dataset_name}_{image_size}px", "images"
        )
        if not osp.exists(ref_imgs_dir):
            report_prc = False
            logger.warning(
                f"reference images not found: {ref_imgs_dir}, "
                f"skip report precision/recall/f-score"
            )

    # report FID/ISC
    logger.info("start calculating FID/ISC")
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=gen_imgs_dir,
        input2=None,
        fid_statistics_file=fid_stats_file_path,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )
    fid = metrics_dict["frechet_inception_distance"]
    isc_mean = metrics_dict["inception_score_mean"]
    isc_std = metrics_dict["inception_score_std"]

    prc, rcl, fsc = "-", "-", "-"  # N/A
    if report_prc:
        logger.info("start calculating P/R/F-Score")
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=gen_imgs_dir,
            input2=ref_imgs_dir,
            fid_statistics_file=fid_stats_file_path,
            cuda=True,
            isc=False,
            fid=False,
            kid=False,
            prc=True,
            verbose=True,
        )
        prc = metrics_dict["precision"]
        rcl = metrics_dict["recall"]
        fsc = metrics_dict["f_score"]

    # write to table
    headers = [
        "task_mode",
        "use_ema",
        "seed_sampling",
        "use_sampling_scheduler",
        "cache_sampling_noise",
        "fid_stats_used_from",
        "image_size",
        "num_imgs",
        "fid",
        "isc_mean",
        "isc_std",
        "forward_steps",
        "cfg",
        "cfg_position",
    ]
    row = [
        task_mode,
        use_ema,
        seed_sampling,
        use_sampling_scheduler,
        cache_sampling_noise,
        fid_stats_used_from,
        image_size,
        num_imgs,
        fid,
        isc_mean,
        isc_std,
        forward_steps,
        cfg,
        cfg_position,
    ]
    if report_prc:
        headers += ["precision", "recall", "f-score"]
        row += [prc, rcl, fsc]

    log_table = tabulate([row], headers=headers, tablefmt="pipe")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ckpt_path = f"ckpt: {ckpt_epoch}, time: {now}"
    task = "gen" if task_mode == "generation" else "rec"
    file_path = f"{tabl_dir}/eval_tabl_{ckpt_epoch}_{task}_ema={use_ema}.txt"
    with open(file_path, "a") as f:
        f.write("\n" + ckpt_path + "\n-----\n" + log_table + "\n")


def untar_file_from_manifold(src_fold_path, dst_fold_path, tar_name):
    tar_file_path = osp.join(src_fold_path, tar_name)
    os.makedirs(dst_fold_path, exist_ok=True)
    assert osp.exists(tar_file_path), f"{tar_file_path} does not exist"
    shutil.copy(tar_file_path, dst_fold_path)
    shutil.unpack_archive(osp.join(dst_fold_path, tar_name), dst_fold_path)
    logger.info(f"extracted to {dst_fold_path}")


@torch.inference_mode()
def run_to_meaure_flops(args, model, device="cuda", ptdtype=torch.float16):
    from sphere.flops import FvcoreWrapper
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    flops_model = FvcoreWrapper(
        model,
        gen_kwargs={
            "batch_size": 1,
            "y": torch.zeros(1, dtype=torch.long, device=device),
            "cfg": 1.0,
            "cfg_position": "combo",
            "forward_steps": args.flops_steps,
            "device": device,
        },
    )

    with torch.autocast(device_type="cuda", dtype=ptdtype):

        # api needs a sample input, but it won't be used
        dummy_input = torch.randn(1).to(device=device, dtype=ptdtype)
        flops = FlopCountAnalysis(flops_model, dummy_input)
        flops.unsupported_ops_warnings(False)
        gflops = flops.total() / 1e9

    logger.info(f"total GFLOPs: {gflops:.3f}")
    logger.info(flop_count_table(flops, max_depth=2))
    return gflops


if __name__ == "__main__":
    main(cli_args)
