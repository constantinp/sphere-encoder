# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sphere.lpips import PerceptualLoss
from sphere.utils import vector_compute_magnitude
from torch import Tensor


def l1_loss(x, y, reduction="mean"):
    return F.smooth_l1_loss(x, y, reduction=reduction)


def l2_loss(x, y, reduction="mean"):
    return F.mse_loss(x, y, reduction=reduction)


def cos_sim(x, y, reduction="mean"):
    x = x.flatten(start_dim=1)
    y = y.flatten(start_dim=1)
    return 1.0 - F.cosine_similarity(x, y, dim=1, eps=1e-6)


class DistLoss(nn.Module):
    def __init__(self, distance="l2", reduction="mean"):
        super().__init__()
        self.fn = {"l1": l1_loss, "l2": l2_loss, "cosine": cos_sim}[distance]
        self.reduction = reduction

    def forward(self, input, target):
        x, y = input, target
        assert x.shape == y.shape
        v = self.fn(x, y, reduction="none")  # [B, ...]
        if v.ndim > 1:
            v = v.mean(dim=tuple(range(1, x.ndim)), keepdim=True)  # [B]
        return v.mean() if self.reduction == "mean" else v


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        perceptual_loss: str = "lpips-convnext_s-1.0-0.1",
        perceptual_ckpt_path: str = "",
        perceptual_weight: float = 1.0,
        perceptual_loss_chns_range: tuple[int, int] = (0, 5),
        distance_loss_type: str = "l2",
        distance_weight: float = 1.0,
        pixel_consistency_distance_weight: float = 0.0,
        pixel_consistency_perceptual_weight: float = 0.0,
        latent_consistency_weight: float = 0.0,
    ):
        super().__init__()
        self.distance_loss_type = distance_loss_type
        assert self.distance_loss_type in ["l1", "l2", "l2+l1", "l1+l2"]
        self.distance_weight = distance_weight

        self.perceptual_loss = PerceptualLoss(
            model_name=perceptual_loss,
            ckpt_path=perceptual_ckpt_path,
            perceptual_loss_chns_range=perceptual_loss_chns_range,
        )
        self.perceptual_loss.eval().requires_grad_(False)
        self.perceptual_weight = perceptual_weight

        self.pix_con_dist_weight = pixel_consistency_distance_weight
        self.pix_con_perc_weight = pixel_consistency_perceptual_weight
        self.lat_con_weight = latent_consistency_weight

        if self.lat_con_weight > 0:
            self.lat_con_loss_fn = DistLoss(distance="cosine")

        self.log_dict = {}
        self._data_range_checked = False

    def compute_distance_loss(self, x, y, dim):
        if self.distance_loss_type == "l1":
            v = l1_loss(x, y, reduction="none")

        elif self.distance_loss_type == "l2":
            v = l2_loss(x, y, reduction="none")

        elif self.distance_loss_type in ["l2+l1", "l1+l2"]:
            a = l1_loss(x, y, reduction="none")
            b = l2_loss(x, y, reduction="none")
            v = 0.5 * a + 0.5 * b

        v = v.mean(dim=dim, keepdim=True)  # [B, ...]
        return v

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        epoch: int = 0,
        extra_loss: dict = {},
        noisy_latent=None,
        clean_latent=None,
    ):
        # cast to fp32
        inp, tgt = input.float().contiguous(), target.float().contiguous()

        if noisy_latent is not None:
            noisy_latent = noisy_latent.float().contiguous()

        if clean_latent is not None:
            clean_latent = clean_latent.float().contiguous()

        # basic info
        dim = tuple(range(1, inp.ndim))  # w/o batch dimension
        device = target.device

        # pixel consistency loss
        pix_con_dist_loss = torch.zeros((), device=device)
        pix_con_perc_loss = torch.zeros((), device=device)

        if self.pix_con_dist_weight > 0 or self.pix_con_perc_weight > 0:
            assert inp.shape[0] == 2 * tgt.shape[0]

            inp_NOISY, inp_noisy = inp.chunk(chunks=2, dim=0)

            # 1. on this loss path, the real target is only for the output of the clean latent
            #    or the small noisy latent (anchor)
            # 2. the output of the large NOISY latent is distilled by the output of the clean or
            #    small noisy latent (bootstrap)
            tgt_NOISY = inp_noisy.clone().detach()  # stop gradient
            inp = inp_noisy

            # calc pixel consistency loss (distance + perceptual)
            _dist_loss = self.compute_distance_loss(inp_NOISY, tgt_NOISY, dim)
            pix_con_dist_psnr = -10 * torch.log10(_dist_loss.clone().detach().mean())
            _dist_w = self.pix_con_dist_weight
            pix_con_dist_loss = _dist_w * _dist_loss.mean(dim=dim).mean()

            _perc_loss = self.perceptual_loss(inp_NOISY, tgt_NOISY)
            _perc_w = self.pix_con_perc_weight
            pix_con_perc_loss = _perc_w * _perc_loss.mean(dim=dim).mean()

        # check shapes
        assert tgt.shape == inp.shape, f"target {tgt.shape} != input {inp.shape}"

        # check value ranges
        if not self._data_range_checked:
            assert torch.all((tgt >= 0) & (tgt <= 1)), "values not in [0, 1]"
            self._data_range_checked = True

        # calc l1/l2 loss
        dist_loss = self.compute_distance_loss(inp, tgt, dim)  # [B, ...]
        dist_psnr = -10 * torch.log10(dist_loss.clone().detach().mean())
        dist_loss = self.distance_weight * dist_loss.mean(dim=dim).mean()

        # calc perceptual loss
        perc_loss = self.perceptual_loss(inp, tgt)
        perc_loss = self.perceptual_weight * perc_loss.mean(dim=dim).mean()

        # calc latent bootstrap loss
        if (
            self.lat_con_weight > 0
            and noisy_latent is not None
            and clean_latent is not None
        ):
            lat_con_loss = self.lat_con_loss_fn(noisy_latent, clean_latent.detach())
            lat_con_loss = self.lat_con_weight * lat_con_loss

        # calc total loss
        total_loss = (
            dist_loss + perc_loss + pix_con_dist_loss + pix_con_perc_loss + lat_con_loss
        )
        self.log_dict.update(
            {
                "total_loss": total_loss.clone().detach(),
                "dist_loss": dist_loss.clone().detach(),
                "perc_loss": perc_loss.clone().detach(),
                "dist_psnr": dist_psnr.detach(),
            }
        )

        if self.pix_con_dist_weight > 0:
            self.log_dict["pix_con_dist_loss"] = pix_con_dist_loss.clone().detach()
            self.log_dict["pix_con_dist_psnr"] = pix_con_dist_psnr

        if self.pix_con_perc_weight > 0:
            self.log_dict["pix_con_perc_loss"] = pix_con_perc_loss.clone().detach()

        if self.lat_con_weight > 0:
            self.log_dict["lat_con_loss"] = lat_con_loss.clone().detach()

        return total_loss


class SWDLoss(nn.Module):
    """
    sliced wasserstein distance (SWD) loss
    """

    def __init__(
        self,
        input,
        normalized_k=None,
        num_projections=128,
        norm_fn=None,
    ):
        super().__init__()
        self.inp = input
        self.f = norm_fn
        self.N = num_projections
        self.k = normalized_k

    @torch.autocast("cuda", enabled=True)
    def reduce_swd_loss(self, p=2, device="cuda"):
        assert self.inp.ndim in [3, 4]  # [B, C, H, W] or [B, N, D]
        dim = tuple(range(1, self.inp.ndim))  # w/o batch dimension

        if self.k is not None:
            inp = torch.cat([self.inp, self.k], dim=0)  # [K 16384 + B, ...]
        else:
            inp = self.inp

        tgt = torch.randn_like(inp)
        tgt = self.f(tgt) if self.f is not None else tgt

        # mag_z = vector_compute_magnitude(self.inp)
        # mag_e = vector_compute_magnitude(tgt)
        # print(inp.shape, tgt.shape, mag_z.mean(), mag_e.mean())

        # sample random projections (random lines in space)
        P = torch.randn((self.N, math.prod(inp.shape[1:])), device=device)
        # use ortho projections
        Q = torch.linalg.qr(P.T)[0]
        P = Q.T
        P = P.reshape(self.N, *inp.shape[1:])
        P = F.normalize(P, p=2, dim=dim)

        # proj input and target onto the random lines
        inp = torch.tensordot(inp, P, dims=[dim, dim])  # [B, N]
        tgt = torch.tensordot(tgt, P, dims=[dim, dim])  # [B, N]

        # sort projections along the batch dimension
        # unpaired set-to-set matching
        inp, _ = torch.sort(inp, dim=0)
        tgt, _ = torch.sort(tgt, dim=0)

        # calc the distance between the sorted distributions
        diff = torch.abs(inp - tgt)
        loss = torch.pow(diff, p).mean()
        return loss.float()
