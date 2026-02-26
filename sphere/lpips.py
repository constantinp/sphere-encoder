# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
import logging
from collections import namedtuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_LPIPS_MEAN = [-0.030, -0.088, -0.188]
_LPIPS_STD = [0.458, 0.448, 0.450]

URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}
CKPT_MAP = {"vgg_lpips": "vgg.pth"}
MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def download(url: str, local_path: str, chunk_size: int = 1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path: str):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name: str, root: str, check: bool = False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        logger.info(
            "downloading {} model from {} to {}".format(name, URL_MAP[name], path)
        )
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class vgg16(nn.Module):
    def __init__(
        self,
        requires_grad: bool = False,
        pretrained: bool = True,
    ):
        super(vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(
            pretrained=pretrained
        ).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5

        # build feature slices
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "vgg_outputs",
            ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"],
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LPIPS(nn.Module):
    def __init__(
        self,
        ckpt_pth="work_dirs/ckpts/lpips",
        use_dropout=True,
        chns_range=(0, 5),
    ):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vgg16 features
        assert 0 <= chns_range[0] < chns_range[1] <= len(self.chns)
        self.lidx, self.ridx = chns_range
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.load_from_pretrained(ckpt_pth=ckpt_pth)
        for param in self.parameters():
            param.requires_grad = False

        self._data_range_checked = False

    def load_from_pretrained(self, ckpt_pth="work_dirs/ckpts/lpips", name="vgg_lpips"):
        if ckpt_pth is None:
            raise ValueError("no pretrained weights found for LPIPS loss.")
        ckpt = get_ckpt_path(name, ckpt_pth, check=True)
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        logger.info("loaded pretrained LPIPS loss from {}".format(ckpt))

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        inps = self.net(self.scaling_layer(input))
        tgts = self.net(self.scaling_layer(target))

        lins = self.lins[self.lidx : self.ridx]
        chns = self.chns[self.lidx : self.ridx]
        inps = inps[self.lidx : self.ridx]
        tgts = tgts[self.lidx : self.ridx]

        vals = []
        for k in range(len(chns)):
            f0 = inps[k]
            f1 = tgts[k]

            f0 = f0 * torch.rsqrt(torch.sum(f0**2, dim=1, keepdim=True) + 1e-10)
            f1 = f1 * torch.rsqrt(torch.sum(f1**2, dim=1, keepdim=True) + 1e-10)

            diff = lins[k].model((f0 - f1) ** 2)
            diff = diff.mean([2, 3], keepdim=True)

            vals.append(diff)
        return torch.stack(vals, dim=0).sum(dim=0)


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_s",
        ckpt_path: str = "",
        perceptual_loss_chns_range: int = 10,
    ):
        super().__init__()
        assert "lpips" in model_name or "convnext_s" in model_name
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None
        self._data_range_checked = False

        # Parsing the model name. We support name formatted in
        # "lpips-convnext_s-{float_number}-{float_number}", where the
        # {float_number} refers to the loss weight for each component.
        # E.g., lpips-convnext_s-1.0-2.0 refers to compute the perceptual loss
        # using both the convnext_s and lpips, and average the final loss with
        # (1.0 * loss(lpips) + 2.0 * loss(convnext_s)) / (1.0 + 2.0).
        if "lpips" in model_name:
            self.lpips = LPIPS(
                ckpt_pth=ckpt_path, chns_range=perceptual_loss_chns_range
            ).eval()

        if "convnext_s" in model_name:
            self.convnext = torchvision.models.convnext_small(
                weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            ).eval()

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split("-")[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(
                loss_config[0]
            ), float(loss_config[1])
            logger.info(
                f"loss weights - lpips: {self.loss_weight_lpips}, convnext: {self.loss_weight_convnext}"
            )

        if self.loss_weight_convnext is None:
            self.loss_weight_convnext = 1.0

        if self.loss_weight_lpips is None:
            self.loss_weight_lpips = 1.0

        self.register_buffer(
            "imagenet_mean",
            torch.Tensor(_IMAGENET_MEAN).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.Tensor(_IMAGENET_STD).reshape(1, 3, 1, 1),
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        assert input.shape == target.shape, f"{input.shape=} != {target.shape}="

        if not self._data_range_checked:
            assert (
                target.min() >= 0.0 and target.max() <= 1.0
            ), f"{target.min()=} ~ {target.max()=}. reminder to normalize input and target to [0, 1]."
            self._data_range_checked = True

        self.eval()
        inp, tgt = input, target
        loss = 0.0
        num_losses = 0.0

        if self.lpips is not None:
            lpips_loss = self.lpips(inp * 2 - 1, tgt * 2 - 1)  # [0, 1] -> [-1, 1]
            loss += self.loss_weight_lpips * lpips_loss
            num_losses += self.loss_weight_lpips

        if self.convnext is not None:
            inp = F.interpolate(inp, size=224, mode="bilinear", antialias=True)
            tgt = F.interpolate(tgt, size=224, mode="bilinear", antialias=True)
            inp_logits = self.convnext((inp - self.imagenet_mean) / self.imagenet_std)
            tgt_logits = self.convnext((tgt - self.imagenet_mean) / self.imagenet_std)

            convnext_loss = F.mse_loss(inp_logits, tgt_logits, reduction="mean")
            loss += self.loss_weight_convnext * convnext_loss
            num_losses += self.loss_weight_convnext

        return loss / num_losses


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor(_LPIPS_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("scale", torch.Tensor(_LPIPS_STD).reshape(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)
