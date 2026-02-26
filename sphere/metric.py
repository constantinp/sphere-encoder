# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_metric_feature_extractor(device="cuda"):
    # load InceptionV3
    # https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_inceptionv3.py
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    ckpt_path = "/mnt/wsfuse/kaiyuyue/pretrained_models/inception-v3/weights-inception-2015-12-05-6726825d.pth"
    activation_dim = 2048

    model = FeatureExtractorInceptionV3(
        # name from https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/registry.py#L172
        name="inception-v3-compat",
        features_list=[str(activation_dim), "logits_unbiased"],  # [for fid, for isc]
        feature_extractor_weights_path=ckpt_path,
        feature_extractor_internal_dtype="float32",
    )
    model.to(device).eval().requires_grad_(False)
    return model, activation_dim


def extract_metric_features(loader, model, activation_dim, device="cuda"):
    # init
    features = []
    logits_unbiased = []

    # loop
    pbar = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, batch_data in pbar:

        imgs, clss = batch_data[:2]  # FIXME: ignore the rest
        imgs = imgs.to(device, non_blocking=True)
        clss = clss.to(device, non_blocking=True)

        # check
        assert torch.all(imgs >= -1) and torch.all(
            imgs <= 1
        ), "input values are out of range [-1, 1]"

        # shift pixel values from [-1, 1] to [0, 1]
        imgs = imgs * 0.5 + 0.5
        imgs = torch.clamp(imgs, min=0.0, max=1.0)

        # bring back to [0, 255] uint8
        imgs = imgs * 255.0
        imgs = torch.floor(imgs).to(torch.uint8)

        # forward
        x = model(imgs)
        assert len(x) == 2

        # to np
        feat = x[0].data.cpu().numpy()
        logt = x[1].data.cpu().numpy()

        # append
        features.append(feat)
        logits_unbiased.append(logt)

    # stack
    logits_unbiased = np.concatenate(logits_unbiased, axis=0)
    features = np.concatenate(features, axis=0)
    logger.info(f"metric feature tensor shape: {features.shape}")

    # calc mu
    mu = np.mean(features, axis=0)
    assert mu.shape == (activation_dim,)

    # calc sigma
    sigma = np.cov(features, rowvar=False)
    assert sigma.shape == (activation_dim, activation_dim)

    return mu, sigma, features, logits_unbiased


def compute_fid(mu1, mu2, sigma1, sigma2):
    assert torch.is_tensor(mu1) and torch.is_tensor(mu2)
    assert torch.is_tensor(sigma1) and torch.is_tensor(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)
    fid = a + b - 2 * c
    return fid.item()


def compute_isc(features, splits=10):
    assert torch.is_tensor(features) and features.ndim == 2

    # random shuffle rows
    idx = torch.randperm(features.shape[0])
    features = features[idx]

    # calc prob and logits
    prob = features.softmax(dim=1)
    log_prob = features.log_softmax(dim=1)

    # chunk into groups
    prob = prob.chunk(splits, dim=0)
    log_prob = log_prob.chunk(splits, dim=0)

    # calculate score per split
    mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
    kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
    kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
    kl = torch.stack(kl_)

    return kl.mean().item(), kl.std().item()


def compute_prc(features_gen, features_ref, neighborhood=3):
    return
