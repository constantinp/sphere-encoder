# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class TokenMixer(nn.Module):

    def __init__(self, num_tokens):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Linear(num_tokens, num_tokens),
            nn.SiLU(),
            nn.Linear(num_tokens, num_tokens),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, D, N]
        x = self.mix(x)
        x = x.transpose(1, 2)  # [B, N, D]
        return x


class ChannelMixer(nn.Module):

    def __init__(self, dim, expansion_factor=2):
        super().__init__()
        _dim = dim * expansion_factor
        self.mix = nn.Sequential(
            nn.Linear(dim, _dim),
            nn.SiLU(),
            nn.Linear(_dim, dim),
        )

    def forward(self, x):
        return self.mix(x)  # [B, N, D]


class MixerBlock(nn.Module):

    def __init__(self, num_tokens, dim, expansion_factor=2):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.tok_mixer = TokenMixer(num_tokens)

        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.chn_mixer = ChannelMixer(dim, expansion_factor=expansion_factor)

        self.alpha1 = nn.Parameter(torch.tensor([0.0]))
        self.alpha2 = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        x = x + self.alpha1 * self.tok_mixer(self.norm1(x))
        x = x + self.alpha2 * self.chn_mixer(self.norm2(x))
        return x


class MLPMixer(nn.Module):

    def __init__(
        self,
        num_tokens,
        dim,
        depth=2,
        expansion_factor=2,
        use_affine=True,
        norm_layer=None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MixerBlock(num_tokens, dim, expansion_factor=expansion_factor)
                for _ in range(depth)
            ]
        )
        self.norm = nn.RMSNorm(dim, elementwise_affine=use_affine, eps=1e-6)
        if norm_layer is not None:
            self.norm = norm_layer(dim)

    def forward(self, x):
        _dtype = x.dtype
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.to(_dtype)
