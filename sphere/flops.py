# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class FvcoreWrapper(nn.Module):
    def __init__(self, model, gen_kwargs):
        super().__init__()
        self.model = model
        self.gen_kwargs = gen_kwargs

    def forward(self, dummy_input):
        # fvcore passes 'dummy_input' here, but we ignore it.
        # We trigger the exact inference logic you want to measure.
        return self.model.generate(**self.gen_kwargs)
