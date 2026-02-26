# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib


def fold_in(seed: int, *args):
    """
    args can be anything hashable: step, rank, etc.
    """
    data = str((seed,) + args)
    h = hashlib.sha256(data.encode("utf-8")).hexdigest()
    folded_seed = int(h, 16) % (2**63)
    return folded_seed
