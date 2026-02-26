# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import sys

import numpy as np
import torch
from rich.logging import RichHandler


def setup_logging(output_path, name=None, rank=0):
    logging.captureWarnings(True)

    logger = logging.getLogger(name)
    logger.propagate = False

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)  # Other GPUs only log warnings/errors

    # google glog format: [IWEF]yyyymmdd hh:mm:ss logger filename:line] msg
    fmt_prefix = "%(levelname).1s%(asctime)s %(name)s %(filename)s:%(lineno)s] "
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # stdout logging for main worker only
    if rank != 0:
        return

    if sys.stdout.isatty():
        handler = RichHandler(markup=False, show_time=False, show_level=False)
    else:
        handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # file logging
    if output_path is not None:
        if os.path.splitext(output_path)[-1] in (".txt", ".log"):
            # if output_path is a file, use it directly
            filename = output_path
        else:
            # if output_path is a directory, use the directory/log.txt
            filename = os.path.join(output_path, "log.txt")

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def append_log(file_path, entry: dict):
    d = {}
    for k, v in entry.items():
        if isinstance(v, torch.Tensor):
            v = v.data.cpu().float().numpy()
        else:
            v = float(v)

        d[k] = str(np.round(v, 5))

    with open(file_path, "a") as f:
        f.write(json.dumps(d) + "\n")
