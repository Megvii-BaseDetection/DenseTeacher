#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn

__all__ = ["ModelEMA", "is_parallel"]


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
        """
        # Create EMA(FP32)
        self.model = deepcopy(model.module if is_parallel(model) else model).eval()
        # decay exponential ramp (to help early epochs)
        # self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model, decay=None):
        if decay is None:
            decay = self.decay
        # Update EMA parameters
        with torch.no_grad():
            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )  # model state_dict
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay
                    v += (1.0 - decay) * msd[k].detach()
