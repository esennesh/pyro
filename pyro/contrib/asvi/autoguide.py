# Copyright (c) 2017-2023 Eli Sennesh
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Tuple

import torch

from pyro.infer.autoguide import AutoMessenger

# from .asvi import AsviMessenger, NeuralAsviMessenger
from .asvi import AsviMessenger
from .util import mlp_amortizer

class AutoAsviMessenger(AutoMessenger,AsviMessenger):
    def __init__(
        self,
        model: Callable,
        data: torch.Tensor,
        event_shape: torch.Size,
        *,
        init_amortizer: Callable = mlp_amortizer,
        amortized_plates: Tuple[str, ...] = (),
    ):
        if not callable(init_amortizer):
            raise ValueError("Expected callable to construct ASVI amortizers")
        super(AutoMessenger, self).__init__(model,
                                            amortized_plates=amortized_plates)
        super(NeuralAsviMessenger, self).__init__(
            self, data, event_shape, init_amortizer=init_amortizer
        )
