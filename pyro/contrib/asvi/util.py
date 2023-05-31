# Copyright (c) 2017-2023 Eli Sennesh
# SPDX-License-Identifier: Apache-2.0

import typing

import torch.nn as nn
from pyro.distributions.distribution import Distribution

def dist_params(dist: Distribution):
    return {k: v for k, v in dist.__dict__.items() if k[0] != '_'}

def mlp_amortizer(dom, cod):
    return nn.Sequential(
        nn.Linear(dom, cod * 2),
        nn.ReLU(),
        nn.Linear(cod * 2, cod)
    )
