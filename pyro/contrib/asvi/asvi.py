# Copyright (c) 2017-2023 Eli Sennesh
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import torch
from torch.distributions import biject_to, transform_to

import pyro.distributions as dist
from pyro.distributions.distribution import Distribution
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr, helpful_support_errors, _product
from pyro.nn.module import PyroModule, PyroParam
from pyro.poutine.guide import GuideMessenger
from pyro.poutine.handlers import _make_handler
from pyro.poutine.runtime import get_conditions

from .util import dist_params, mlp_amortizer

class AsviMessenger(GuideMessenger):
    def __init__(self, model: Callable, module: PyroModule):
        super().__init__(model)
        self.module = module

    def _get_params(self, name: str, prior: Distribution):
        try:
            prior_logits = deep_getattr(self.module.prior_logits, name)
            mean_fields = {deep_getattr(self.module.mean_fields, name + "." + k)
                           for k, v in dist_params(prior).items()}
            return prior_logits, mean_fields
        except AttributeError:
            pass

        # Initialize
        for k, v in dist_params(prior).items():
            with torch.no_grad():
                transform = biject_to(prior.support)
                event_dim = transform.domain.event_dim
                unconstrained = torch.zeros_like(v)
                unconstrained = self._adjust_plates(unconstrained, event_dim)
            deep_setattr(self, "module.mean_fields." + name + "." + k,
                         PyroParam(unconstrained, event_dim=event_dim))
        with torch.no_grad():
            event_dim_start = len(v.size()) - len(prior.event_shape)
            prior_logits = torch.zeros(v.shape[:event_dim_start],
                                       device=v.device)
            prior_logits = self._adjust_plates(prior_logits, event_dim)
        deep_setattr(self, "module.prior_logits." + name,
                     PyroParam(prior_logits))

        return self._get_params(name, prior)

    def get_posterior(self, name: str, prior: Distribution) -> Distribution:
        with helpful_support_errors({"name": name, "fn": prior}):
            transform = biject_to(prior.support)
            event_shape = prior.event_shape
            if isinstance(prior, dist.Independent):
                independent_ndims = prior.reinterpreted_batch_ndims
                prior = prior.base_dist
            else:
                independent_ndims = 0
            prior_params = dist_params(prior)
        alphas, lamdas = self._get_params(name, prior)
        alphas = torch.sigmoid(alphas)
        alphas = alphas.reshape(alphas.shape + (1,) * len(event_shape))
        for k, v in prior_params.items():
            param_transform = transform_to(prior.arg_constraints[k])
            lam = param_transform(lamdas[k])
            prior_params[k] = alphas * v + (1 - alphas) * lam

        proposal = prior.__class__(**prior_params)
        if independent_ndims:
            proposal = dist.Independent(proposal, independent_ndims)
        posterior = dist.TransformedDistribution(proposal,
                                                 [transform.with_cache()])
        return posterior

class NeuralAsviMessenger(AsviMessenger):
    def __init__(
        self,
        model: Callable,
        module: PyroModule,
        obs_name: str,
        event_shape: torch.Size,
        *,
        init_amortizer: Callable = mlp_amortizer,
    ):
        AsviMessenger.__init__(self, model, module)
        if not callable(init_amortizer):
            raise ValueError("Expected callable to construct ASVI amortizers")
        self._obs_name = obs_name
        self._event_shape = event_shape
        self._init_amortizer = init_amortizer

    def _get_params(self, name: str, prior: Distribution):
        conditions = get_conditions()
        data = {}
        for condition in conditions:
            data = data | condition
        try:
            obs = data[self._obs_name]
            batch_shape = obs.shape[:-len(self._event_shape)]
        except KeyError:
            raise KeyError("Expected observation %s on which to condition amortized ASVI" % self._obs_name)

        try:
            obs = obs.view(*batch_shape, *self._event_shape)
            prior_logits = deep_getattr(self.module.prior_logits, name)(obs).squeeze()
            mean_fields = {k: deep_getattr(self.module.mean_fields, name + "." + k)(obs)
                           for k, v in dist_params(prior).items()}
            return prior_logits, mean_fields
        except AttributeError:
            pass

        # Initialize amortizer nets
        for k, v in dist_params(prior).items():
            v_dim = _product(v.shape[len(batch_shape):])
            amortizer = self._init_amortizer(_product(self._event_shape), v_dim)
            deep_setattr(self, "module.mean_fields." + name + "." + k,
                         amortizer.to(device=v.device))
        amortizer = self._init_amortizer(_product(self._event_shape), 1)
        deep_setattr(self, "module.prior_logits." + name,
                     amortizer.to(device=v.device))

        return self._get_params(name, prior)

_msngrs = [
    AsviMessenger,
    NeuralAsviMessenger,
]

for _msngr_cls in _msngrs:
    _handler_name, _handler = _make_handler(_msngr_cls)
    _handler.__module__ = __name__
    locals()[_handler_name] = _handler
