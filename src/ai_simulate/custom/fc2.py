from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.library import Library


_LIB = Library("custom", "DEF")
_LIB.define("fc2(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor")


def _fc2_meta(x, weight, bias=None):
    out_shape = list(x.shape[:-1]) + [weight.shape[0]]
    return x.new_empty(out_shape, device="meta")


def _fc2_impl(x, weight, bias=None):
    output = x.reshape(-1, x.shape[-1]).matmul(weight.t())
    if bias is not None:
        output = output + bias
    return output.reshape(*x.shape[:-1], weight.shape[0])


_LIB.impl("fc2", _fc2_meta, "Meta")
_LIB.impl("fc2", _fc2_impl, "CPU")


class CustomFC2(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x):
        return torch.ops.custom.fc2(x, self.weight, self.bias)


def fc2(x, weight, bias=None):
    return torch.ops.custom.fc2(x, weight, bias)
