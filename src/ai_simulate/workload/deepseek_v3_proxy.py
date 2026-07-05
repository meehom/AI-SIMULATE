from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn

from ai_simulate.custom import CustomFC2


DEFAULT_DEEPSEEK_V3_PROXY_CONFIG = {
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "activation": "gelu",
    "norm_eps": 1e-5,
}


@dataclass(frozen=True)
class ProxyInputSpec:
    shape: List[int]
    hidden_size: int
    intermediate_size: int
    activation: str
    analysis_phase: str


class DeepSeekV3ProxyMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = self._build_activation(activation)
        self.cus_fc2 = CustomFC2(intermediate_size, hidden_size)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        activation_name = name.lower()
        if activation_name == "gelu":
            return nn.GELU()
        if activation_name == "relu":
            return nn.ReLU()
        raise ValueError(f"Unsupported activation for DeepSeek V3 proxy: {name}")

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        return self.cus_fc2(x)


def build_deepseek_v3_proxy(
    workload_config: Dict[str, Any],
    phase: str = "prefill",
) -> tuple[DeepSeekV3ProxyMLP, ProxyInputSpec]:
    if phase != "prefill":
        raise ValueError(f"Unsupported analysis phase for v1 proxy: {phase}")

    hidden_size = int(workload_config.get("proxy_hidden_size", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["hidden_size"]))
    intermediate_size = int(
        workload_config.get("proxy_intermediate_size", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["intermediate_size"])
    )
    activation = str(workload_config.get("proxy_activation", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["activation"]))
    norm_eps = float(workload_config.get("proxy_norm_eps", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["norm_eps"]))

    batch_size = int(workload_config["global_batch_size"])
    input_seq_len = int(workload_config["input_seq_len"])

    with torch.device("meta"):
        model = DeepSeekV3ProxyMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            norm_eps=norm_eps,
        )

    input_spec = ProxyInputSpec(
        shape=[batch_size, input_seq_len, hidden_size],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        analysis_phase=phase,
    )
    return model, input_spec
