from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn

from ai_simulate.custom import CustomFC2


DEFAULT_DEEPSEEK_V3_PROXY_CONFIG = {
    "vocab_size": 102400,
    "num_layers": 1,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "num_attention_heads": 56,
    "activation": "gelu",
    "norm_eps": 1e-5,
}


@dataclass(frozen=True)
class ProxyInputSpec:
    shape: List[int]
    dtype: str
    input_kind: str
    hidden_size: int
    intermediate_size: int
    activation: str
    num_layers: int
    analysis_phase: str


class DeepSeekV3ProxySelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_attention_heads}"
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size, seq_len, hidden_size = q.shape
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(context)


class DeepSeekV3ProxyDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        activation: str = "gelu",
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = DeepSeekV3ProxySelfAttention(hidden_size, num_attention_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.activation = DeepSeekV3ProxyDecoderBlock._build_activation(activation)
        self.down_proj = CustomFC2(intermediate_size, hidden_size)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        activation_name = name.lower()
        if activation_name == "gelu":
            return nn.GELU()
        if activation_name == "relu":
            return nn.ReLU()
        raise ValueError(f"Unsupported activation for DeepSeek V3 proxy: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.down_proj(self.activation(self.up_proj(self.norm2(x))))
        return x


class DeepSeekV3ProxyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        activation: str = "gelu",
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                DeepSeekV3ProxyDecoderBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    activation=activation,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=norm_eps)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


def build_deepseek_v3_proxy(
    workload_config: Dict[str, Any],
    phase: str = "prefill",
) -> tuple[DeepSeekV3ProxyModel, ProxyInputSpec]:
    if phase != "prefill":
        raise ValueError(f"Unsupported analysis phase for v1 proxy: {phase}")

    vocab_size = int(workload_config.get("proxy_vocab_size", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["vocab_size"]))
    num_layers = int(workload_config.get("proxy_num_layers", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["num_layers"]))
    hidden_size = int(workload_config.get("proxy_hidden_size", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["hidden_size"]))
    intermediate_size = int(
        workload_config.get("proxy_intermediate_size", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["intermediate_size"])
    )
    num_attention_heads = int(
        workload_config.get(
            "proxy_num_attention_heads",
            DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["num_attention_heads"],
        )
    )
    activation = str(workload_config.get("proxy_activation", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["activation"]))
    norm_eps = float(workload_config.get("proxy_norm_eps", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["norm_eps"]))

    batch_size = int(workload_config["global_batch_size"])
    input_seq_len = int(workload_config["input_seq_len"])

    with torch.device("meta"):
        model = DeepSeekV3ProxyModel(
            vocab_size=vocab_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            activation=activation,
            norm_eps=norm_eps,
        )

    input_spec = ProxyInputSpec(
        shape=[batch_size, input_seq_len],
        dtype="int64",
        input_kind="token_ids",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        num_layers=num_layers,
        analysis_phase=phase,
    )
    return model, input_spec
