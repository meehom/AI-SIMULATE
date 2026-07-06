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
    "activation": "silu",
    "norm_eps": 1e-5,
    "use_moe": True,
    "num_experts": 8,
    "top_k": 2,
    "expert_intermediate_size": 18432,
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


class DeepSeekV3ProxyMoEExpert(nn.Module):
    def __init__(self, hidden_size: int, expert_intermediate_size: int, activation: str = "silu") -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, expert_intermediate_size)
        self.up_proj = nn.Linear(hidden_size, expert_intermediate_size)
        self.activation = DeepSeekV3ProxyDecoderBlock._build_activation(activation)
        self.down_proj = CustomFC2(expert_intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class DeepSeekV3ProxyMoEFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k={top_k} must satisfy 0 < top_k <= num_experts={num_experts}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [
                DeepSeekV3ProxyMoEExpert(
                    hidden_size=hidden_size,
                    expert_intermediate_size=expert_intermediate_size,
                    activation=activation,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_values, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        gates = torch.zeros_like(router_probs).scatter(-1, topk_indices, topk_values)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        return (expert_outputs * gates.unsqueeze(-1)).sum(dim=2)


class DeepSeekV3ProxyDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        activation: str = "silu",
        norm_eps: float = 1e-5,
        use_moe: bool = True,
        num_experts: int = 8,
        top_k: int = 2,
        expert_intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = DeepSeekV3ProxySelfAttention(hidden_size, num_attention_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        if use_moe:
            self.ffn = DeepSeekV3ProxyMoEFFN(
                hidden_size=hidden_size,
                expert_intermediate_size=expert_intermediate_size or intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                activation=activation,
            )
        else:
            self.ffn = DeepSeekV3ProxyMoEExpert(
                hidden_size=hidden_size,
                expert_intermediate_size=intermediate_size,
                activation=activation,
            )

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        activation_name = name.lower()
        if activation_name == "gelu":
            return nn.GELU()
        if activation_name == "relu":
            return nn.ReLU()
        if activation_name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation for DeepSeek V3 proxy: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DeepSeekV3ProxyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        activation: str = "silu",
        norm_eps: float = 1e-5,
        use_moe: bool = True,
        num_experts: int = 8,
        top_k: int = 2,
        expert_intermediate_size: int | None = None,
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
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k,
                    expert_intermediate_size=expert_intermediate_size,
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
    use_moe = bool(workload_config.get("proxy_use_moe", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["use_moe"]))
    num_experts = int(workload_config.get("proxy_num_experts", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["num_experts"]))
    top_k = int(workload_config.get("proxy_top_k", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["top_k"]))
    expert_intermediate_size = int(
        workload_config.get(
            "proxy_expert_intermediate_size",
            DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["expert_intermediate_size"],
        )
    )

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
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            expert_intermediate_size=expert_intermediate_size,
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
