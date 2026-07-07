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
    "attention_impl": "mla",
    "mla_q_lora_rank": 1536,
    "mla_kv_lora_rank": 512,
    "mla_qk_nope_head_dim": 64,
    "mla_qk_rope_head_dim": 64,
    "mla_v_head_dim": 128,
    "rope_theta": 10000.0,
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
    kv_cache_seq_len: int | None = None


class DeepSeekV3ProxyMLAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_theta: float = 10000.0,
        phase: str = "prefill",
        decode_kv_cache_len: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.phase = phase
        self.decode_kv_cache_len = decode_kv_cache_len
        self.q_latent_proj = nn.Linear(hidden_size, q_lora_rank)
        self.q_nope_proj = nn.Linear(q_lora_rank, num_attention_heads * qk_nope_head_dim)
        self.q_rope_proj = nn.Linear(q_lora_rank, num_attention_heads * qk_rope_head_dim)
        self.kv_latent_proj = nn.Linear(hidden_size, kv_lora_rank)
        self.k_nope_proj = nn.Linear(kv_lora_rank, num_attention_heads * qk_nope_head_dim)
        self.v_proj = nn.Linear(kv_lora_rank, num_attention_heads * v_head_dim)
        self.k_rope_proj = nn.Linear(hidden_size, qk_rope_head_dim)
        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size)

        # Pre-absorbed decode weights (precomputed offline in a real engine).
        # q_nope_absorb folds W_k_nope into the query so scores are computed against
        # the compressed kv latent directly: (q_nope @ W_k_nope) @ kv_latent^T.
        self.q_nope_absorb = nn.Parameter(
            torch.empty(1, num_attention_heads, qk_nope_head_dim, kv_lora_rank)
        )
        # vo_absorb folds W_v @ W_o into one matrix mapping the latent-space attention
        # output straight to hidden: (probs @ kv_latent) @ (W_v @ W_o).
        self.vo_absorb = nn.Parameter(
            torch.empty(1, num_attention_heads, kv_lora_rank, hidden_size)
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _apply_rope(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[-2]
        dim = x.shape[-1]
        positions = torch.arange(offset, offset + seq_len, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
        )
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin

    def _forward_prefill(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q_latent = self.q_latent_proj(x)
        q_nope = self.q_nope_proj(q_latent).view(
            batch_size, seq_len, self.num_attention_heads, self.qk_nope_head_dim
        ).transpose(1, 2)
        q_rope = self.q_rope_proj(q_latent).view(
            batch_size, seq_len, self.num_attention_heads, self.qk_rope_head_dim
        ).transpose(1, 2)
        q_rope = self._apply_rope(q_rope, offset=0)

        kv_latent = self.kv_latent_proj(x)
        k_nope = self.k_nope_proj(kv_latent).view(
            batch_size, seq_len, self.num_attention_heads, self.qk_nope_head_dim
        ).transpose(1, 2)
        v = self.v_proj(kv_latent).view(
            batch_size, seq_len, self.num_attention_heads, self.v_head_dim
        ).transpose(1, 2)

        k_rope = self.k_rope_proj(x).view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_rope = k_rope.expand(batch_size, self.num_attention_heads, seq_len, self.qk_rope_head_dim)
        k_rope = self._apply_rope(k_rope, offset=0)

        scores_nope = torch.matmul(q_nope, k_nope.transpose(-1, -2))
        scores_rope = torch.matmul(q_rope, k_rope.transpose(-1, -2))
        scores = (scores_nope + scores_rope) / ((self.qk_nope_head_dim + self.qk_rope_head_dim) ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_attention_heads * self.v_head_dim
        )
        return self.o_proj(context)

    def _forward_decode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Current-step query in the compressed latent space (nope branch absorbed).
        q_latent = self.q_latent_proj(x)
        q_nope = self.q_nope_proj(q_latent).view(
            batch_size, seq_len, self.num_attention_heads, self.qk_nope_head_dim
        ).transpose(1, 2)
        # Absorb W_k_nope into the query: q_nope @ (W_q_nope-side @ W_k_nope) -> latent.
        q_nope_latent = torch.matmul(q_nope, self.q_nope_absorb)

        q_rope = self.q_rope_proj(q_latent).view(
            batch_size, seq_len, self.num_attention_heads, self.qk_rope_head_dim
        ).transpose(1, 2)
        q_rope = self._apply_rope(q_rope, offset=self.decode_kv_cache_len)

        # KV cache stores the compressed latent (kv_lora_rank), not expanded K/V.
        current_kv_latent = self.kv_latent_proj(x).unsqueeze(1).expand(
            batch_size, self.num_attention_heads, seq_len, self.kv_lora_rank
        )
        cache_kv_latent = torch.zeros(
            batch_size,
            self.num_attention_heads,
            self.decode_kv_cache_len,
            self.kv_lora_rank,
            device=x.device,
            dtype=x.dtype,
        )
        kv_latent_full = torch.cat((cache_kv_latent, current_kv_latent), dim=2)

        # Decoupled rope key branch is cached separately (small, per-position).
        current_k_rope = self.k_rope_proj(x).view(
            batch_size, seq_len, 1, self.qk_rope_head_dim
        ).transpose(1, 2)
        current_k_rope = current_k_rope.expand(
            batch_size,
            self.num_attention_heads,
            seq_len,
            self.qk_rope_head_dim,
        )
        current_k_rope = self._apply_rope(current_k_rope, offset=self.decode_kv_cache_len)
        cache_k_rope = torch.zeros(
            batch_size,
            self.num_attention_heads,
            self.decode_kv_cache_len,
            self.qk_rope_head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        k_rope_full = torch.cat((cache_k_rope, current_k_rope), dim=2)

        # Scores computed directly against the compressed latent + rope branch.
        scores_nope = torch.matmul(q_nope_latent, kv_latent_full.transpose(-1, -2))
        scores_rope = torch.matmul(q_rope, k_rope_full.transpose(-1, -2))
        scores = (scores_nope + scores_rope) / ((self.qk_nope_head_dim + self.qk_rope_head_dim) ** 0.5)
        probs = torch.softmax(scores, dim=-1)

        # Attention output stays in latent space, then a single absorbed (W_v @ W_o)
        # matrix maps it straight to hidden per head, summed over heads.
        latent_context = torch.matmul(probs, kv_latent_full)
        head_hidden = torch.matmul(latent_context, self.vo_absorb)
        context = head_hidden.sum(dim=1)
        return context + self.o_proj.bias.view(1, 1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.phase == "decode":
            return self._forward_decode(x)
        return self._forward_prefill(x)


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

        if x.is_meta:
            weighted_outputs = []
            for slot, expert in enumerate(self.experts[: self.top_k]):
                expert_output = expert(x)
                gate = topk_values[..., slot].unsqueeze(-1)
                weighted_outputs.append(expert_output * gate)
            return torch.stack(weighted_outputs, dim=2).sum(dim=2)

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
        attention_impl: str = "mla",
        mla_q_lora_rank: int = 1536,
        mla_kv_lora_rank: int = 512,
        mla_qk_nope_head_dim: int = 64,
        mla_qk_rope_head_dim: int = 64,
        mla_v_head_dim: int = 128,
        rope_theta: float = 10000.0,
        phase: str = "prefill",
        decode_kv_cache_len: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        if attention_impl == "mla":
            self.attn = DeepSeekV3ProxyMLAAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                q_lora_rank=mla_q_lora_rank,
                kv_lora_rank=mla_kv_lora_rank,
                qk_nope_head_dim=mla_qk_nope_head_dim,
                qk_rope_head_dim=mla_qk_rope_head_dim,
                v_head_dim=mla_v_head_dim,
                rope_theta=rope_theta,
                phase=phase,
                decode_kv_cache_len=decode_kv_cache_len,
            )
        else:
            raise ValueError(f"Unsupported attention implementation for DeepSeek V3 proxy: {attention_impl}")
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
        attention_impl: str = "mla",
        mla_q_lora_rank: int = 1536,
        mla_kv_lora_rank: int = 512,
        mla_qk_nope_head_dim: int = 64,
        mla_qk_rope_head_dim: int = 64,
        mla_v_head_dim: int = 128,
        rope_theta: float = 10000.0,
        phase: str = "prefill",
        decode_kv_cache_len: int = 0,
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
                    attention_impl=attention_impl,
                    mla_q_lora_rank=mla_q_lora_rank,
                    mla_kv_lora_rank=mla_kv_lora_rank,
                    mla_qk_nope_head_dim=mla_qk_nope_head_dim,
                    mla_qk_rope_head_dim=mla_qk_rope_head_dim,
                    mla_v_head_dim=mla_v_head_dim,
                    rope_theta=rope_theta,
                    phase=phase,
                    decode_kv_cache_len=decode_kv_cache_len,
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
    if phase not in {"prefill", "decode"}:
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
    attention_impl = str(
        workload_config.get("proxy_attention_impl", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["attention_impl"])
    )
    mla_q_lora_rank = int(
        workload_config.get(
            "proxy_mla_q_lora_rank",
            DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["mla_q_lora_rank"],
        )
    )
    mla_kv_lora_rank = int(
        workload_config.get(
            "proxy_mla_kv_lora_rank",
            DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["mla_kv_lora_rank"],
        )
    )
    mla_qk_nope_head_dim = int(
        workload_config.get(
            "proxy_mla_qk_nope_head_dim",
            DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["mla_qk_nope_head_dim"],
        )
    )
    mla_qk_rope_head_dim = int(
        workload_config.get(
            "proxy_mla_qk_rope_head_dim",
            DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["mla_qk_rope_head_dim"],
        )
    )
    mla_v_head_dim = int(
        workload_config.get(
            "proxy_mla_v_head_dim",
            DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["mla_v_head_dim"],
        )
    )
    rope_theta = float(workload_config.get("proxy_rope_theta", DEFAULT_DEEPSEEK_V3_PROXY_CONFIG["rope_theta"]))

    batch_size = int(workload_config["global_batch_size"])
    input_seq_len = int(workload_config["input_seq_len"])
    output_seq_len = int(workload_config["output_seq_len"])
    default_decode_kv_cache_len = input_seq_len + (output_seq_len // 2)
    decode_kv_cache_len = int(workload_config.get("proxy_decode_kv_cache_len", default_decode_kv_cache_len))

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
            attention_impl=attention_impl,
            mla_q_lora_rank=mla_q_lora_rank,
            mla_kv_lora_rank=mla_kv_lora_rank,
            mla_qk_nope_head_dim=mla_qk_nope_head_dim,
            mla_qk_rope_head_dim=mla_qk_rope_head_dim,
            mla_v_head_dim=mla_v_head_dim,
            rope_theta=rope_theta,
            phase=phase,
            decode_kv_cache_len=decode_kv_cache_len,
        )

    input_shape = [batch_size, input_seq_len] if phase == "prefill" else [batch_size, 1]
    kv_cache_seq_len = None if phase == "prefill" else decode_kv_cache_len
    input_spec = ProxyInputSpec(
        shape=input_shape,
        dtype="int64",
        input_kind="token_ids",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        num_layers=num_layers,
        analysis_phase=phase,
        kv_cache_seq_len=kv_cache_seq_len,
    )
    return model, input_spec
