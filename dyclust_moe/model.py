"""DyClust-MoE 完整 Pre-LN RoPE Transformer 语言模型与公版 MoE 转换适配器。

严格对齐论文 Appendix B.1、Appendix F 与 Appendix G：
- Pre-LayerNorm (eps=1e-5) + RoPE 位置编码 + 绑定词表嵌入；
- 每层独立的层局部（layer-local）MoE 分组状态（修复早期代码全层共享同一 MoE 实例的问题）；
- 支持从稠密父模型一键稀疏升级（lowrank_1e-3 / fullrank_gaussian 0.05）与无稠密父模型从零训练；
- 提供 PublicMoEAdapter，支持将 SwiGLU 3 矩阵 + norm_topk_prob=False (+可选未压缩共享专家分支)
  的公版 MoE（OLMoE-1B-7B / Qwen1.5-MoE-A2.7B）确定性转换为 DyClust-MoE，并核算全模型 vs. 路由模块参数账本。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyclust_moe.config import DyClustConfig, TransformerLMConfig
from dyclust_moe.layer import DyClustMoELayer, MoELayerOutput
from dyclust_moe.lifecycle import RegroupEventReport


class RotaryEmbedding(nn.Module):
    """标准 RoPE 旋转位置编码（Su et al., 2021）。"""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对形状 (B, n_heads, T, head_dim) 的 Q/K 张量施加 RoPE 旋转。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos[None, None, :, :] + rotated * sin[None, None, :, :]


class CausalSelfAttention(nn.Module):
    """带 RoPE 的多头因果自注意力层。"""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(t, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        scale = 1.0 / math.sqrt(float(self.head_dim))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(ctx)


class DenseGELUFFN(nn.Module):
    """与论文一致的两矩阵偏置保留稠密 GELU FFN。"""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=True)
        self.down = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class DyClustTransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer 块（可配置为稠密 FFN 或独立的 DyClustMoELayer）。"""

    def __init__(
        self,
        layer_id: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        layernorm_eps: float = 1e-5,
        moe_config: Optional[DyClustConfig] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_moe = moe_config is not None
        self.ln1 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len=max_seq_len)
        self.ln2 = nn.LayerNorm(d_model, eps=layernorm_eps)

        if moe_config is not None:
            layer_cfg = copy.deepcopy(moe_config)
            layer_cfg.layer_id = layer_id
            layer_cfg.d_model = d_model
            layer_cfg.d_ff = d_ff
            self.ffn: nn.Module = DyClustMoELayer(layer_cfg)
        else:
            self.ffn = DenseGELUFFN(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        record_history: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[MoELayerOutput]]:
        # Pre-LayerNorm 残差结构
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        h2 = self.ln2(x)
        if self.is_moe:
            moe_out: MoELayerOutput = self.ffn(h2, record_history=record_history)
            x = x + moe_out.output
            return x, moe_out.aux_loss, moe_out
        x = x + self.ffn(h2)
        return x, x.new_zeros(()), None


@dataclass
class LMForwardOutput:
    """语言模型前向完整输出。"""

    logits: torch.Tensor
    task_loss: Optional[torch.Tensor]
    aux_loss: torch.Tensor
    total_loss: Optional[torch.Tensor]
    mean_bases_per_token: float
    mean_owners_per_token: float
    total_fb_wire_bytes: int
    layer_outputs: Dict[int, MoELayerOutput]


class DyClustTransformerLM(nn.Module):
    """完整 Pre-LN RoPE DyClust-MoE 因果语言模型（论文 Appendix B.1 架构）。"""

    def __init__(self, config: TransformerLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        moe_set = set(config.moe_layers)

        self.blocks = nn.ModuleList(
            [
                DyClustTransformerBlock(
                    layer_id=idx,
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    max_seq_len=config.max_seq_len,
                    layernorm_eps=config.layernorm_eps,
                    moe_config=config.moe_config if idx in moe_set else None,
                )
                for idx in range(config.num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(config.d_model, eps=config.layernorm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._init_transformer_weights()
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self._round_robin_cursor: int = 0

    @torch.no_grad()
    def _init_transformer_weights(self) -> None:
        """按标准 Pre-LN Transformer 规范初始化嵌入层与注意力/稠密投影，防止绑定 lm_head 初始 logits 爆炸。"""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        resid_std = 0.02 / math.sqrt(2.0 * max(1, self.config.num_layers))
        for blk in self.blocks:
            nn.init.normal_(blk.attn.qkv.weight, mean=0.0, std=0.02)
            nn.init.zeros_(blk.attn.qkv.bias)
            nn.init.normal_(blk.attn.out_proj.weight, mean=0.0, std=resid_std)
            nn.init.zeros_(blk.attn.out_proj.bias)
            if not blk.is_moe and isinstance(blk.ffn, DenseGELUFFN):
                nn.init.normal_(blk.ffn.up.weight, mean=0.0, std=0.02)
                nn.init.zeros_(blk.ffn.up.bias)
                nn.init.normal_(blk.ffn.down.weight, mean=0.0, std=resid_std)
                nn.init.zeros_(blk.ffn.down.bias)

    def get_moe_layers(self) -> List[Tuple[int, DyClustMoELayer]]:
        """按层号顺序返回全部层局部 DyClustMoELayer。"""
        return [
            (blk.layer_id, blk.ffn)
            for blk in self.blocks
            if blk.is_moe and isinstance(blk.ffn, DyClustMoELayer)
        ]

    @torch.no_grad()
    def upcycle_from_dense_parent(
        self,
        dense_parent: "DyClustTransformerLM",
        noise_mode: str = "lowrank_1e-3",
        rel_noise_norm: float = 1e-3,
        seed: int = 0,
    ) -> Dict[int, float]:
        """从共享稠密父模型执行确定性稀疏升级（支持微扰与 0.05 全秩高噪），返回各层初始重构 RMS。"""
        self.embed.weight.copy_(dense_parent.embed.weight)
        self.final_ln.load_state_dict(dense_parent.final_ln.state_dict())
        init_rms: Dict[int, float] = {}

        for idx, (blk, parent_blk) in enumerate(zip(self.blocks, dense_parent.blocks)):
            blk.ln1.load_state_dict(parent_blk.ln1.state_dict())
            blk.attn.load_state_dict(parent_blk.attn.state_dict())
            blk.ln2.load_state_dict(parent_blk.ln2.state_dict())

            if blk.is_moe and isinstance(blk.ffn, DyClustMoELayer) and isinstance(parent_blk.ffn, DenseGELUFFN):
                dense_w = {
                    "up": parent_blk.ffn.up.weight.detach(),
                    "down": parent_blk.ffn.down.weight.detach(),
                }
                dense_b = {
                    "up": parent_blk.ffn.up.bias.detach(),
                    "down": parent_blk.ffn.down.bias.detach(),
                }
                _, rms = blk.ffn.ffn.init_from_dense_upcycling(
                    dense_w,
                    dense_biases=dense_b,
                    noise_mode=noise_mode,
                    rel_noise_norm=rel_noise_norm,
                    seed=seed + idx * 1009,
                )
                blk.ffn.sync_single_state()
                init_rms[idx] = rms
            elif not blk.is_moe and isinstance(blk.ffn, DenseGELUFFN) and isinstance(parent_blk.ffn, DenseGELUFFN):
                blk.ffn.load_state_dict(parent_blk.ffn.state_dict())

        return init_rms

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        record_history: bool = True,
    ) -> LMForwardOutput:
        x = self.embed(input_ids)
        aux_losses: List[torch.Tensor] = []
        bases_list: List[float] = []
        owners_list: List[float] = []
        total_wire = 0
        layer_outs: Dict[int, MoELayerOutput] = {}

        for blk in self.blocks:
            x, aux_l, moe_o = blk(x, attention_mask=attention_mask, record_history=record_history)
            if moe_o is not None:
                aux_losses.append(aux_l)
                bases_list.append(moe_o.bases_per_token)
                owners_list.append(moe_o.comm_stats.mean_owners_per_token)
                total_wire += moe_o.comm_stats.fb_wire_bytes
                layer_outs[blk.layer_id] = moe_o

        logits = self.lm_head(self.final_ln(x))
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())

        task_loss: Optional[torch.Tensor] = None
        total_loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = task_loss + aux_loss

        return LMForwardOutput(
            logits=logits,
            task_loss=task_loss,
            aux_loss=aux_loss,
            total_loss=total_loss,
            mean_bases_per_token=(sum(bases_list) / len(bases_list)) if bases_list else 0.0,
            mean_owners_per_token=(sum(owners_list) / len(owners_list)) if owners_list else 0.0,
            total_fb_wire_bytes=total_wire,
            layer_outputs=layer_outs,
        )

    def trigger_round_robin_regrouping(
        self,
        shadow_input_ids: Optional[torch.Tensor] = None,
        data_cursor: int = 0,
        rng_counter: int = 0,
    ) -> Optional[RegroupEventReport]:
        """按论文 Appendix A.1 轮询（round-robin）顺序访问下一个 MoE 层执行在线重组。"""
        moe_layers = self.get_moe_layers()
        if not moe_layers:
            return None
        target_idx = self._round_robin_cursor % len(moe_layers)
        self._round_robin_cursor += 1
        layer_id, moe_layer = moe_layers[target_idx]

        # 若提供影子批次 token，先运行前缀层提取该 MoE 层的真实输入隐状态
        shadow_h: Optional[torch.Tensor] = None
        if shadow_input_ids is not None:
            with torch.no_grad():
                h = self.embed(shadow_input_ids)
                for blk in self.blocks:
                    if blk.layer_id == layer_id:
                        h = h + blk.attn(blk.ln1(h))
                        shadow_h = blk.ln2(h).reshape(-1, self.config.d_model)
                        break
                    h, _, _ = blk(h, record_history=False)

        return moe_layer.step_online_regrouping(
            shadow_batch=shadow_h,
            data_cursor=data_cursor,
            rng_counter=rng_counter,
        )

    def whole_model_scalar_inventory(self) -> Dict[str, float]:
        """计算整模型（含词表、注意力、非 MoE 层）与纯 MoE 路由模块的唯一存储参数量及压缩比。"""
        # 唯一存储参数量（考虑词表权重绑定只计一次）
        unique_compact = sum(p.numel() for p in self.parameters())
        full_moe_extra = 0
        compact_moe_total = 0
        full_moe_total = 0
        for _, layer in self.get_moe_layers():
            led = layer.scalar_ledger()
            compact_moe_total += led.stored_compact_scalars
            full_moe_total += led.stored_full_expert_scalars
            full_moe_extra += led.stored_full_expert_scalars - led.stored_compact_scalars

        unique_full_equiv = unique_compact + full_moe_extra
        return {
            "whole_model_compact_scalars": float(unique_compact),
            "whole_model_full_equiv_scalars": float(unique_full_equiv),
            "whole_model_reduction_pct": 100.0 * (1.0 - unique_compact / max(1, unique_full_equiv)),
            "routed_module_compact_scalars": float(compact_moe_total),
            "routed_module_full_scalars": float(full_moe_total),
            "routed_module_reduction_pct": 100.0 * (1.0 - compact_moe_total / max(1, full_moe_total)),
        }


class PublicMoEAdapter:
    """公版预训练 MoE（OLMoE-1B-7B / Qwen1.5-MoE-A2.7B）确定性转换适配器。

    严格保留原生非归一化 softmax 权重（norm_topk_prob=False）、无偏置组选择器、
    SwiGLU 三矩阵表示以及可选未压缩常驻共享专家分支（Qwen1.5-MoE）。
    """

    @staticmethod
    def build_olmoe_adapter_config(
        d_model: int = 2048,
        d_ff: int = 1024,
        num_experts: int = 64,
        group_size: int = 8,
        rank: int = 8,
        top_k: int = 8,
        num_devices: int = 8,
    ) -> DyClustConfig:
        """返回与论文 Appendix F 对齐的 OLMoE-1B-7B 适配配置。"""
        return DyClustConfig(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            group_size=group_size,
            rank=rank,
            top_k=top_k,
            ffn_type="swiglu_3mat",
            retain_bias=False,
            has_shared_expert=False,
            norm_topk_prob=False,
            selector_bias=False,
            selector_mass_threshold=0.995,
            max_extra_groups=6,
            num_devices=num_devices,
        )

    @staticmethod
    def build_qwen_adapter_config(
        d_model: int = 2048,
        d_ff: int = 1408,
        num_experts: int = 60,
        group_size: int = 6,
        rank: int = 8,
        top_k: int = 4,
        shared_expert_d_ff: int = 5632,
        num_devices: int = 8,
    ) -> DyClustConfig:
        """返回与论文 Appendix G 对齐的 Qwen1.5-MoE-A2.7B 适配配置（保留未压缩共享分支）。"""
        return DyClustConfig(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            group_size=group_size,
            rank=rank,
            top_k=top_k,
            ffn_type="swiglu_3mat",
            retain_bias=False,
            has_shared_expert=True,
            shared_expert_d_ff=shared_expert_d_ff,
            norm_topk_prob=False,
            selector_bias=False,
            selector_mass_threshold=0.995,
            max_extra_groups=6,
            num_devices=num_devices,
        )

    @staticmethod
    @torch.no_grad()
    def convert_swiglu_experts(
        config: DyClustConfig,
        expert_gate: torch.Tensor,
        expert_up: torch.Tensor,
        expert_down: torch.Tensor,
        router_weight: torch.Tensor,
        shared_expert_weights: Optional[Dict[str, torch.Tensor]] = None,
        seed: int = 0,
    ) -> Tuple[DyClustMoELayer, float]:
        """将成熟预训练 SwiGLU 专家与原生路由器确定性一键转换为紧凑 DyClustMoELayer（0B 转换）。"""
        layer = DyClustMoELayer(config)
        rms = layer.ffn.init_from_full_experts(
            full_weights={"gate": expert_gate, "up": expert_up, "down": expert_down},
            run_grouping=True,
            seed=seed,
        )
        layer.router.expert_router.weight.copy_(router_weight.to(layer.router.expert_router.weight.dtype))
        if layer.router.expert_router.bias is not None:
            layer.router.expert_router.bias.zero_()
        layer.router.init_selector_from_expert_router(layer.ffn.expert_to_group)
        layer.sync_single_state()

        if shared_expert_weights is not None and layer.ffn.shared_expert is not None:
            layer.ffn.shared_expert.gate.weight.copy_(shared_expert_weights["gate"])
            layer.ffn.shared_expert.up.weight.copy_(shared_expert_weights["up"])
            layer.ffn.shared_expert.down.weight.copy_(shared_expert_weights["down"])

        return layer, rms
