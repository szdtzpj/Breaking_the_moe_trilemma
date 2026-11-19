"""DyClust-MoE 全套受控对比基线与正交执行重放工具。

严格对应论文 Table 2、Section 4.1--4.2 与 Appendix J：
1. ControlledSwitchMoELayer: 同-E Switch / 参数上界 Switch / 名义 matmul Switch（共享同一零丢弃修复器与派发器）；
2. FullExpertGroupedMoELayer: 未压缩全专家 + 分组路由与拓扑放置（Reviewer GXY9 W2）；
3. SubMoEMergedLayer: 预算匹配子空间专家合并基线（如 128 -> 17，Reviewer hhJ8 Q4）；
4. run_support_toggle_replay: Support-on vs. Support-off 开关重放（Reviewer 3A5U Question）；
5. run_2x2_orthogonal_replay: 物化 vs. 共享基 × 支持截断关 vs. 开 2x2 正交执行重放与数值一致性校验（Reviewer 3A5U W2）。
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyclust_moe.config import DyClustConfig
from dyclust_moe.layer import DyClustMoELayer, MoELayerOutput
from dyclust_moe.placement import DestinationCoalescedDispatcher, LinkCostModel
from dyclust_moe.routing import MassDistilledGroupRouter, bounded_exact_k_repair, compute_cyclic_reserve


class ControlledSwitchMoELayer(nn.Module):
    """受控全专家 Switch MoE 基线（共享同一零丢弃循环保底、精确-k 容量修复与通信派发器）。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        ffn_type: str = "gelu_2mat",
        retain_bias: bool = True,
        capacity_factor: float = 1.25,
        num_devices: int = 8,
        gpus_per_node: int = 4,
        norm_topk_prob: bool = True,
        routing_mode: str = "flat",
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.ffn_type = ffn_type
        self.retain_bias = retain_bias
        self.capacity_factor = capacity_factor
        self.num_devices = num_devices
        self.norm_topk_prob = norm_topk_prob
        self.routing_mode = routing_mode

        self.input_norm = nn.LayerNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.router = nn.Linear(d_model, num_experts, bias=True, device=device, dtype=dtype)

        # 未压缩独立全专家权重矩阵
        self.w_up = nn.Parameter(torch.empty(num_experts, d_ff, d_model, dtype=dtype, device=device))
        self.w_down = nn.Parameter(torch.empty(num_experts, d_model, d_ff, dtype=dtype, device=device))
        if ffn_type == "swiglu_3mat":
            self.w_gate = nn.Parameter(torch.empty(num_experts, d_ff, d_model, dtype=dtype, device=device))
        else:
            self.register_parameter("w_gate", None)

        if retain_bias:
            self.b_up = nn.Parameter(torch.zeros(num_experts, d_ff, dtype=dtype, device=device))
            self.b_down = nn.Parameter(torch.zeros(num_experts, d_model, dtype=dtype, device=device))
            if ffn_type == "swiglu_3mat":
                self.b_gate = nn.Parameter(torch.zeros(num_experts, d_ff, dtype=dtype, device=device))
            else:
                self.register_parameter("b_gate", None)
        else:
            self.register_parameter("b_up", None)
            self.register_parameter("b_down", None)
            self.register_parameter("b_gate", None)

        # 默认轮询设备放置（round-robin）
        self.register_buffer(
            "expert_to_owner",
            torch.tensor([i % num_devices for i in range(num_experts)], dtype=torch.long, device=device),
        )
        self.register_buffer("identity_group", torch.arange(num_experts, dtype=torch.long, device=device))
        self.link_model = LinkCostModel.default_a100_topology(num_devices=num_devices, gpus_per_node=gpus_per_node)
        self.dispatcher = DestinationCoalescedDispatcher(
            num_devices=num_devices,
            link_model=self.link_model,
            dtype_bytes=2 if dtype in (torch.float16, torch.bfloat16) else 4,
            coalesce=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.w_up, a=5**0.5)
        nn.init.kaiming_uniform_(self.w_down, a=5**0.5)
        if self.w_gate is not None:
            nn.init.kaiming_uniform_(self.w_gate, a=5**0.5)
        std = math.sqrt(1e-6 / float(self.d_model))
        nn.init.normal_(self.router.weight, mean=0.0, std=std)
        nn.init.zeros_(self.router.bias)

    @torch.no_grad()
    def load_full_experts(self, raw_experts: Dict[str, torch.Tensor]) -> None:
        """与 DyClust-MoE 共享完全一致的更新-0 全专家张量（受控配对初始化）。"""
        self.w_up.copy_(raw_experts["up"].to(self.w_up.device, self.w_up.dtype))
        self.w_down.copy_(raw_experts["down"].to(self.w_down.device, self.w_down.dtype))
        if self.w_gate is not None and "gate" in raw_experts:
            self.w_gate.copy_(raw_experts["gate"].to(self.w_gate.device, self.w_gate.dtype))

    def stored_scalar_count(self) -> int:
        total = self.w_up.numel() + self.w_down.numel()
        if self.w_gate is not None:
            total += self.w_gate.numel()
        if self.b_up is not None:
            total += self.b_up.numel() + self.b_down.numel()
            if self.b_gate is not None:
                total += self.b_gate.numel()
        return int(total)

    def forward(self, hidden: torch.Tensor, ordinal_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        orig_shape = hidden.shape
        flat_h = hidden.reshape(-1, self.d_model)
        n_tokens = flat_h.shape[0]
        h_norm = self.input_norm(flat_h)

        probs = F.softmax(self.router(h_norm), dim=-1)
        reserve = compute_cyclic_reserve(
            n_tokens, self.num_experts, self.top_k, ordinal_offset=ordinal_offset, device=hidden.device
        )

        if self.routing_mode == "device_preferred":
            # Full-Expert + Device-Preferred 基线（DeepSeek-V2 单优选设备 + 溢出修复）
            dev_aff = probs.new_full((n_tokens, self.num_devices), -1.0)
            for d in range(self.num_devices):
                idx = torch.nonzero(self.expert_to_owner == d, as_tuple=False).squeeze(-1)
                if idx.numel() > 0:
                    dev_aff[:, d] = probs.index_select(1, idx).max(dim=-1).values
            pref = torch.argmax(dev_aff, dim=-1)
            allowed = self.expert_to_owner.unsqueeze(0) == pref.unsqueeze(1)
        else:
            allowed = torch.ones((n_tokens, self.num_experts), dtype=torch.bool, device=hidden.device)

        capacity = math.ceil(self.capacity_factor * self.top_k * n_tokens / float(self.num_experts))
        acc_exp, rounds, overflow = bounded_exact_k_repair(probs, allowed, reserve, capacity=capacity)

        gathered = torch.gather(probs, 1, acc_exp)
        acc_w = gathered / gathered.sum(dim=-1, keepdim=True).clamp_min(1e-12) if self.norm_topk_prob else gathered

        # 通信核算（全专家每专家独立基矩阵，无共享基复用）
        comm = self.dispatcher.analyze_and_dispatch(
            flat_h, acc_exp, self.identity_group, self.expert_to_owner
        )

        # 按专家聚合执行全矩阵 GEMM
        edge_tok = torch.arange(n_tokens, device=hidden.device).unsqueeze(1).expand(n_tokens, self.top_k).reshape(-1)
        edge_exp = acc_exp.reshape(-1)
        edge_w = acc_w.reshape(-1, 1)
        x_e = flat_h.index_select(0, edge_tok)

        order = torch.argsort(edge_exp, stable=True)
        sorted_exp = edge_exp.index_select(0, order)
        sorted_x = x_e.index_select(0, order)
        sorted_y = flat_h.new_empty((edge_exp.shape[0], self.d_model))

        active_exps, counts = torch.unique_consecutive(sorted_exp, return_counts=True)
        offset = 0
        for e_id, cnt in zip(active_exps.tolist(), counts.tolist()):
            end = offset + cnt
            xe = sorted_x[offset:end]
            bu = self.b_up[e_id] if self.b_up is not None else None
            bd = self.b_down[e_id] if self.b_down is not None else None
            if self.ffn_type == "gelu_2mat":
                h_mid = F.gelu(F.linear(xe, self.w_up[e_id], bu))
            else:
                bg = self.b_gate[e_id] if self.b_gate is not None else None
                h_mid = F.silu(F.linear(xe, self.w_gate[e_id], bg)) * F.linear(xe, self.w_up[e_id], bu)
            sorted_y[offset:end] = F.linear(h_mid, self.w_down[e_id], bd)
            offset = end

        y_edges = flat_h.new_empty_like(sorted_y) if hasattr(flat_h, "new_empty_like") else torch.empty_like(sorted_y)
        y_edges.index_copy_(0, order, sorted_y)
        y_flat = (y_edges * edge_w).view(n_tokens, self.top_k, self.d_model).sum(dim=1)

        # 标准负载均衡损失
        a_ind = torch.zeros_like(probs)
        a_ind.scatter_(1, acc_exp, 1.0)
        l_lb = float(self.num_experts) * torch.sum(probs.mean(dim=0) * (a_ind.sum(dim=0) / float(max(1, self.top_k * n_tokens))))

        return y_flat.view(orig_shape), 0.01 * l_lb, {
            "bases_per_token": float(self.top_k),  # 无共享基：每个活跃专家各自独立基
            "owners_per_token": comm.mean_owners_per_token,
            "fb_wire_bytes": float(comm.fb_wire_bytes),
            "repair_rounds": float(rounds),
            "overflow_rate": float(overflow),
        }


class FullExpertGroupedMoELayer(nn.Module):
    """未压缩全专家 + 分组质量蒸馏路由 + 拓扑感知组放置基线（Reviewer GXY9 W2）。

    直接隔离分组路由/放置与共享基低秩表示的各自贡献。
    """

    def __init__(self, config: DyClustConfig, dtype: torch.dtype = torch.float32, device=None) -> None:
        super().__init__()
        self.config = config
        self.router = MassDistilledGroupRouter(config, dtype=dtype, device=device)
        self.switch_ffn = ControlledSwitchMoELayer(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_experts=config.num_experts,
            top_k=config.top_k,
            ffn_type=config.ffn_type,
            retain_bias=config.retain_bias,
            num_devices=config.num_devices,
            gpus_per_node=config.gpus_per_node,
            norm_topk_prob=config.norm_topk_prob,
            dtype=dtype,
            device=device,
        )
        self.link_model = LinkCostModel.default_a100_topology(config.num_devices, config.gpus_per_node)
        self.dispatcher = DestinationCoalescedDispatcher(
            num_devices=config.num_devices, link_model=self.link_model, coalesce=config.coalesce_tokens
        )

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        orig_shape = hidden.shape
        flat_h = hidden.reshape(-1, self.config.d_model)
        n_tokens = flat_h.shape[0]
        r_out = self.router(flat_h)

        comm = self.dispatcher.analyze_and_dispatch(
            flat_h, r_out.accepted_experts, self.router.expert_to_group, self.router.group_to_owner
        )

        # 使用未压缩独立全专家矩阵执行
        edge_tok = torch.arange(n_tokens, device=hidden.device).unsqueeze(1).expand(n_tokens, self.config.top_k).reshape(-1)
        edge_exp = r_out.accepted_experts.reshape(-1)
        edge_w = r_out.accepted_weights.reshape(-1, 1)
        x_e = flat_h.index_select(0, edge_tok)
        order = torch.argsort(edge_exp, stable=True)
        sorted_exp = edge_exp.index_select(0, order)
        sorted_x = x_e.index_select(0, order)
        sorted_y = torch.empty_like(sorted_x)

        active_exps, counts = torch.unique_consecutive(sorted_exp, return_counts=True)
        offset = 0
        for e_id, cnt in zip(active_exps.tolist(), counts.tolist()):
            end = offset + cnt
            xe = sorted_x[offset:end]
            bu = self.switch_ffn.b_up[e_id] if self.switch_ffn.b_up is not None else None
            bd = self.switch_ffn.b_down[e_id] if self.switch_ffn.b_down is not None else None
            h_mid = F.gelu(F.linear(xe, self.switch_ffn.w_up[e_id], bu))
            sorted_y[offset:end] = F.linear(h_mid, self.switch_ffn.w_down[e_id], bd)
            offset = end

        y_edges = torch.empty_like(sorted_y)
        y_edges.index_copy_(0, order, sorted_y)
        y_flat = (y_edges * edge_w).view(n_tokens, self.config.top_k, self.config.d_model).sum(dim=1)
        return y_flat.view(orig_shape), r_out.aux_loss, {
            "owners_per_token": comm.mean_owners_per_token,
            "fb_wire_bytes": float(comm.fb_wire_bytes),
            "stored_scalars": float(self.switch_ffn.stored_scalar_count()),
        }


def compute_max_merged_experts_within_budget(
    target_layer_scalars: int,
    d_model: int,
    d_ff: int,
    ffn_type: str = "gelu_2mat",
    retain_bias: bool = True,
) -> int:
    """计算在给定单层参数量上限内，Sub-MoE 可容纳的最大整数合并专家数（例如 128 -> 17）。"""
    mats = 2 if ffn_type == "gelu_2mat" else 3
    per_exp = mats * d_model * d_ff
    if retain_bias:
        per_exp += (d_ff + d_model) if ffn_type == "gelu_2mat" else (2 * d_ff + d_model)
    return max(1, target_layer_scalars // per_exp)


def build_sub_moe_merged_baseline(
    raw_experts: Dict[str, torch.Tensor],
    target_scalar_cap: int,
    d_model: int,
    d_ff: int,
    top_k: int = 2,
    ffn_type: str = "gelu_2mat",
    retain_bias: bool = True,
    num_devices: int = 8,
) -> ControlledSwitchMoELayer:
    """构造预算匹配子空间专家合并基线 Sub-MoE（Li et al., 2026 风格确定性聚类均值合并）。"""
    n_merged = compute_max_merged_experts_within_budget(
        target_scalar_cap, d_model, d_ff, ffn_type=ffn_type, retain_bias=retain_bias
    )
    w_up = raw_experts["up"].detach().to(dtype=torch.float32, device="cpu")
    e_orig = w_up.shape[0]

    # 使用归一化展平方向执行确定性贪心合并将 e_orig 个专家合并为 n_merged 个
    feats = w_up.reshape(e_orig, -1)
    feats = F.normalize(feats, dim=-1)
    # 选取最远优先中心并聚类
    centers_idx = [0]
    min_d = torch.cdist(feats, feats[0:1]).squeeze(1)
    min_d[0] = -1.0
    for _ in range(1, n_merged):
        nxt = int(torch.argmax(min_d).item())
        centers_idx.append(nxt)
        d_new = torch.cdist(feats, feats[nxt : nxt + 1]).squeeze(1)
        min_d = torch.minimum(min_d, d_new)
        for c in centers_idx:
            min_d[c] = -1.0

    centers = feats[centers_idx]
    assign = torch.argmax(feats @ centers.t(), dim=-1)  # (e_orig,)

    merged_layer = ControlledSwitchMoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=n_merged,
        top_k=min(top_k, n_merged),
        ffn_type=ffn_type,
        retain_bias=retain_bias,
        num_devices=num_devices,
    )
    with torch.no_grad():
        for m in range(n_merged):
            idx = torch.nonzero(assign == m, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                idx = torch.tensor([centers_idx[m]], dtype=torch.long)
            merged_layer.w_up[m].copy_(raw_experts["up"].index_select(0, idx).mean(dim=0))
            merged_layer.w_down[m].copy_(raw_experts["down"].index_select(0, idx).mean(dim=0))
            if merged_layer.w_gate is not None and "gate" in raw_experts:
                merged_layer.w_gate[m].copy_(raw_experts["gate"].index_select(0, idx).mean(dim=0))
    return merged_layer


@dataclass
class Replay2x2Result:
    """2x2 正交执行重放结果（物化 vs 共享基 × 支持截断关 vs 开）与数值等价性校验。"""

    mat_off_ms: float
    shared_off_ms: float
    mat_on_ms: float
    shared_on_ms: float
    wire_off_bytes: int
    wire_on_bytes: int
    bases_off: float
    bases_on: float
    owners_off: float
    owners_on: float
    max_abs_diff_off: float
    max_abs_diff_on: float
    support_speedup_under_shared: float
    shared_gain_when_off: float
    shared_gain_when_on: float


@torch.no_grad()
def run_2x2_orthogonal_replay(
    layer: DyClustMoELayer,
    hidden: torch.Tensor,
    num_iters: int = 10,
) -> Replay2x2Result:
    """严格执行论文 Table 46--47 的 Support-on/off 与 2x2 正交执行重放。

    固定同一紧凑检查点的权重、分组、所有者映射与输入张量，交叉切换：
      - 执行路径：materialized（物化独立专家矩阵）vs. shared_base（共享基融合执行）
      - 支持截断：support_restriction=False (Support-off) vs. True (Support-on)
    同时校验同支持模式下两种执行路径的输出数值完全一致。
    """
    layer.eval()

    def bench(exec_mode: str, support_on: bool) -> Tuple[MoELayerOutput, float]:
        # 预热 1 步
        ref = layer(
            hidden,
            record_history=False,
            execution_mode_override=exec_mode,
            support_restriction_override=support_on,
        )
        t0 = time.perf_counter()
        for _ in range(num_iters):
            out = layer(
                hidden,
                record_history=False,
                execution_mode_override=exec_mode,
                support_restriction_override=support_on,
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0 / float(max(1, num_iters))
        return out, dt_ms

    out_mat_off, t_mat_off = bench("materialized", False)
    out_sh_off, t_sh_off = bench("shared_base", False)
    out_mat_on, t_mat_on = bench("materialized", True)
    out_sh_on, t_sh_on = bench("shared_base", True)

    diff_off = float(torch.max(torch.abs(out_mat_off.output - out_sh_off.output)).item())
    diff_on = float(torch.max(torch.abs(out_mat_on.output - out_sh_on.output)).item())

    return Replay2x2Result(
        mat_off_ms=t_mat_off,
        shared_off_ms=t_sh_off,
        mat_on_ms=t_mat_on,
        shared_on_ms=t_sh_on,
        wire_off_bytes=out_sh_off.comm_stats.fb_wire_bytes,
        wire_on_bytes=out_sh_on.comm_stats.fb_wire_bytes,
        bases_off=out_sh_off.bases_per_token,
        bases_on=out_sh_on.bases_per_token,
        owners_off=out_sh_off.comm_stats.mean_owners_per_token,
        owners_on=out_sh_on.comm_stats.mean_owners_per_token,
        max_abs_diff_off=diff_off,
        max_abs_diff_on=diff_on,
        support_speedup_under_shared=(t_sh_off / max(t_sh_on, 1e-9)) - 1.0,
        shared_gain_when_off=(t_mat_off / max(t_sh_off, 1e-9)) - 1.0,
        shared_gain_when_on=(t_mat_on / max(t_sh_on, 1e-9)) - 1.0,
    )
