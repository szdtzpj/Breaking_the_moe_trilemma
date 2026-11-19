"""DyClust-MoE 共享基 + 低秩残差 FFN 表示层与精确参数账本。

严格实现论文 Section 3.1 (Eq. 2) 与 Appendix B.1：
- 支持两矩阵 GELU FFN (up, down，保留每专家独立 bias) 与三矩阵 SwiGLU FFN (gate, up, down)；
- 支持可选未压缩共享专家分支（如 Qwen1.5-MoE）；
- 实现真正的组级共享基融合批执行（按组聚合仅读一次基矩阵，不物化全量专家矩阵），
  同时提供物化执行路径以支持 2x2 正交重放与数值等价校验（max diff <= 1e-5）；
- 提供 64 切片插值桥前向路径与精确参数/压缩比解析账本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyclust_moe.config import DyClustConfig


@dataclass
class ScalarLedger:
    """精确的参数与活跃计算账本（区分唯一存储标量与名义/实际活跃标量）。"""

    stored_full_expert_scalars: int
    stored_compact_scalars: int
    storage_reduction_ratio: float
    nominal_one_group_active_scalars: int
    realized_active_scalars: float
    mean_bases_per_token: float


class SharedBaseLowRankProjection(nn.Module):
    """单个投影矩阵（gate / up / down）的组共享基 + 逐专家低秩残差表示。

    参数化形式（Eq. 2）：
        W_tilde[i] = W_base[g(i)] + A[i] @ B[i]^T,   rank(A[i] @ B[i]^T) <= r
    其中权重形状统一为 (out_dim, in_dim)，便于与 nn.Linear 权重约定一致。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_experts: int,
        num_groups: int,
        rank: int,
        retain_bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.num_groups = num_groups
        self.rank = rank
        self.retain_bias = retain_bias

        # 组共享稠密基矩阵: (G, out_dim, in_dim)
        self.group_bases = nn.Parameter(
            torch.empty(num_groups, out_dim, in_dim, dtype=dtype, device=device)
        )
        # 逐专家低秩因子: A (E, out_dim, r), B (E, in_dim, r)
        self.factor_A = nn.Parameter(
            torch.empty(num_experts, out_dim, rank, dtype=dtype, device=device)
        )
        self.factor_B = nn.Parameter(
            torch.empty(num_experts, in_dim, rank, dtype=dtype, device=device)
        )
        if retain_bias:
            self.expert_bias = nn.Parameter(
                torch.zeros(num_experts, out_dim, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("expert_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """默认 Kaiming 均匀初始化共享基，并对低秩残差使用小尺度初始化。"""
        nn.init.kaiming_uniform_(self.group_bases, a=5**0.5)
        if self.rank > 0:
            nn.init.normal_(self.factor_A, mean=0.0, std=0.02 / (self.rank**0.5))
            nn.init.normal_(self.factor_B, mean=0.0, std=0.02 / (self.rank**0.5))
        if self.expert_bias is not None:
            nn.init.zeros_(self.expert_bias)

    def materialize_weights(self, expert_to_group: torch.Tensor) -> torch.Tensor:
        """物化所有 E 个专家的有效权重矩阵 W_tilde，形状 (E, out_dim, in_dim)。

        仅用于重组重拟合、插值桥快照以及 2x2 正交重放中的物化基线对照。
        """
        base_per_expert = self.group_bases.index_select(0, expert_to_group)
        if self.rank == 0:
            return base_per_expert
        residual = torch.bmm(self.factor_A, self.factor_B.transpose(1, 2))
        return base_per_expert + residual

    def stored_scalar_count(self) -> int:
        """返回该投影层的唯一存储标量总数（含共享基、低秩因子与保留偏置）。"""
        total = self.group_bases.numel() + self.factor_A.numel() + self.factor_B.numel()
        if self.expert_bias is not None:
            total += self.expert_bias.numel()
        return int(total)

    def full_expert_scalar_count(self) -> int:
        """返回同等几何下未压缩独立全专家层的存储标量总数。"""
        total = self.num_experts * self.out_dim * self.in_dim
        if self.expert_bias is not None:
            total += self.num_experts * self.out_dim
        return int(total)

    def forward_by_edges_shared_base(
        self,
        x_edges: torch.Tensor,
        edge_token_ids: torch.Tensor,
        edge_expert_ids: torch.Tensor,
        expert_to_group: torch.Tensor,
        reuse_per_token_group: bool = True,
    ) -> torch.Tensor:
        """按组共享基融合批执行，不物化全量专家矩阵。

        Args:
            x_edges: 每条活跃边对应的输入张量，形状 (M, in_dim)。
            edge_token_ids: 每条活跃边对应的原始 token 编号，形状 (M,)。
            edge_expert_ids: 每条活跃边对应的专家 ID，形状 (M,)。
            expert_to_group: 长度为 E 的专家到组映射表。
            reuse_per_token_group: 若为 True（用于 up/gate 投影，同一 token 对同组多专家输入相同），
                则按唯一的 (token_id, group_id) 去重计算一次共享基输出并复用，进一步减少基计算与访存。

        Returns:
            每条活跃边的线性输出，形状 (M, out_dim)。
        """
        m_edges = x_edges.shape[0]
        if m_edges == 0:
            return x_edges.new_zeros((0, self.out_dim))

        edge_groups = expert_to_group.index_select(0, edge_expert_ids)
        out = x_edges.new_empty((m_edges, self.out_dim))

        # 第一阶段：按组聚合执行共享基乘法（每个命中组只读取一次 W_base[g]）
        if reuse_per_token_group:
            # 将 (group_id, token_id) 编码为单键去重，同 token 落入同组的多条边只算一次基投影
            max_tok = int(edge_token_ids.max().item()) + 1
            pair_keys = edge_groups * max_tok + edge_token_ids
            unique_keys, inverse_idx = torch.unique(pair_keys, sorted=True, return_inverse=True)
            # 为每个唯一键找到一条代表边索引
            num_unique = unique_keys.shape[0]
            first_edge = torch.empty(num_unique, dtype=torch.long, device=x_edges.device)
            first_edge.scatter_(0, inverse_idx, torch.arange(m_edges, device=x_edges.device))
            uniq_groups = unique_keys // max_tok
            uniq_x = x_edges.index_select(0, first_edge)
            uniq_base_out = x_edges.new_empty((num_unique, self.out_dim))

            # 按组连续切片执行 GEMM（保证每个共享基矩阵仅加载一次）
            active_groups, counts = torch.unique_consecutive(uniq_groups, return_counts=True)
            offset = 0
            for g_id, cnt in zip(active_groups.tolist(), counts.tolist()):
                end = offset + cnt
                uniq_base_out[offset:end] = F.linear(uniq_x[offset:end], self.group_bases[g_id])
                offset = end

            out.copy_(uniq_base_out.index_select(0, inverse_idx))
        else:
            # 用于 down 投影：不同专家输入不同，按组排序批量乘共享基（每个共享基矩阵仍只加载一次）
            order = torch.argsort(edge_groups, stable=True)
            sorted_groups = edge_groups.index_select(0, order)
            sorted_x = x_edges.index_select(0, order)
            sorted_base_out = x_edges.new_empty((m_edges, self.out_dim))

            active_groups, counts = torch.unique_consecutive(sorted_groups, return_counts=True)
            offset = 0
            for g_id, cnt in zip(active_groups.tolist(), counts.tolist()):
                end = offset + cnt
                sorted_base_out[offset:end] = F.linear(sorted_x[offset:end], self.group_bases[g_id])
                offset = end

            out.index_copy_(0, order, sorted_base_out)

        # 第二阶段：叠加逐专家低秩残差 (x @ B_e) @ A_e^T 与专家偏置
        if self.rank > 0:
            exp_order = torch.argsort(edge_expert_ids, stable=True)
            sorted_exp = edge_expert_ids.index_select(0, exp_order)
            sorted_x = x_edges.index_select(0, exp_order)
            sorted_res = x_edges.new_empty((m_edges, self.out_dim))

            active_exps, counts = torch.unique_consecutive(sorted_exp, return_counts=True)
            offset = 0
            for e_id, cnt in zip(active_exps.tolist(), counts.tolist()):
                end = offset + cnt
                # 低秩顺序乘法：(cnt, in_dim) @ (in_dim, r) -> (cnt, r) @ (r, out_dim)
                low = torch.matmul(sorted_x[offset:end], self.factor_B[e_id])
                sorted_res[offset:end] = torch.matmul(low, self.factor_A[e_id].transpose(0, 1))
                offset = end

            out = out + sorted_res.index_select(0, torch.argsort(exp_order))

        if self.expert_bias is not None:
            out = out + self.expert_bias.index_select(0, edge_expert_ids)

        return out

    def forward_by_edges_materialized(
        self,
        x_edges: torch.Tensor,
        edge_expert_ids: torch.Tensor,
        expert_to_group: torch.Tensor,
        override_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """物化专家矩阵对照执行路径（用于 2x2 正交重放与 64-slice 插值桥）。"""
        m_edges = x_edges.shape[0]
        if m_edges == 0:
            return x_edges.new_zeros((0, self.out_dim))

        weights = (
            override_weights
            if override_weights is not None
            else self.materialize_weights(expert_to_group)
        )
        exp_order = torch.argsort(edge_expert_ids, stable=True)
        sorted_exp = edge_expert_ids.index_select(0, exp_order)
        sorted_x = x_edges.index_select(0, exp_order)
        sorted_out = x_edges.new_empty((m_edges, self.out_dim))

        active_exps, counts = torch.unique_consecutive(sorted_exp, return_counts=True)
        offset = 0
        for e_id, cnt in zip(active_exps.tolist(), counts.tolist()):
            end = offset + cnt
            bias = self.expert_bias[e_id] if self.expert_bias is not None else None
            sorted_out[offset:end] = F.linear(sorted_x[offset:end], weights[e_id], bias)
            offset = end

        out = x_edges.new_empty((m_edges, self.out_dim))
        out.index_copy_(0, exp_order, sorted_out)
        return out


class SharedBaseLowRankFFN(nn.Module):
    """完整 MoE 专家 FFN 模块，支持 GELU-2mat、SwiGLU-3mat 与可选共享专家分支。"""

    def __init__(
        self,
        config: DyClustConfig,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.num_experts = config.num_experts
        self.group_size = config.group_size
        self.num_groups = config.num_groups
        self.rank = config.rank
        self.ffn_type = config.ffn_type

        # 初始均衡分组 g^(0)(i) = i % G（与 Appendix A.1 对齐的 0-based 版本）
        init_map = torch.tensor(
            [i % self.num_groups for i in range(self.num_experts)],
            dtype=torch.long,
            device=device,
        )
        self.register_buffer("expert_to_group", init_map)

        # 构建各投影层的共享基 + 低秩表示
        self.projections = nn.ModuleDict()
        if self.ffn_type == "gelu_2mat":
            self.projections["up"] = SharedBaseLowRankProjection(
                self.d_model, self.d_ff, self.num_experts, self.num_groups,
                self.rank, config.retain_bias, dtype=dtype, device=device,
            )
            self.projections["down"] = SharedBaseLowRankProjection(
                self.d_ff, self.d_model, self.num_experts, self.num_groups,
                self.rank, config.retain_bias, dtype=dtype, device=device,
            )
        else:
            # 三矩阵 SwiGLU（OLMoE / Qwen1.5-MoE）
            self.projections["gate"] = SharedBaseLowRankProjection(
                self.d_model, self.d_ff, self.num_experts, self.num_groups,
                self.rank, config.retain_bias, dtype=dtype, device=device,
            )
            self.projections["up"] = SharedBaseLowRankProjection(
                self.d_model, self.d_ff, self.num_experts, self.num_groups,
                self.rank, config.retain_bias, dtype=dtype, device=device,
            )
            self.projections["down"] = SharedBaseLowRankProjection(
                self.d_ff, self.d_model, self.num_experts, self.num_groups,
                self.rank, config.retain_bias, dtype=dtype, device=device,
            )

        # 可选未压缩常驻共享专家分支（如 Qwen1.5-MoE）
        self.shared_expert: Optional[nn.Module] = None
        if config.has_shared_expert:
            if self.ffn_type == "gelu_2mat":
                self.shared_expert = nn.Sequential(
                    nn.Linear(self.d_model, config.shared_expert_d_ff, bias=config.retain_bias, device=device, dtype=dtype),
                    nn.GELU(),
                    nn.Linear(config.shared_expert_d_ff, self.d_model, bias=config.retain_bias, device=device, dtype=dtype),
                )
            else:
                self.shared_expert = _SwiGLUDenseFFN(
                    self.d_model, config.shared_expert_d_ff, bias=config.retain_bias, device=device, dtype=dtype
                )

    def get_group_members(self) -> Dict[int, List[int]]:
        """返回按组 ID 索引的有序专家 ID 列表。"""
        groups: Dict[int, List[int]] = {g: [] for g in range(self.num_groups)}
        for i, g in enumerate(self.expert_to_group.tolist()):
            groups[int(g)].append(i)
        return groups

    def materialize_all_weights(self) -> Dict[str, torch.Tensor]:
        """物化所有投影的全专家权重，返回 {proj_name: Tensor(E, out_dim, in_dim)}。"""
        return {
            name: proj.materialize_weights(self.expert_to_group)
            for name, proj in self.projections.items()
        }

    @torch.no_grad()
    def apply_partition_and_factors(
        self,
        new_expert_to_group: torch.Tensor,
        new_bases: Dict[str, torch.Tensor],
        new_A: Dict[str, torch.Tensor],
        new_B: Dict[str, torch.Tensor],
    ) -> None:
        """原子写入新分组映射与精确重拟合后的共享基与低秩因子。"""
        self.expert_to_group.copy_(new_expert_to_group.to(self.expert_to_group.device))
        for name, proj in self.projections.items():
            proj.group_bases.copy_(new_bases[name].to(proj.group_bases.device, dtype=proj.group_bases.dtype))
            proj.factor_A.copy_(new_A[name].to(proj.factor_A.device, dtype=proj.factor_A.dtype))
            proj.factor_B.copy_(new_B[name].to(proj.factor_B.device, dtype=proj.factor_B.dtype))

    @torch.no_grad()
    def init_from_dense_upcycling(
        self,
        dense_weights: Dict[str, torch.Tensor],
        dense_biases: Optional[Dict[str, torch.Tensor]] = None,
        noise_mode: str = "lowrank_1e-3",
        rel_noise_norm: float = 1e-3,
        seed: int = 0,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """从稠密父模型 FFN 执行稀疏升级初始化（支持微扰与 0.05 全秩高噪模式）。

        Args:
            dense_weights: 稠密父模型权重字典 {proj_name: (out_dim, in_dim)}。
            dense_biases: 可选稠密父模型偏置字典 {proj_name: (out_dim,)}。
            noise_mode:
                - ``"lowrank_1e-3"``: 论文默认稀疏升级（相对范数 1e-3 的秩-r 微扰）；
                - ``"fullrank_gaussian"``: Rebuttal 鲁棒性实验中的全秩高噪（如 0.05 相对范数）。
            rel_noise_norm: 相对 Frobenius 范数尺度。
            seed: 确定性计数器随机种子。

        Returns:
            (raw_full_experts, initial_conversion_rms):
                raw_full_experts 供同-E Switch 基线共享同一更新-0 张量；
                initial_conversion_rms 为初始紧凑化相对重构 RMS 误差。
        """
        from dyclust_moe.grouping import exact_refit_fixed_partition

        gen = torch.Generator(device="cpu").manual_seed(seed)
        raw_experts: Dict[str, torch.Tensor] = {}

        for p_idx, name in enumerate(self.config.projection_names):
            w0 = dense_weights[name].detach().to(dtype=torch.float64, device="cpu")
            out_dim, in_dim = w0.shape
            base_norm = torch.linalg.norm(w0, ord="fro").item()
            exps = w0.unsqueeze(0).repeat(self.num_experts, 1, 1)

            for e in range(self.num_experts):
                if noise_mode == "lowrank_1e-3" and self.rank > 0:
                    u = torch.randn(out_dim, self.rank, generator=gen, dtype=torch.float64)
                    v = torch.randn(in_dim, self.rank, generator=gen, dtype=torch.float64)
                    delta = u @ v.t()
                else:
                    delta = torch.randn(out_dim, in_dim, generator=gen, dtype=torch.float64)
                d_norm = torch.linalg.norm(delta, ord="fro").item()
                if d_norm > 0 and base_norm > 0:
                    delta = delta * (rel_noise_norm * base_norm / d_norm)
                exps[e] = exps[e] + delta

            raw_experts[name] = exps

            if dense_biases is not None and name in dense_biases and self.projections[name].expert_bias is not None:
                b0 = dense_biases[name].detach().to(
                    device=self.projections[name].expert_bias.device,
                    dtype=self.projections[name].expert_bias.dtype,
                )
                self.projections[name].expert_bias.copy_(b0.unsqueeze(0).repeat(self.num_experts, 1))

        # 在初始均衡分组上执行确定性冷启动 FP64 重拟合
        refit = exact_refit_fixed_partition(
            weights=raw_experts,
            expert_to_group=self.expert_to_group.cpu(),
            num_groups=self.num_groups,
            rank=self.rank,
            max_sweeps=self.config.refit_max_sweeps,
            tol=self.config.refit_tol,
            eps_w=self.config.eps_w,
        )
        self.apply_partition_and_factors(
            self.expert_to_group, refit.bases, refit.factor_A, refit.factor_B
        )
        return raw_experts, refit.rms_error

    @torch.no_grad()
    def init_from_full_experts(
        self,
        full_weights: Dict[str, torch.Tensor],
        full_biases: Optional[Dict[str, torch.Tensor]] = None,
        run_grouping: bool = True,
        seed: int = 0,
    ) -> float:
        """将成熟预训练全专家 FFN（如 OLMoE / Qwen1.5-MoE）确定性转换为紧凑表示（0B 转换）。"""
        from dyclust_moe.grouping import DeterministicGrouper, exact_refit_fixed_partition

        weights_fp64 = {
            k: v.detach().to(dtype=torch.float64, device="cpu") for k, v in full_weights.items()
        }
        if full_biases is not None:
            for name, proj in self.projections.items():
                if name in full_biases and proj.expert_bias is not None:
                    proj.expert_bias.copy_(
                        full_biases[name].to(device=proj.expert_bias.device, dtype=proj.expert_bias.dtype)
                    )

        if run_grouping:
            grouper = DeterministicGrouper(self.config)
            proposal = grouper.propose_and_refit(
                weights=weights_fp64,
                input_centroids=None,
                exposed_mask=None,
                co_routing=None,
                incumbent_group_map=self.expert_to_group.cpu(),
                run_seed=seed,
                layer_id=self.config.layer_id,
                event_id=0,
                skip_prescreen=True,
            )
            self.apply_partition_and_factors(
                proposal.expert_to_group, proposal.bases, proposal.factor_A, proposal.factor_B
            )
            return proposal.rms_error

        refit = exact_refit_fixed_partition(
            weights=weights_fp64,
            expert_to_group=self.expert_to_group.cpu(),
            num_groups=self.num_groups,
            rank=self.rank,
            max_sweeps=self.config.refit_max_sweeps,
            tol=self.config.refit_tol,
            eps_w=self.config.eps_w,
        )
        self.apply_partition_and_factors(
            self.expert_to_group, refit.bases, refit.factor_A, refit.factor_B
        )
        return refit.rms_error

    def forward_routed(
        self,
        hidden: torch.Tensor,
        accepted_experts: torch.Tensor,
        accepted_weights: torch.Tensor,
        execution_mode: Optional[str] = None,
        override_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """根据已接受路由边执行前向计算。

        Args:
            hidden: 输入隐状态，形状 (N, d_model)。
            accepted_experts: 每个 token 接受的 k 个专家 ID，形状 (N, k)。
            accepted_weights: 对应的路由权重 q_hat，形状 (N, k)。
            execution_mode: ``"shared_base"`` 或 ``"materialized"``；默认读取 config.execution_mode。
            override_weights: 可选覆盖权重字典（供 64 切片插值桥使用）。

        Returns:
            (output, stats): output 形状 (N, d_model)，stats 包含 bases_per_token 等执行指标。
        """
        n_tokens, k = accepted_experts.shape
        mode = execution_mode or self.config.execution_mode
        if override_weights is not None:
            mode = "materialized"

        # 展平为活跃边列表 (M = N * k)
        edge_token_ids = torch.arange(n_tokens, device=hidden.device).unsqueeze(1).expand(n_tokens, k).reshape(-1)
        edge_expert_ids = accepted_experts.reshape(-1)
        edge_weights = accepted_weights.reshape(-1, 1)
        x_edges = hidden.index_select(0, edge_token_ids)

        # 统计每个 token 实际触达的不同共享基组数（bases/token）
        token_groups = self.expert_to_group.index_select(0, edge_expert_ids).view(n_tokens, k)
        # 对每行去重计数（支持任意 k）
        sorted_tg, _ = torch.sort(token_groups, dim=1)
        if k > 1:
            unique_counts = 1 + (sorted_tg[:, 1:] != sorted_tg[:, :-1]).sum(dim=1)
        else:
            unique_counts = torch.ones(n_tokens, dtype=torch.long, device=hidden.device)
        bases_per_token = unique_counts.float().mean().item()

        if self.ffn_type == "gelu_2mat":
            if mode == "shared_base":
                h_up = self.projections["up"].forward_by_edges_shared_base(
                    x_edges, edge_token_ids, edge_expert_ids, self.expert_to_group, reuse_per_token_group=True
                )
                h_act = F.gelu(h_up)
                h_down = self.projections["down"].forward_by_edges_shared_base(
                    h_act, edge_token_ids, edge_expert_ids, self.expert_to_group, reuse_per_token_group=False
                )
            else:
                ow_up = override_weights["up"] if override_weights is not None else None
                ow_dn = override_weights["down"] if override_weights is not None else None
                h_up = self.projections["up"].forward_by_edges_materialized(
                    x_edges, edge_expert_ids, self.expert_to_group, override_weights=ow_up
                )
                h_act = F.gelu(h_up)
                h_down = self.projections["down"].forward_by_edges_materialized(
                    h_act, edge_expert_ids, self.expert_to_group, override_weights=ow_dn
                )
        else:
            if mode == "shared_base":
                h_gate = self.projections["gate"].forward_by_edges_shared_base(
                    x_edges, edge_token_ids, edge_expert_ids, self.expert_to_group, reuse_per_token_group=True
                )
                h_up = self.projections["up"].forward_by_edges_shared_base(
                    x_edges, edge_token_ids, edge_expert_ids, self.expert_to_group, reuse_per_token_group=True
                )
                h_act = F.silu(h_gate) * h_up
                h_down = self.projections["down"].forward_by_edges_shared_base(
                    h_act, edge_token_ids, edge_expert_ids, self.expert_to_group, reuse_per_token_group=False
                )
            else:
                ow_g = override_weights["gate"] if override_weights is not None else None
                ow_u = override_weights["up"] if override_weights is not None else None
                ow_d = override_weights["down"] if override_weights is not None else None
                h_gate = self.projections["gate"].forward_by_edges_materialized(
                    x_edges, edge_expert_ids, self.expert_to_group, override_weights=ow_g
                )
                h_up = self.projections["up"].forward_by_edges_materialized(
                    x_edges, edge_expert_ids, self.expert_to_group, override_weights=ow_u
                )
                h_act = F.silu(h_gate) * h_up
                h_down = self.projections["down"].forward_by_edges_materialized(
                    h_act, edge_expert_ids, self.expert_to_group, override_weights=ow_d
                )

        # 按路由权重加权聚合回每个 token（Eq. 8）
        weighted = (h_down * edge_weights).view(n_tokens, k, self.d_model)
        out = weighted.sum(dim=1)

        if self.shared_expert is not None:
            out = out + self.shared_expert(hidden)

        return out, {"bases_per_token": bases_per_token}

    def forward_interpolated(
        self,
        hidden: torch.Tensor,
        accepted_experts: torch.Tensor,
        accepted_weights: torch.Tensor,
        pre_weights: Dict[str, torch.Tensor],
        cand_weights: Dict[str, torch.Tensor],
        slice_idx: int,
        num_slices: int,
    ) -> torch.Tensor:
        """64 切片参数插值桥前向路径（Section 3.4）：

            W^{(s)} = W_cand + (1 - s / H) * (W_pre - W_cand),   s = 1..H
        """
        alpha_old = 1.0 - float(slice_idx) / float(num_slices)
        interp = {
            name: cand_weights[name] + alpha_old * (pre_weights[name] - cand_weights[name])
            for name in self.config.projection_names
        }
        out, _ = self.forward_routed(
            hidden, accepted_experts, accepted_weights, execution_mode="materialized", override_weights=interp
        )
        return out

    def compute_scalar_ledger(self, mean_bases_per_token: float = 1.0) -> ScalarLedger:
        """计算本层唯一存储标量、压缩比与名义/实际活跃参数账本。"""
        full_scalars = sum(p.full_expert_scalar_count() for p in self.projections.values())
        compact_scalars = sum(p.stored_scalar_count() for p in self.projections.values())
        reduction = 1.0 - (compact_scalars / max(1, full_scalars))

        # 单组名义活跃标量 = 1 个组基 + k 个专家低秩因子与偏置
        one_group_base = sum(p.out_dim * p.in_dim for p in self.projections.values())
        per_expert_res = sum(
            (p.out_dim + p.in_dim) * p.rank + (p.out_dim if p.expert_bias is not None else 0)
            for p in self.projections.values()
        )
        nominal_active = one_group_base + self.config.top_k * per_expert_res
        realized_active = mean_bases_per_token * one_group_base + self.config.top_k * per_expert_res

        return ScalarLedger(
            stored_full_expert_scalars=full_scalars,
            stored_compact_scalars=compact_scalars,
            storage_reduction_ratio=reduction,
            nominal_one_group_active_scalars=nominal_active,
            realized_active_scalars=realized_active,
            mean_bases_per_token=mean_bases_per_token,
        )


class _SwiGLUDenseFFN(nn.Module):
    """未压缩常驻共享 SwiGLU 分支（用于 Qwen1.5-MoE 迁移）。"""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, device=None, dtype=None) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias, device=device, dtype=dtype)
        self.up = nn.Linear(d_model, d_ff, bias=bias, device=device, dtype=dtype)
        self.down = nn.Linear(d_ff, d_model, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
