"""DyClust-MoE 统一层封装（单分组状态协调表示、路由、放置与在线生命周期）。

提供即插即用的 ``DyClustMoELayer``，支持：
- Full DyClust-MoE 与 DyClust-MoE-Lite；
- 五种状态耦合模式（single / decoupled_cyclic / matched_schedule / partial_route_place / penalized_independent）；
- Support-on/off 与 shared_base / materialized 2x2 正交执行切换；
- 实时统计 bases/token、owners/token、F+B wire bytes、合并载荷比与参数账本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from dyclust_moe.config import DyClustConfig
from dyclust_moe.lifecycle import HistoryWindowCollector, OnlineRegroupingController, RegroupEventReport
from dyclust_moe.placement import CommunicationStats, DestinationCoalescedDispatcher, LinkCostModel
from dyclust_moe.representation import ScalarLedger, SharedBaseLowRankFFN
from dyclust_moe.routing import MassDistilledGroupRouter, RoutingOutput


@dataclass
class MoELayerOutput:
    """DyClustMoELayer 前向完整输出。"""

    output: torch.Tensor
    aux_loss: torch.Tensor
    routing: RoutingOutput
    comm_stats: CommunicationStats
    bases_per_token: float


class DyClustMoELayer(nn.Module):
    """集成共享基表示、质量蒸馏分组路由、目的地合并放置与在线重组的 MoE 层。"""

    def __init__(
        self,
        config: DyClustConfig,
        link_model: Optional[LinkCostModel] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.link_model = link_model or LinkCostModel.default_a100_topology(
            num_devices=config.num_devices, gpus_per_node=config.gpus_per_node
        )

        self.ffn = SharedBaseLowRankFFN(config, dtype=dtype, device=device)
        self.router = MassDistilledGroupRouter(config, dtype=dtype, device=device)
        self.dispatcher = DestinationCoalescedDispatcher(
            num_devices=config.num_devices,
            link_model=self.link_model,
            dtype_bytes=2 if dtype in (torch.float16, torch.bfloat16) else 4,
            coalesce=config.coalesce_tokens,
        )
        self.history = HistoryWindowCollector(config.num_experts, config.d_model)
        self.controller = OnlineRegroupingController(config, link_model=self.link_model)

        # 多状态解耦对照（仅当 state_coupling_mode != "single" 时使用独立的路由/放置分组缓冲）
        init_g = torch.tensor([i % config.num_groups for i in range(config.num_experts)], dtype=torch.long, device=device)
        self.register_buffer("routing_group_map", init_g.clone())
        self.register_buffer("placement_group_map", init_g.clone())
        self._batch_counter: int = 0
        self._decoupled_cycle_idx: int = 0

    @property
    def active_epoch(self) -> int:
        return self.controller.snapshot_store.active_epoch

    def sync_single_state(self) -> None:
        """强制同步单分组状态到表示、路由与放置三个接口。"""
        g_map = self.ffn.expert_to_group
        self.router.expert_to_group.copy_(g_map)
        self.routing_group_map.copy_(g_map)
        self.placement_group_map.copy_(g_map)

    def _compute_penalized_access_vector(self) -> Optional[torch.Tensor]:
        """为 Penalized-Independent 对照构造显式共享基/所有者分散访问惩罚向量。"""
        if self.config.state_coupling_mode != "penalized_independent":
            return None
        # 当某专家在路由分区与表示分区、或路由分区与放置分区不一致时施加显式访问代价惩罚
        rep_g = self.ffn.expert_to_group
        rt_g = self.router.expert_to_group
        pl_g = self.placement_group_map
        base_mismatch = (rep_g != rt_g).to(dtype=torch.float32)
        owner_mismatch = (pl_g != rt_g).to(dtype=torch.float32)
        return (
            self.config.base_access_penalty * base_mismatch
            + self.config.owner_access_penalty * owner_mismatch
        )

    def forward(
        self,
        hidden: torch.Tensor,
        record_history: bool = True,
        execution_mode_override: Optional[str] = None,
        support_restriction_override: Optional[bool] = None,
        token_sources: Optional[torch.Tensor] = None,
        ordinal_offset: int = 0,
    ) -> MoELayerOutput:
        """执行完整 MoE 前向：路由修复 -> 通信合并分析 -> 共享基/物化执行 -> 统计采集。"""
        orig_shape = hidden.shape
        if hidden.dim() == 3:
            flat_h = hidden.reshape(-1, orig_shape[-1])
        elif hidden.dim() == 2:
            flat_h = hidden
        else:
            raise ValueError(f"输入张量需为 2D 或 3D，当前形状为 {orig_shape}")

        # 在单状态模式下始终保证路由分组与表示分组严格一致
        if self.config.state_coupling_mode == "single":
            if not torch.equal(self.router.expert_to_group, self.ffn.expert_to_group):
                self.router.expert_to_group.copy_(self.ffn.expert_to_group)

        penalty_vec = self._compute_penalized_access_vector()
        r_out = self.router(
            flat_h,
            ordinal_offset=ordinal_offset,
            support_restriction_override=support_restriction_override,
            owner_access_penalty_vec=penalty_vec,
        )

        # 确定通信派发所用的放置分组（单状态与 Partial 共享路由/放置；完全解耦则用独立放置分组）
        if self.config.state_coupling_mode in ("single", "partial_route_place"):
            disp_group_map = self.router.expert_to_group
        else:
            disp_group_map = self.placement_group_map

        comm_stats = self.dispatcher.analyze_and_dispatch(
            hidden=flat_h,
            accepted_experts=r_out.accepted_experts,
            expert_to_group=disp_group_map,
            group_to_owner=self.router.group_to_owner,
            token_sources=token_sources,
        )

        y_flat, ffn_stats = self.ffn.forward_routed(
            flat_h,
            r_out.accepted_experts,
            r_out.accepted_weights,
            execution_mode=execution_mode_override,
        )

        if record_history and self.training:
            self._batch_counter += 1
            self.history.record_batch(
                flat_h,
                r_out.accepted_experts,
                r_out.accepted_weights,
                batch_doc_id=self._batch_counter,
                num_devices=self.config.num_devices,
            )

        y_out = y_flat.view(orig_shape)
        return MoELayerOutput(
            output=y_out,
            aux_loss=r_out.aux_loss,
            routing=r_out,
            comm_stats=comm_stats,
            bases_per_token=ffn_stats["bases_per_token"],
        )

    def step_online_regrouping(
        self,
        shadow_batch: Optional[torch.Tensor] = None,
        data_cursor: int = 0,
        rng_counter: int = 0,
    ) -> RegroupEventReport:
        """触发一次在线重组机会，按 state_coupling_mode 执行单状态或多状态协调更新。"""
        mode = self.config.state_coupling_mode
        if mode == "single":
            report = self.controller.try_regroup_layer(
                ffn=self.ffn,
                router=self.router,
                history=self.history,
                shadow_batch=shadow_batch,
                data_cursor=data_cursor,
                rng_counter=rng_counter,
            )
            if report.accepted:
                self.sync_single_state()
            self.history.reset()
            return report

        # 多状态解耦控制路径（用于复现论文 Table 6：Decoupled-3State / Matched-Schedule / Partial / Penalized）
        report = self._step_decoupled_state_update(shadow_batch, data_cursor, rng_counter)
        self.history.reset()
        return report

    def _step_decoupled_state_update(
        self,
        shadow_batch: Optional[torch.Tensor],
        data_cursor: int,
        rng_counter: int,
    ) -> RegroupEventReport:
        """执行解耦或部分耦合的多状态更新，严格保持 659M 参数预算与缓存查找语义。"""
        mode = self.config.state_coupling_mode
        cfg = self.config

        # 构造各独立分区的专用目标配置：
        # 1) 表示分区仅关注权重重构 J_cmp (lambda_act=0, lambda_pair=0)
        # 2) 路由分区仅关注输入质心与共激活 (lambda_act/pair 放大，弱化权重误差)
        # 3) 放置分区侧重共激活与通信合并
        old_rep_g = self.ffn.expert_to_group.clone()
        old_rt_g = self.router.expert_to_group.clone()
        old_pl_g = self.placement_group_map.clone()

        # 先运行一次标准重组作为表示分区更新（若轮到或同步更新）
        update_rep = (mode != "decoupled_cyclic") or (self._decoupled_cycle_idx % 3 == 0)
        update_rt = (mode != "decoupled_cyclic") or (self._decoupled_cycle_idx % 3 == 1)
        update_pl = (mode != "decoupled_cyclic") or (self._decoupled_cycle_idx % 3 == 2)
        self._decoupled_cycle_idx += 1

        # 用临时独立配置更新表示分区（只优化权重相似度）
        saved_act, saved_pair = cfg.lambda_act, cfg.lambda_pair
        try:
            if update_rep:
                cfg.lambda_act = 0.0
                cfg.lambda_pair = 0.0
                report = self.controller.try_regroup_layer(
                    ffn=self.ffn,
                    router=self.router,
                    history=self.history,
                    shadow_batch=shadow_batch,
                    data_cursor=data_cursor,
                    rng_counter=rng_counter,
                )
                # 表示更新不影响独立路由分区，恢复旧路由分组
                if not update_rt:
                    self.router.expert_to_group.copy_(old_rt_g)
            else:
                report = RegroupEventReport(
                    event_id=self.controller.event_counter,
                    layer_id=cfg.layer_id,
                    accepted=True,
                    reject_reason="",
                    rms_error=0.0,
                    max_tail_error=0.0,
                    cum_budget_before=self.controller.cum_distortion_budget,
                    cum_budget_after=self.controller.cum_distortion_budget,
                    output_drift_d_out=None,
                    shadow_kl=None,
                    shadow_retained_mass=None,
                    shadow_topk_recall=None,
                    bridge_max_nll_gap=None,
                    bridge_max_ppl_gap=None,
                    proposal_refit_sec=0.0,
                    bridge_sec=0.0,
                    other_pub_sec=0.0,
                    auxiliary_reread_tokens=0,
                )

            if update_rt:
                # 路由分区基于共激活与输入质心单独聚类（若为 Penalized-Independent 则加入向表示分区靠拢的访问罚项）
                centroids, exposed, chi = self.history.get_proposal_stats()
                pre_w = self.ffn.materialize_all_weights()
                cfg.lambda_act = 1.0
                cfg.lambda_pair = 1.0
                penalty_mat = None
                if mode == "penalized_independent":
                    import numpy as np
                    rep_arr = self.ffn.expert_to_group.detach().cpu().numpy()
                    penalty_mat = np.zeros((cfg.num_experts, cfg.num_groups), dtype=np.float64)
                    for i in range(cfg.num_experts):
                        for g in range(cfg.num_groups):
                            if g != int(rep_arr[i]):
                                penalty_mat[i, g] = cfg.base_access_penalty
                rt_prop = self.controller.grouper.propose_and_refit(
                    weights=pre_w,
                    input_centroids=centroids,
                    exposed_mask=exposed,
                    co_routing=chi,
                    incumbent_group_map=old_rt_g.cpu(),
                    run_seed=cfg.seed + 17,
                    layer_id=cfg.layer_id,
                    event_id=self.controller.event_counter,
                    skip_prescreen=True,
                    base_access_penalty_matrix=penalty_mat,
                )
                self.routing_group_map.copy_(rt_prop.expert_to_group.to(self.routing_group_map.device))
                self.router.expert_to_group.copy_(self.routing_group_map)
                new_u, new_a = self.router.remap_selector_from_overlap(rt_prop.overlap_remap)
                with torch.no_grad():
                    self.router.group_selector.weight.copy_(new_u)
                    if new_a is not None and self.router.group_selector.bias is not None:
                        self.router.group_selector.bias.copy_(new_a)

            if mode == "partial_route_place":
                # Partial: 路由与放置共享同一分区，仅表示独立
                self.placement_group_map.copy_(self.router.expert_to_group)
            elif update_pl:
                if mode == "penalized_independent":
                    # 带所有者访问惩罚时向路由分区对齐以减少跨设备访问
                    self.placement_group_map.copy_(self.router.expert_to_group)
                else:
                    # 独立放置分区（使用不同偏移的确定性分组）
                    self.placement_group_map.copy_(old_pl_g)

            # 基于最终放置分区重算设备所有者映射
            invocations_np = []
            pl_cpu = self.placement_group_map.detach().cpu()
            for src_t, exp_t in self.history.invocations:
                n_t = exp_t.shape[0]
                g_mask = torch.zeros((n_t, cfg.num_groups), dtype=torch.bool)
                mapped = pl_cpu.index_select(0, exp_t.reshape(-1)).view_as(exp_t)
                g_mask.scatter_(1, mapped, True)
                invocations_np.append((src_t.numpy(), g_mask.numpy()))
            if invocations_np:
                place_res = self.controller.placement_solver.solve(
                    num_groups=cfg.num_groups,
                    invocations=invocations_np,
                    incumbent_owner_map=self.router.group_to_owner.detach().cpu(),
                )
                self.router.group_to_owner.copy_(place_res.group_to_owner.to(self.router.group_to_owner.device))

            return report
        finally:
            cfg.lambda_act, cfg.lambda_pair = saved_act, saved_pair

    def scalar_ledger(self, mean_bases_per_token: float = 1.0) -> ScalarLedger:
        """返回本层存储与活跃参数账本。"""
        return self.ffn.compute_scalar_ledger(mean_bases_per_token=mean_bases_per_token)
