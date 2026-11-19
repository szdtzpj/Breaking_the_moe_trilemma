"""DyClust-MoE 在线重组门控、影子路由预检、64 切片插值桥、原子发布与崩溃恢复。

严格实现论文 Section 3.4、Appendix A.2 / A.4 (Proposition 4 & 5) 与 Appendix J.1：
- 不可变纪元快照存储与崩溃恢复状态机 (FENCED -> SHADOW_READY -> SLICE_PROGRESS -> UPDATE_READY -> COMMITTED/ABORTED)；
- SHA-256 全量张量/元数据哈希校验与线性化原子 CAS 发布，拒绝过期纪元请求（防父子混读）；
- 历史滑窗 7/8 提案视图 + 1/8 筛选视图统计采集器；
- 4 项重构与输出漂移门控 (b_{l,e}, max tau, B_l + b_{l,e}, D_out)、64 步影子路由预检、
  64 切片冻结路由参数插值桥（含即时切换 sidecar 对比）；
- 完整支持 Full DyClust-MoE、DyClust-MoE-Lite、阈值缩放 alpha 及全部端到端消融开关；
- 内置同状态反事实分叉检查器 (run_same_state_counterfactual_fork)。
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyclust_moe.config import DyClustConfig
from dyclust_moe.grouping import DeterministicGrouper, GroupingProposal
from dyclust_moe.placement import FixedChargePlacementSolver, LinkCostModel
from dyclust_moe.representation import SharedBaseLowRankFFN
from dyclust_moe.routing import MassDistilledGroupRouter


class EpochPhase(str, Enum):
    """Appendix A.4 定义的不可变纪元生成阶段。"""

    FENCED = "FENCED"
    SHADOW_READY = "SHADOW_READY"
    SLICE_PROGRESS = "SLICE_PROGRESS"
    UPDATE_READY = "UPDATE_READY"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


def hash_tensor_dict(tensors: Dict[str, torch.Tensor]) -> str:
    """按规范顺序对一组张量（含 dtype、shape、字节长度与原始字节）计算 SHA-256 摘要。"""
    h = hashlib.sha256()
    for key in sorted(tensors.keys()):
        t = tensors[key].detach().cpu().contiguous()
        raw = t.numpy().tobytes()
        meta = f"{key}|{str(t.dtype)}|{tuple(t.shape)}|{len(raw)}|".encode("ascii")
        h.update(meta)
        h.update(raw)
    return h.hexdigest()


@dataclass
class EpochSnapshotRecord:
    """不可变持久化快照记录（不原地修改父代缓冲）。"""

    generation_id: int
    parent_epoch: int
    candidate_epoch: int
    layer_id: int
    phase: EpochPhase
    completed_slice: int
    total_slices: int
    data_cursor: int
    rng_counter: int
    cum_distortion_budget: float
    tensor_payload: Dict[str, torch.Tensor]
    payload_sha256: str
    record_sha256: str
    rank_completion_bitmap: int
    world_size: int

    def verify_integrity(self) -> bool:
        """校验载荷哈希、记录哈希及全 rank 完成位图是否完整无损。"""
        if hash_tensor_dict(self.tensor_payload) != self.payload_sha256:
            return False
        expected_rec = _compute_record_hash(
            self.generation_id,
            self.parent_epoch,
            self.candidate_epoch,
            self.layer_id,
            self.phase.value,
            self.completed_slice,
            self.data_cursor,
            self.rng_counter,
            self.cum_distortion_budget,
            self.payload_sha256,
        )
        if expected_rec != self.record_sha256:
            return False
        full_mask = (1 << self.world_size) - 1
        return self.rank_completion_bitmap == full_mask


def _compute_record_hash(
    generation_id: int,
    parent_epoch: int,
    candidate_epoch: int,
    layer_id: int,
    phase_str: str,
    completed_slice: int,
    data_cursor: int,
    rng_counter: int,
    cum_budget: float,
    payload_sha256: str,
) -> str:
    msg = (
        f"{generation_id}|{parent_epoch}|{candidate_epoch}|{layer_id}|{phase_str}|"
        f"{completed_slice}|{data_cursor}|{rng_counter}|{cum_budget:.12g}|{payload_sha256}"
    ).encode("ascii")
    return hashlib.sha256(msg).hexdigest()


class EpochSnapshotStore:
    """崩溃安全纪元快照存储与线性化 CAS 激活器（Proposition 5）。"""

    def __init__(self, world_size: int = 1) -> None:
        self.world_size = world_size
        self.active_epoch: int = 0
        self.generations: Dict[int, EpochSnapshotRecord] = {}
        self.active_payload: Optional[Dict[str, torch.Tensor]] = None
        self._next_gen_id: int = 1

    def write_snapshot(
        self,
        parent_epoch: int,
        candidate_epoch: int,
        layer_id: int,
        phase: EpochPhase,
        completed_slice: int,
        total_slices: int,
        data_cursor: int,
        rng_counter: int,
        cum_budget: float,
        tensor_payload: Dict[str, torch.Tensor],
        rank_bitmap: Optional[int] = None,
    ) -> EpochSnapshotRecord:
        """向新的生成地址写入不可变快照，绝不原地覆写父代缓冲。"""
        cloned = {k: v.detach().cpu().clone() for k, v in tensor_payload.items()}
        p_hash = hash_tensor_dict(cloned)
        gen_id = self._next_gen_id
        self._next_gen_id += 1
        bitmap = ((1 << self.world_size) - 1) if rank_bitmap is None else rank_bitmap
        r_hash = _compute_record_hash(
            gen_id, parent_epoch, candidate_epoch, layer_id, phase.value,
            completed_slice, data_cursor, rng_counter, cum_budget, p_hash,
        )
        rec = EpochSnapshotRecord(
            generation_id=gen_id,
            parent_epoch=parent_epoch,
            candidate_epoch=candidate_epoch,
            layer_id=layer_id,
            phase=phase,
            completed_slice=completed_slice,
            total_slices=total_slices,
            data_cursor=data_cursor,
            rng_counter=rng_counter,
            cum_distortion_budget=cum_budget,
            tensor_payload=cloned,
            payload_sha256=p_hash,
            record_sha256=r_hash,
            rank_completion_bitmap=bitmap,
            world_size=self.world_size,
        )
        self.generations[gen_id] = rec
        return rec

    def atomic_cas_publish(self, record: EpochSnapshotRecord) -> bool:
        """单线性化点 CAS：仅当父纪元匹配且快照哈希与法定人数位图均通过验证时原子发布。"""
        if record.parent_epoch != self.active_epoch:
            return False
        if not record.verify_integrity():
            return False
        # 原子切换活跃纪元指针
        self.active_epoch = record.candidate_epoch
        self.active_payload = record.tensor_payload
        record.phase = EpochPhase.COMMITTED
        return True

    def validate_epoch_request(self, request_epoch: int) -> bool:
        """纪元标记请求校验：拒绝过期父纪元请求，保证调用方永不观测到混合状态。"""
        return request_epoch == self.active_epoch

    def plan_crash_recovery(self, record: EpochSnapshotRecord) -> Dict[str, object]:
        """按 Appendix A.4 确定崩溃后的精确重放策略。"""
        if not record.verify_integrity():
            return {
                "action": "abort_and_retain_parent",
                "replay_start_slice": None,
                "reapply_optimizer_update": False,
                "target_epoch": self.active_epoch,
            }
        if record.phase == EpochPhase.COMMITTED:
            return {
                "action": "already_committed",
                "replay_start_slice": None,
                "reapply_optimizer_update": False,
                "target_epoch": record.candidate_epoch,
            }
        if record.phase == EpochPhase.UPDATE_READY:
            # 无需重做优化器更新，直接完成 CAS 最终化
            return {
                "action": "finalize_without_reapplying_update",
                "replay_start_slice": None,
                "reapply_optimizer_update": False,
                "target_epoch": record.candidate_epoch,
            }
        if record.phase == EpochPhase.FENCED and record.completed_slice == 0:
            return {
                "action": "replay_full_batch",
                "replay_start_slice": 0,
                "reapply_optimizer_update": True,
                "target_epoch": record.candidate_epoch,
            }
        # SHADOW_READY 或 SLICE_PROGRESS：仅重放未完成切片及后缀
        return {
            "action": "replay_remaining_slices",
            "replay_start_slice": record.completed_slice,
            "reapply_optimizer_update": True,
            "target_epoch": record.candidate_epoch,
        }


class HistoryWindowCollector:
    """维护在线重组历史滑窗统计，并按文档哈希确定性切分 7/8 提案视图与 1/8 筛选视图。"""

    def __init__(self, num_experts: int, d_model: int, max_screen_tokens: int = 256) -> None:
        self.num_experts = num_experts
        self.d_model = d_model
        self.max_screen_tokens = max_screen_tokens
        self.reset()

    def reset(self) -> None:
        self.exposure_counts = torch.zeros(self.num_experts, dtype=torch.float64)
        self.input_sums = torch.zeros((self.num_experts, self.d_model), dtype=torch.float64)
        self.pair_counts = torch.zeros((self.num_experts, self.num_experts), dtype=torch.float64)
        self.screen_hidden: List[torch.Tensor] = []
        self.screen_experts: List[torch.Tensor] = []
        self.screen_weights: List[torch.Tensor] = []
        self.invocations: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (sources, group_mask)
        self._screen_token_count = 0

    @torch.no_grad()
    def record_batch(
        self,
        hidden: torch.Tensor,
        accepted_experts: torch.Tensor,
        accepted_weights: torch.Tensor,
        batch_doc_id: int,
        num_devices: int = 8,
    ) -> None:
        """按文档哈希取模 8 确定性划入 7/8 提案视图或 1/8 筛选视图（两视图严格互不相交）。"""
        digest = hashlib.sha256(f"doc-{batch_doc_id}".encode("ascii")).digest()
        is_screening = (digest[0] % 8) == 0

        h_cpu = hidden.detach().to(dtype=torch.float64, device="cpu")
        exp_cpu = accepted_experts.detach().to(dtype=torch.long, device="cpu")
        w_cpu = accepted_weights.detach().to(dtype=torch.float32, device="cpu")
        n_tok, top_k = exp_cpu.shape

        if is_screening:
            if self._screen_token_count < self.max_screen_tokens:
                take = min(n_tok, self.max_screen_tokens - self._screen_token_count)
                self.screen_hidden.append(hidden[:take].detach().cpu().clone())
                self.screen_experts.append(exp_cpu[:take].clone())
                self.screen_weights.append(w_cpu[:take].clone())
                self._screen_token_count += take
            return

        # 7/8 提案视图：累积曝光次数、输入激活质心与无序共激活对频次
        for t in range(n_tok):
            exps_t = exp_cpu[t].tolist()
            h_t = h_cpu[t]
            for e in exps_t:
                self.exposure_counts[e] += 1.0
                self.input_sums[e] += h_t
            if top_k > 1:
                for a in range(top_k):
                    for b in range(a + 1, top_k):
                        ea, eb = exps_t[a], exps_t[b]
                        self.pair_counts[ea, eb] += 1.0
                        self.pair_counts[eb, ea] += 1.0

        # 记录调用轨迹供设备级放置校准使用
        sources = (torch.arange(n_tok) * num_devices // max(1, n_tok)).clamp(max=num_devices - 1)
        if len(self.invocations) < 8:
            self.invocations.append((sources, exp_cpu.clone()))

    def get_proposal_stats(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回单位归一化专家质心 u_i、曝光掩码 zeta_i 与归一化对称共激活频次 chi_{ij}。"""
        exposed = (self.exposure_counts > 0).to(dtype=torch.float64)
        centroids = torch.zeros_like(self.input_sums)
        for e in range(self.num_experts):
            if self.exposure_counts[e] > 0:
                mean_vec = self.input_sums[e] / self.exposure_counts[e]
                nrm = float(torch.linalg.norm(mean_vec).item())
                if nrm > 1e-12:
                    centroids[e] = mean_vec / nrm

        pair_sum = float(torch.triu(self.pair_counts, diagonal=1).sum().item())
        if pair_sum > 0:
            chi = self.pair_counts / pair_sum
        else:
            chi = torch.zeros_like(self.pair_counts)
        return centroids, exposed, chi

    def get_screening_view(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """返回 1/8 提案不相交的筛选视图（hidden, accepted_experts, accepted_weights）。"""
        if not self.screen_hidden:
            return None
        return (
            torch.cat(self.screen_hidden, dim=0),
            torch.cat(self.screen_experts, dim=0),
            torch.cat(self.screen_weights, dim=0),
        )


@dataclass
class RegroupEventReport:
    """单次在线重组事件的完整审计记录（涵盖拒绝原因、门控值与互斥阶段耗时）。"""

    event_id: int
    layer_id: int
    accepted: bool
    reject_reason: str
    rms_error: float
    max_tail_error: float
    cum_budget_before: float
    cum_budget_after: float
    output_drift_d_out: Optional[float]
    shadow_kl: Optional[float]
    shadow_retained_mass: Optional[float]
    shadow_topk_recall: Optional[float]
    bridge_max_nll_gap: Optional[float]
    bridge_max_ppl_gap: Optional[float]
    proposal_refit_sec: float
    bridge_sec: float
    other_pub_sec: float
    auxiliary_reread_tokens: int
    rejected_solely_by_cum_budget: bool = False
    proposal: Optional[GroupingProposal] = None


class OnlineRegroupingController:
    """DyClust-MoE 在线重组控制器（支持 Full / Lite / 端到端门控消融与反事实分叉）。"""

    def __init__(
        self,
        config: DyClustConfig,
        link_model: Optional[LinkCostModel] = None,
        world_size: int = 1,
    ) -> None:
        self.config = config
        self.grouper = DeterministicGrouper(config)
        self.link_model = link_model or LinkCostModel.default_a100_topology(
            num_devices=config.num_devices, gpus_per_node=config.gpus_per_node
        )
        self.placement_solver = FixedChargePlacementSolver(
            link_model=self.link_model,
            num_devices=config.num_devices,
            payload_bytes_per_token=config.d_model * 2,
            time_limit_s=config.placement_time_limit_s,
        )
        self.snapshot_store = EpochSnapshotStore(world_size=world_size)

        # 账本状态
        self.cum_distortion_budget: float = 0.0   # B_l
        self.event_counter: int = 0
        self.accepted_events: int = 0
        self.total_aux_reread_tokens: int = 0
        self.total_proposal_refit_sec: float = 0.0
        self.total_bridge_sec: float = 0.0
        self.total_other_pub_sec: float = 0.0
        self.local_lr_ramp_remaining: int = 0
        self.event_history: List[RegroupEventReport] = []

    def get_local_lr_multiplier(self) -> float:
        """返回重组提交后 256 步局部学习率线性爬坡倍率，并向前推进一步。"""
        if self.local_lr_ramp_remaining <= 0:
            return 1.0
        total = max(1, self.config.local_lr_ramp_steps)
        step_done = total - self.local_lr_ramp_remaining + 1
        self.local_lr_ramp_remaining -= 1
        return min(1.0, float(step_done) / float(total))

    def try_regroup_layer(
        self,
        ffn: SharedBaseLowRankFFN,
        router: MassDistilledGroupRouter,
        history: HistoryWindowCollector,
        shadow_batch: Optional[torch.Tensor] = None,
        bridge_loss_fn: Optional[Callable[[torch.Tensor, Dict[str, torch.Tensor]], Tuple[float, float]]] = None,
        data_cursor: int = 0,
        rng_counter: int = 0,
        force_accept_proposal: Optional[GroupingProposal] = None,
    ) -> RegroupEventReport:
        """执行一次完整的 propose -> check admission -> publish atomically 生命周期。"""
        cfg = self.config
        thr = cfg.effective_thresholds
        event_id = self.event_counter
        self.event_counter += 1
        aux_rereads = 0

        # ---- 阶段 1：提案生成与精确 FP64 重拟合 ----
        t0 = time.perf_counter()
        pre_weights = {
            k: v.detach().clone() for k, v in ffn.materialize_all_weights().items()
        }
        centroids, exposed, chi = history.get_proposal_stats()

        if force_accept_proposal is not None:
            proposal = force_accept_proposal
        else:
            proposal = self.grouper.propose_and_refit(
                weights=pre_weights,
                input_centroids=centroids,
                exposed_mask=exposed,
                co_routing=chi,
                incumbent_group_map=ffn.expert_to_group.detach().cpu(),
                run_seed=cfg.seed,
                layer_id=cfg.layer_id,
                event_id=event_id,
                skip_prescreen=not cfg.enable_quality_gates,
            )
        t_refit = time.perf_counter() - t0
        self.total_proposal_refit_sec += t_refit

        cum_before = self.cum_distortion_budget
        b_le = proposal.rms_error
        tau_max = proposal.max_expert_rel_error

        # 探测预筛或收敛未过直接中止
        if not proposal.prescreen_passed or (cfg.enable_quality_gates and not proposal.converged):
            rep = RegroupEventReport(
                event_id=event_id, layer_id=cfg.layer_id, accepted=False,
                reject_reason=proposal.reject_reason or "prescreen_or_convergence_failed",
                rms_error=b_le, max_tail_error=tau_max,
                cum_budget_before=cum_before, cum_budget_after=cum_before,
                output_drift_d_out=None, shadow_kl=None, shadow_retained_mass=None,
                shadow_topk_recall=None, bridge_max_nll_gap=None, bridge_max_ppl_gap=None,
                proposal_refit_sec=t_refit, bridge_sec=0.0, other_pub_sec=0.0,
                auxiliary_reread_tokens=aux_rereads, proposal=proposal,
            )
            self.event_history.append(rep)
            return rep

        # ---- 阶段 2：失真门控、筛选输出漂移、累计预算检查 ----
        t_other_start = time.perf_counter()
        d_out: Optional[float] = None
        cand_weights = {
            name: proposal.bases[name].index_select(0, proposal.expert_to_group)
            + (
                torch.bmm(proposal.factor_A[name], proposal.factor_B[name].transpose(1, 2))
                if cfg.rank > 0
                else 0.0
            )
            for name in cfg.projection_names
        }
        dev = next(ffn.parameters()).device
        dt = next(ffn.parameters()).dtype
        cand_weights_dev = {k: v.to(device=dev, dtype=dt) for k, v in cand_weights.items()}

        if cfg.enable_quality_gates and force_accept_proposal is None:
            # 筛选视图输出漂移 D_out（Full 模式启用；Lite 模式跳过）
            if cfg.enable_output_drift_gate:
                screen = history.get_screening_view()
                if screen is not None:
                    s_h, s_exp, s_w = screen
                    aux_rereads += int(s_h.shape[0])
                    with torch.no_grad():
                        y_old, _ = ffn.forward_routed(
                            s_h.to(dev, dt), s_exp.to(dev), s_w.to(dev, dt),
                            execution_mode="materialized", override_weights=pre_weights,
                        )
                        y_new, _ = ffn.forward_routed(
                            s_h.to(dev, dt), s_exp.to(dev), s_w.to(dev, dt),
                            execution_mode="materialized", override_weights=cand_weights_dev,
                        )
                        num = float(((y_new - y_old) ** 2).sum().item())
                        den = float((y_old ** 2).sum().item()) + 1e-12
                        d_out = math.sqrt(num / den)

            step_ok = (b_le <= thr.rms_cap) and (tau_max <= thr.tail_cap)
            dout_ok = (d_out is None) or (d_out <= thr.output_drift_cap)
            cum_ok = (not cfg.enable_cum_ledger) or (cum_before + b_le <= thr.cum_budget)

            if not step_ok or not dout_ok or not cum_ok:
                solely_cum = step_ok and dout_ok and (not cum_ok)
                reason = (
                    "cum_budget_exceeded"
                    if solely_cum
                    else ("step_rms_or_tail_exceeded" if not step_ok else "output_drift_exceeded")
                )
                t_other = time.perf_counter() - t_other_start
                self.total_other_pub_sec += t_other
                self.total_aux_reread_tokens += aux_rereads
                rep = RegroupEventReport(
                    event_id=event_id, layer_id=cfg.layer_id, accepted=False,
                    reject_reason=reason, rms_error=b_le, max_tail_error=tau_max,
                    cum_budget_before=cum_before, cum_budget_after=cum_before,
                    output_drift_d_out=d_out, shadow_kl=None, shadow_retained_mass=None,
                    shadow_topk_recall=None, bridge_max_nll_gap=None, bridge_max_ppl_gap=None,
                    proposal_refit_sec=t_refit, bridge_sec=0.0, other_pub_sec=t_other,
                    auxiliary_reread_tokens=aux_rereads,
                    rejected_solely_by_cum_budget=solely_cum,
                    proposal=proposal,
                )
                self.event_history.append(rep)
                return rep

        # ---- 阶段 3：64 步影子路由预检（Full 模式启用；Lite 模式跳过） ----
        new_u, new_a = router.remap_selector_from_overlap(proposal.overlap_remap)
        shadow_kl: Optional[float] = None
        shadow_mass: Optional[float] = None
        shadow_recall: Optional[float] = None

        if cfg.enable_quality_gates and cfg.enable_shadow_preflight and shadow_batch is not None and force_accept_proposal is None:
            new_u, new_a, shadow_kl, shadow_mass, shadow_recall, s_rereads = self._run_shadow_preflight(
                router=router,
                new_group_map=proposal.expert_to_group.to(dev),
                init_u=new_u,
                init_a=new_a,
                shadow_batch=shadow_batch.to(dev, dt),
            )
            aux_rereads += s_rereads
            if (
                shadow_kl > thr.selector_kl_cap
                or shadow_mass < thr.retained_mass_floor
                or shadow_recall < thr.topk_recall_floor
            ):
                t_other = time.perf_counter() - t_other_start
                self.total_other_pub_sec += t_other
                self.total_aux_reread_tokens += aux_rereads
                rep = RegroupEventReport(
                    event_id=event_id, layer_id=cfg.layer_id, accepted=False,
                    reject_reason="shadow_preflight_failed", rms_error=b_le, max_tail_error=tau_max,
                    cum_budget_before=cum_before, cum_budget_after=cum_before,
                    output_drift_d_out=d_out, shadow_kl=shadow_kl,
                    shadow_retained_mass=shadow_mass, shadow_topk_recall=shadow_recall,
                    bridge_max_nll_gap=None, bridge_max_ppl_gap=None,
                    proposal_refit_sec=t_refit, bridge_sec=0.0, other_pub_sec=t_other,
                    auxiliary_reread_tokens=aux_rereads, proposal=proposal,
                )
                self.event_history.append(rep)
                return rep

        t_other = time.perf_counter() - t_other_start

        # ---- 阶段 4：64 切片冻结路由参数插值桥（Full 模式启用；Lite / w/o Bridge 跳过） ----
        bridge_sec = 0.0
        max_nll_gap: Optional[float] = None
        max_ppl_gap: Optional[float] = None
        parent_epoch = self.snapshot_store.active_epoch
        cand_epoch = parent_epoch + 1

        if cfg.enable_quality_gates and cfg.enable_bridge and shadow_batch is not None and force_accept_proposal is None:
            tb0 = time.perf_counter()
            max_nll_gap, max_ppl_gap, b_rereads = self._run_interpolation_bridge(
                ffn=ffn,
                router=router,
                pre_weights=pre_weights,
                cand_weights=cand_weights_dev,
                bridge_batch=shadow_batch.to(dev, dt),
                bridge_loss_fn=bridge_loss_fn,
            )
            bridge_sec = time.perf_counter() - tb0
            self.total_bridge_sec += bridge_sec
            aux_rereads += b_rereads

            if max_nll_gap >= thr.bridge_nll_cap or max_ppl_gap >= thr.bridge_ppl_cap:
                self.total_other_pub_sec += t_other
                self.total_aux_reread_tokens += aux_rereads
                rep = RegroupEventReport(
                    event_id=event_id, layer_id=cfg.layer_id, accepted=False,
                    reject_reason="bridge_path_gate_failed", rms_error=b_le, max_tail_error=tau_max,
                    cum_budget_before=cum_before, cum_budget_after=cum_before,
                    output_drift_d_out=d_out, shadow_kl=shadow_kl,
                    shadow_retained_mass=shadow_mass, shadow_topk_recall=shadow_recall,
                    bridge_max_nll_gap=max_nll_gap, bridge_max_ppl_gap=max_ppl_gap,
                    proposal_refit_sec=t_refit, bridge_sec=bridge_sec, other_pub_sec=t_other,
                    auxiliary_reread_tokens=aux_rereads, proposal=proposal,
                )
                self.event_history.append(rep)
                return rep

        # ---- 阶段 5：设备级放置更新与线性化原子 CAS 发布 ----
        t_pub0 = time.perf_counter()
        # 构造校准调用轨迹（把专家级调用投影到新分组上）
        g_new_np = proposal.expert_to_group.cpu().numpy()
        invocations_np = []
        for src_t, exp_t in history.invocations:
            n_t = exp_t.shape[0]
            g_mask = torch.zeros((n_t, cfg.num_groups), dtype=torch.bool)
            mapped = proposal.expert_to_group.index_select(0, exp_t.reshape(-1)).view_as(exp_t)
            g_mask.scatter_(1, mapped, True)
            invocations_np.append((src_t.numpy(), g_mask.numpy()))

        place_res = self.placement_solver.solve(
            num_groups=cfg.num_groups,
            invocations=invocations_np,
            incumbent_owner_map=router.group_to_owner.detach().cpu(),
        )

        # 写入不可变快照并通过哈希与 CAS 原子提交
        new_cum = cum_before + b_le
        payload: Dict[str, torch.Tensor] = {
            "expert_to_group": proposal.expert_to_group.detach().cpu(),
            "group_to_owner": place_res.group_to_owner.detach().cpu(),
            "selector_weight": new_u.detach().cpu(),
        }
        if new_a is not None:
            payload["selector_bias"] = new_a.detach().cpu()
        for name in cfg.projection_names:
            payload[f"base_{name}"] = proposal.bases[name].detach().cpu()
            payload[f"A_{name}"] = proposal.factor_A[name].detach().cpu()
            payload[f"B_{name}"] = proposal.factor_B[name].detach().cpu()

        snap = self.snapshot_store.write_snapshot(
            parent_epoch=parent_epoch,
            candidate_epoch=cand_epoch,
            layer_id=cfg.layer_id,
            phase=EpochPhase.UPDATE_READY,
            completed_slice=cfg.bridge_slices if cfg.enable_bridge else 0,
            total_slices=cfg.bridge_slices,
            data_cursor=data_cursor,
            rng_counter=rng_counter,
            cum_budget=new_cum,
            tensor_payload=payload,
        )
        published = self.snapshot_store.atomic_cas_publish(snap)
        if not published:
            t_other += time.perf_counter() - t_pub0
            self.total_other_pub_sec += t_other
            self.total_aux_reread_tokens += aux_rereads
            rep = RegroupEventReport(
                event_id=event_id, layer_id=cfg.layer_id, accepted=False,
                reject_reason="atomic_cas_failed", rms_error=b_le, max_tail_error=tau_max,
                cum_budget_before=cum_before, cum_budget_after=cum_before,
                output_drift_d_out=d_out, shadow_kl=shadow_kl,
                shadow_retained_mass=shadow_mass, shadow_topk_recall=shadow_recall,
                bridge_max_nll_gap=max_nll_gap, bridge_max_ppl_gap=max_ppl_gap,
                proposal_refit_sec=t_refit, bridge_sec=bridge_sec, other_pub_sec=t_other,
                auxiliary_reread_tokens=aux_rereads, proposal=proposal,
            )
            self.event_history.append(rep)
            return rep

        # CAS 成功后统一应用张量、路由选择器、设备放置表与累计账本
        with torch.no_grad():
            ffn.apply_partition_and_factors(
                proposal.expert_to_group, proposal.bases, proposal.factor_A, proposal.factor_B
            )
            router.set_group_and_owner_maps(proposal.expert_to_group, place_res.group_to_owner)
            router.group_selector.weight.copy_(new_u.to(router.group_selector.weight.device, router.group_selector.weight.dtype))
            if new_a is not None and router.group_selector.bias is not None:
                router.group_selector.bias.copy_(new_a.to(router.group_selector.bias.device, router.group_selector.bias.dtype))

        self.cum_distortion_budget = new_cum
        self.accepted_events += 1
        self.local_lr_ramp_remaining = cfg.local_lr_ramp_steps
        t_other += time.perf_counter() - t_pub0
        self.total_other_pub_sec += t_other
        self.total_aux_reread_tokens += aux_rereads

        rep = RegroupEventReport(
            event_id=event_id, layer_id=cfg.layer_id, accepted=True,
            reject_reason="", rms_error=b_le, max_tail_error=tau_max,
            cum_budget_before=cum_before, cum_budget_after=new_cum,
            output_drift_d_out=d_out, shadow_kl=shadow_kl,
            shadow_retained_mass=shadow_mass, shadow_topk_recall=shadow_recall,
            bridge_max_nll_gap=max_nll_gap, bridge_max_ppl_gap=max_ppl_gap,
            proposal_refit_sec=t_refit, bridge_sec=bridge_sec, other_pub_sec=t_other,
            auxiliary_reread_tokens=aux_rereads, proposal=proposal,
        )
        self.event_history.append(rep)
        return rep

    def _run_shadow_preflight(
        self,
        router: MassDistilledGroupRouter,
        new_group_map: torch.Tensor,
        init_u: torch.Tensor,
        init_a: Optional[torch.Tensor],
        shadow_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float, float, float, int]:
        """在 7/8 影子批上执行 64 步私有优化器预热，并在 1/8 保留切片上验证 KL / 质量 / top-k 召回。"""
        cfg = self.config
        n_tok = shadow_batch.shape[0]
        split = max(1, (7 * n_tok) // 8)
        train_h = shadow_batch[:split]
        val_h = shadow_batch[split:] if split < n_tok else shadow_batch

        with torch.no_grad():
            h_train_norm = router.input_norm(train_h)
            p_flat_tr, m_tr, _ = router.compute_flat_teacher(h_train_norm, expert_to_group=new_group_map)
            h_val_norm = router.input_norm(val_h)
            p_flat_val, m_val, _ = router.compute_flat_teacher(h_val_norm, expert_to_group=new_group_map)

        # 私有候选选择器 + 从零开始的私有 AdamW 动量（Appendix A.4）
        cand_sel = nn.Linear(
            cfg.d_model, cfg.num_groups, bias=(init_a is not None),
            device=shadow_batch.device, dtype=shadow_batch.dtype,
        )
        with torch.no_grad():
            cand_sel.weight.copy_(init_u)
            if init_a is not None and cand_sel.bias is not None:
                cand_sel.bias.copy_(init_a)

        opt = torch.optim.AdamW(cand_sel.parameters(), lr=5e-3, weight_decay=0.0)
        steps = max(1, cfg.shadow_steps)
        m_tr_safe = m_tr.detach().clamp_min(1e-12)

        for _ in range(steps):
            opt.zero_grad()
            log_pi = F.log_softmax(cand_sel(h_train_norm), dim=-1)
            kl = (m_tr_safe * (torch.log(m_tr_safe) - log_pi)).sum(dim=-1).mean()
            kl.backward()
            opt.step()

        # 在 1/8 保留切片上评估 KL、保留教师质量与教师 top-k 召回率
        with torch.no_grad():
            pi_val = F.softmax(cand_sel(h_val_norm), dim=-1).clamp_min(1e-12)
            m_val_safe = m_val.clamp_min(1e-12)
            val_kl = float((m_val_safe * (torch.log(m_val_safe) - torch.log(pi_val))).sum(dim=-1).mean().item())

            # 构造候选支持集检查保留质量与 top-k 召回
            sorted_p, sorted_g = torch.sort(pi_val, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_p, dim=-1)
            max_open = min(cfg.num_groups, 1 + cfg.max_extra_groups)
            nv = val_h.shape[0]
            open_g = torch.zeros((nv, cfg.num_groups), dtype=torch.bool, device=val_h.device)
            for pos in range(max_open):
                take = torch.ones(nv, dtype=torch.bool, device=val_h.device) if pos == 0 else (cumsum[:, pos - 1] < cfg.selector_mass_threshold)
                open_g[torch.arange(nv, device=val_h.device)[take], sorted_g[take, pos]] = True
            cand_mask = open_g.index_select(1, new_group_map)
            retained_mass = float(torch.where(cand_mask, p_flat_val, torch.zeros_like(p_flat_val)).sum(dim=-1).mean().item())
            topk_idx = torch.topk(p_flat_val, k=cfg.top_k, dim=-1).indices
            topk_recall = float(torch.gather(cand_mask, 1, topk_idx).float().mean().item())

        out_u = cand_sel.weight.detach().clone()
        out_a = cand_sel.bias.detach().clone() if cand_sel.bias is not None else None
        return out_u, out_a, val_kl, retained_mass, topk_recall, int(train_h.shape[0])

    def _run_interpolation_bridge(
        self,
        ffn: SharedBaseLowRankFFN,
        router: MassDistilledGroupRouter,
        pre_weights: Dict[str, torch.Tensor],
        cand_weights: Dict[str, torch.Tensor],
        bridge_batch: torch.Tensor,
        bridge_loss_fn: Optional[Callable[[torch.Tensor, Dict[str, torch.Tensor]], Tuple[float, float]]] = None,
    ) -> Tuple[float, float, int]:
        """在冻结路由下执行 H 切片参数插值桥（同时记录即时切换 sidecar 偏差）。"""
        cfg = self.config
        n_tok = bridge_batch.shape[0]
        h_slices = min(cfg.bridge_slices, max(1, n_tok))
        chunks = torch.chunk(bridge_batch, h_slices, dim=0)

        max_nll_gap = 0.0
        max_ppl_gap = 0.0

        with torch.no_grad():
            for s_idx, chunk in enumerate(chunks, start=1):
                if bridge_loss_fn is not None:
                    alpha_old = 1.0 - float(s_idx) / float(h_slices)
                    w_slice = {
                        k: cand_weights[k] + alpha_old * (pre_weights[k] - cand_weights[k])
                        for k in cfg.projection_names
                    }
                    nll_old, _ = bridge_loss_fn(chunk, pre_weights)
                    nll_grad, ppl_grad = bridge_loss_fn(chunk, w_slice)
                    _, ppl_imm = bridge_loss_fn(chunk, cand_weights)
                    d_nll = nll_grad - nll_old
                    d_ppl = abs(ppl_grad - ppl_imm)
                else:
                    # 无外部任务损失回调时，基于冻结路由的 FFN 相对输出能量漂移代理评估切片平滑度
                    r_out = router(chunk)
                    y_old, _ = ffn.forward_routed(
                        chunk, r_out.accepted_experts, r_out.accepted_weights,
                        execution_mode="materialized", override_weights=pre_weights,
                    )
                    y_grad = ffn.forward_interpolated(
                        chunk, r_out.accepted_experts, r_out.accepted_weights,
                        pre_weights, cand_weights, slice_idx=s_idx, num_slices=h_slices,
                    )
                    y_imm, _ = ffn.forward_routed(
                        chunk, r_out.accepted_experts, r_out.accepted_weights,
                        execution_mode="materialized", override_weights=cand_weights,
                    )
                    base_e = float((y_old ** 2).mean().item()) + 1e-12
                    d_nll = float(((y_grad - y_old) ** 2).mean().item()) / base_e
                    d_ppl = float(((y_grad - y_imm) ** 2).mean().item()) / base_e

                if d_nll > max_nll_gap:
                    max_nll_gap = d_nll
                if d_ppl > max_ppl_gap:
                    max_ppl_gap = d_ppl

        # 辅助重读记账：1 份带梯度桥分支 + 1 份前向即时切换 sidecar = 2 * B_e（Appendix A.4）
        return max_nll_gap, max_ppl_gap, int(2 * n_tok)


def run_same_state_counterfactual_fork(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    apply_rejected_candidate_fn: Callable[[nn.Module], None],
    step_fn: Callable[[nn.Module, torch.optim.Optimizer, int], None],
    eval_nll_fn: Callable[[nn.Module], float],
    fork_steps: int = 64,
) -> Dict[str, float]:
    """严格实现 Rebuttal / Appendix J.1 同状态 64 步反事实分叉检查。

    从完全相同的模型参数、优化器状态与数据游标深拷贝出两条分支：
    - 分支 A (accept): 强制接受仅因累计预算 B_l 被拒的候选，然后执行 fork_steps 步更新；
    - 分支 B (retain): 保留现役分组，在完全相同的批次顺序下执行 fork_steps 步更新；
    最后在固定验证批上对比 Delta NLL = NLL(accept) - NLL(retain)。
    """
    # 在同一 memo 中联合深拷贝 (model, optimizer)，确保 param_groups 与动量 state 100% 精确绑定至克隆参数
    model_accept, opt_accept = copy.deepcopy((model, optimizer))
    model_retain, opt_retain = copy.deepcopy((model, optimizer))

    # 在 accept 分支上应用候选提案
    apply_rejected_candidate_fn(model_accept)

    for step in range(fork_steps):
        step_fn(model_accept, opt_accept, step)
        step_fn(model_retain, opt_retain, step)

    nll_accept = float(eval_nll_fn(model_accept))
    nll_retain = float(eval_nll_fn(model_retain))
    return {
        "nll_accept": nll_accept,
        "nll_retain": nll_retain,
        "delta_nll": nll_accept - nll_retain,
        "worsened": 1.0 if (nll_accept > nll_retain) else 0.0,
    }
