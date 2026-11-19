"""DyClust-MoE 质量蒸馏分组路由、零丢弃循环保底与有界精确-k 并行容量修复。

严格实现论文 Section 3.2 (Eq. 1, Eq. 7--9)、Appendix A.3 (Proposition 3)、
Appendix F (Eq. 14 非归一化 L1 恒等式) 与 Appendix J.2 (Device-Preferred 对照)：
- 1/64 教师子采样质量蒸馏 L_mass、锐度正则 L_conf、专家与组负载均衡 L_lb^E / L_lb^G；
- 支持截断开关 (support_restriction=True/False)，直接支撑 Support-on/off 与 2x2 正交重放；
- 确定性循环保底 + 最多 16 轮单边替换/双 token 交换容量修复，严格满足零丢弃、精确 k 互异专家与容量上限；
- 归一化 TV 误差分解诊断与公版非归一化精确子概率 L1 恒等式诊断；
- 受控 DeepSeek-V2 风格单优选设备路由 (device_preferred) + 跨设备溢出修复。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyclust_moe.config import DyClustConfig


@dataclass
class RoutingOutput:
    """单次路由前向的结构化输出与诊断账本。"""

    accepted_experts: torch.Tensor        # (N, k) 接受的专家 ID（每行严格 k 个互异）
    accepted_weights: torch.Tensor        # (N, k) 接受权重 q_hat（可导）
    aux_loss: torch.Tensor                # 标量辅助损失（不含 L_task）
    loss_components: Dict[str, float]     # l_lb_e, l_lb_g, l_conf, l_mass
    diagnostics: Dict[str, float]         # omitted_mass, router_tv, hard_tv, total_tv, l1_err, repair_rounds 等
    candidate_mask: torch.Tensor          # (N, E) 布尔支持集掩码 S
    supported_probs: torch.Tensor         # (N, E) 支持集概率 q^S


def compute_cyclic_reserve(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    ordinal_offset: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """构造零丢弃确定性循环保底分配 e^(0)_{t,j} = (k * nu_t + j) mod E。

    Proposition 3 保证：当 k <= E 且每专家容量 C >= ceil(k * N / E) 时，
    该初始分配严格满足每个 token 恰有 k 个互异专家、任意两专家负载差 <= 1。
    """
    nu = torch.arange(ordinal_offset, ordinal_offset + num_tokens, dtype=torch.long, device=device)
    j = torch.arange(top_k, dtype=torch.long, device=device)
    return (nu.unsqueeze(1) * top_k + j.unsqueeze(0)) % num_experts


def bounded_exact_k_repair(
    scores: torch.Tensor,
    allowed_mask: torch.Tensor,
    reserve_experts: torch.Tensor,
    capacity: int,
    max_rounds: int = 16,
    bucket_limit: int = 8,
) -> Tuple[torch.Tensor, int, float]:
    """严格实现 Appendix A.3 有界精确-k 并行容量修复。

    从零丢弃循环保底出发，最小化代理代价 sum -A_{ti} log(q_{ti}^S + 1e-12)：
    - 每轮同时生成单 token 直接替换（有空余容量目标）与双 token 边交换（有序专家对桶上限 8）；
    - 按代价降幅降序 + 字典序进行稳定前缀预留并批量提交；
    - 每次提交严格保持：每 token 精确 k 个互异专家、每专家负载 <= capacity、零丢弃。

    Returns:
        (accepted_experts, rounds_used, overflow_rate):
            accepted_experts 形状 (N, k)；rounds_used 为实际迭代轮数；
            overflow_rate 为最终接受边中来自原支持集之外（保底/溢出修复）的比例。
    """
    n_tokens, num_experts = scores.shape
    top_k = reserve_experts.shape[1]
    min_cap = math.ceil(top_k * n_tokens / float(num_experts))
    if capacity < min_cap:
        raise ValueError(f"容量 {capacity} 小于可行下界 ceil(kN/E)={min_cap}")

    # 转为 CPU NumPy 标量运算确保完全确定性与精确字典序破平
    prob_np = scores.detach().to(dtype=torch.float64, device="cpu").numpy()
    allow_np = allowed_mask.detach().to(dtype=torch.bool, device="cpu").numpy().copy()
    assign = reserve_experts.detach().to(dtype=torch.long, device="cpu").numpy().copy()

    # 将保底边纳入允许边集 E_t = S_t union reserve_t
    orig_support_np = allow_np.copy()
    for t in range(n_tokens):
        for slot in range(top_k):
            allow_np[t, int(assign[t, slot])] = True

    costs = -np.log(np.maximum(prob_np, 0.0) + 1e-12)

    # 计算初始负载与每 token 接受集合
    loads = [0] * num_experts
    accepted_sets: List[set[int]] = []
    for t in range(n_tokens):
        s_t = set(int(assign[t, slot]) for slot in range(top_k))
        accepted_sets.append(s_t)
        for e in s_t:
            loads[e] += 1

    rounds_used = 0
    for r in range(1, max_rounds + 1):
        rounds_used = r
        # 1. 构造单 token 直接替换候选（不限桶数，每 token 取最优正增益替换）与交换桶
        # 交换桶键 (i, j) -> 最多保留 bucket_limit=8 个最佳 (gain, token_idx)
        direct_moves: List[Tuple[float, int, int, int]] = []  # (gain, t, i, j)
        buckets: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}

        for t in range(n_tokens):
            acc_t = accepted_sets[t]
            allow_row = allow_np[t]
            cost_row = costs[t]
            # 找出当前 token 允许但未接受的候选目的专家
            unacc_allowed = [j for j in range(num_experts) if allow_row[j] and j not in acc_t]
            if not unacc_allowed:
                continue

            best_direct_gain = 1e-12
            best_direct: Optional[Tuple[float, int, int, int]] = None

            for i in acc_t:
                ci = float(cost_row[i])
                for j in unacc_allowed:
                    gain = ci - float(cost_row[j])
                    # 直接替换（仅当目标专家 j 有空余容量）
                    if loads[j] < capacity and gain > best_direct_gain:
                        best_direct_gain = gain
                        best_direct = (gain, t, i, j)
                    # 记录进交换候选桶（即使单边 gain <= 0，只要对向边足够大，总交换增益仍可为正）
                    key = (i, j)
                    lst = buckets.get(key)
                    if lst is None:
                        buckets[key] = [(gain, t)]
                    else:
                        lst.append((gain, t))

            if best_direct is not None:
                direct_moves.append(best_direct)

        # 对每个有序专家对桶保留前 bucket_limit 个最高 gain（平局按 token 序号升序）
        for key, lst in buckets.items():
            lst.sort(key=lambda x: (-x[0], x[1]))
            if len(lst) > bucket_limit:
                buckets[key] = lst[:bucket_limit]

        # 2. 配对反向桶生成两 token 交换候选 (total_gain > 1e-12)
        # 统一格式：("direct", gain, tie_key, payload) / ("swap", gain, tie_key, payload)
        candidates: List[Tuple[float, Tuple[int, ...], str, Tuple[int, ...]]] = []
        for gain, t, i, j in direct_moves:
            candidates.append((gain, (t, i, j, -1), "direct", (t, i, j)))

        seen_unordered_pairs = set()
        for (i, j), fwd_list in buckets.items():
            if (j, i) not in buckets:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_unordered_pairs:
                continue
            seen_unordered_pairs.add(pair_key)
            rev_list = buckets[(j, i)]
            for g1, t1 in fwd_list:
                for g2, t2 in rev_list:
                    if t1 == t2:
                        continue
                    tot_gain = g1 + g2
                    if tot_gain > 1e-12:
                        # 规范排序保证字典序确定性
                        if t1 < t2:
                            tie = (t1, i, j, t2)
                            payload = (t1, i, j, t2, j, i)
                        else:
                            tie = (t2, j, i, t1)
                            payload = (t2, j, i, t1, i, j)
                        candidates.append((tot_gain, tie, "swap", payload))

        if not candidates:
            break

        # 3. 合并按增益降序、字典序升序执行稳定前缀预留提交
        candidates.sort(key=lambda x: (-x[0], x[1]))
        used_tokens = set()
        committed = 0
        for _, _, kind, payload in candidates:
            if kind == "direct":
                t, i, j = payload
                if t in used_tokens:
                    continue
                if loads[j] + 1 > capacity or i not in accepted_sets[t] or j in accepted_sets[t]:
                    continue
                used_tokens.add(t)
                accepted_sets[t].remove(i)
                accepted_sets[t].add(j)
                loads[i] -= 1
                loads[j] += 1
                committed += 1
            else:
                t1, i1, j1, t2, i2, j2 = payload
                if t1 in used_tokens or t2 in used_tokens:
                    continue
                if (
                    i1 not in accepted_sets[t1]
                    or j1 in accepted_sets[t1]
                    or i2 not in accepted_sets[t2]
                    or j2 in accepted_sets[t2]
                ):
                    continue
                # 双 token 交换净负载变化为 0，永远保持容量可行
                used_tokens.add(t1)
                used_tokens.add(t2)
                accepted_sets[t1].remove(i1)
                accepted_sets[t1].add(j1)
                accepted_sets[t2].remove(i2)
                accepted_sets[t2].add(j2)
                committed += 1

        if committed == 0:
            break

    # 按专家概率降序（平局按专家 ID 升序）输出最终接受矩阵并统计保底/溢出比例
    out_assign = torch.empty((n_tokens, top_k), dtype=torch.long)
    non_support_edges = 0
    for t in range(n_tokens):
        exps = sorted(accepted_sets[t], key=lambda e: (-float(prob_np[t, e]), e))
        out_assign[t] = torch.tensor(exps, dtype=torch.long)
        for e in exps:
            if not orig_support_np[t, e]:
                non_support_edges += 1

    overflow_rate = non_support_edges / float(max(1, n_tokens * top_k))
    return out_assign.to(scores.device), rounds_used, overflow_rate


import numpy as np  # noqa: E402


class MassDistilledGroupRouter(nn.Module):
    """DyClust-MoE 质量蒸馏分组路由器（支持 grouped / flat / device_preferred 三模式）。"""

    def __init__(
        self,
        config: DyClustConfig,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.num_groups = config.num_groups
        self.group_size = config.group_size
        self.top_k = config.top_k

        # 输入层归一化（与论文 Eq. 7 前置 LN(h_t) 对齐）
        self.input_norm = nn.LayerNorm(config.d_model, eps=1e-5, device=device, dtype=dtype)
        # 专家路由器: l_{ti} = v_i^T LN(h_t) + b_i
        self.expert_router = nn.Linear(config.d_model, config.num_experts, bias=True, device=device, dtype=dtype)
        # 轻量组选择器: pi_t = softmax(U LN(h_t) + a)（公版适配器可设 bias=False）
        self.group_selector = nn.Linear(
            config.d_model, config.num_groups, bias=config.selector_bias, device=device, dtype=dtype
        )

        # 初始分组映射与设备映射（默认均分到 num_devices）
        init_group = torch.tensor([i % self.num_groups for i in range(self.num_experts)], dtype=torch.long, device=device)
        init_owner = torch.tensor([g % config.num_devices for g in range(self.num_groups)], dtype=torch.long, device=device)
        self.register_buffer("expert_to_group", init_group)
        self.register_buffer("group_to_owner", init_owner)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """严格按 Appendix B.1 初始化专家路由器与组选择器。"""
        std = math.sqrt(1e-6 / float(self.d_model))
        nn.init.normal_(self.expert_router.weight, mean=0.0, std=std)
        nn.init.zeros_(self.expert_router.bias)
        self.init_selector_from_expert_router(self.expert_to_group)

    @torch.no_grad()
    def init_selector_from_expert_router(self, expert_to_group: torch.Tensor) -> None:
        """按 Appendix B.1 初始化组选择器：U_g = mean_{i in G_g} v_i, a_g = log K + mean b_i。"""
        g_map = expert_to_group.to(self.expert_router.weight.device)
        self.expert_to_group.copy_(g_map)
        for g in range(self.num_groups):
            idx = torch.nonzero(g_map == g, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            self.group_selector.weight[g].copy_(self.expert_router.weight.index_select(0, idx).mean(dim=0))
            if self.group_selector.bias is not None:
                mean_b = self.expert_router.bias.index_select(0, idx).mean()
                self.group_selector.bias[g].copy_(math.log(float(self.group_size)) + mean_b)

    @torch.no_grad()
    def remap_selector_from_overlap(self, omega: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """按 Appendix A.4 计算重叠重映射后的候选选择器参数 (U^+, a^+) = omega @ (U^-, a^-)。"""
        om = omega.to(device=self.group_selector.weight.device, dtype=self.group_selector.weight.dtype)
        new_u = om @ self.group_selector.weight.detach()
        new_a = (om @ self.group_selector.bias.detach()) if self.group_selector.bias is not None else None
        return new_u, new_a

    def set_group_and_owner_maps(
        self,
        expert_to_group: torch.Tensor,
        group_to_owner: Optional[torch.Tensor] = None,
    ) -> None:
        """原子更新路由层持有的分组与设备放置表。"""
        self.expert_to_group.copy_(expert_to_group.to(self.expert_to_group.device))
        if group_to_owner is not None:
            self.group_to_owner.copy_(group_to_owner.to(self.group_to_owner.device))

    def compute_flat_teacher(
        self,
        h_norm: torch.Tensor,
        expert_to_group: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算平坦教师概率 p^flat、组质量 m_g = sum_{i in G_g} p_i^flat 与组内条件分布 rho_{i|g}。"""
        g_map = self.expert_to_group if expert_to_group is None else expert_to_group
        logits = self.expert_router(h_norm)  # (N, E)
        p_flat = F.softmax(logits, dim=-1)

        # 聚合每组总质量 m_g
        n_tokens = h_norm.shape[0]
        m_group = p_flat.new_zeros((n_tokens, self.num_groups))
        m_group.scatter_add_(1, g_map.unsqueeze(0).expand(n_tokens, -1), p_flat)

        # 计算组内条件概率 rho_{i|g} = softmax_{j in G_g}(l_j)
        rho = torch.empty_like(p_flat)
        for g in range(self.num_groups):
            idx = torch.nonzero(g_map == g, as_tuple=False).squeeze(-1)
            rho.index_copy_(1, idx, F.softmax(logits.index_select(1, idx), dim=-1))

        return p_flat, m_group, rho

    def _build_group_support_mask(
        self,
        pi: torch.Tensor,
        reserve_experts: torch.Tensor,
    ) -> torch.Tensor:
        """按 Section 3.2 构造候选支持集掩码 S（(N, E) 布尔张量）。"""
        n_tokens = pi.shape[0]
        if not self.config.support_restriction or self.config.routing_mode == "flat":
            return torch.ones((n_tokens, self.num_experts), dtype=torch.bool, device=pi.device)

        # 按选择器概率降序开组：必开最高组，累加至达到阈值或最多追加 max_extra_groups 个组
        sorted_probs, sorted_groups = torch.sort(pi, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        max_open = min(self.num_groups, 1 + self.config.max_extra_groups)

        # 确定每个 token 打开哪些组
        open_groups = torch.zeros((n_tokens, self.num_groups), dtype=torch.bool, device=pi.device)
        for rank_pos in range(max_open):
            if rank_pos == 0:
                take = torch.ones(n_tokens, dtype=torch.bool, device=pi.device)
            else:
                # 若前一个位置的累加质量尚未达到阈值，则继续打开当前组
                take = cumsum[:, rank_pos - 1] < self.config.selector_mass_threshold
            g_col = sorted_groups[:, rank_pos]
            open_groups[torch.arange(n_tokens, device=pi.device)[take], g_col[take]] = True

        # 并入 k 个确定性循环保底专家所在组
        res_groups = self.expert_to_group.index_select(0, reserve_experts.reshape(-1)).view(n_tokens, self.top_k)
        open_groups.scatter_(1, res_groups, True)

        # 映射回专家维度 (N, E)
        return open_groups.index_select(1, self.expert_to_group)

    def _build_device_preferred_mask(
        self,
        p_flat: torch.Tensor,
    ) -> torch.Tensor:
        """构造 DeepSeek-V2 风格单优选设备支持掩码（Appendix J.2 Device-Preferred 对照）。"""
        n_tokens = p_flat.shape[0]
        expert_owner = self.group_to_owner.index_select(0, self.expert_to_group)  # (E,)
        num_dev = self.config.num_devices

        # 按设备内最大专家亲和度选择单一优选设备
        dev_affinity = p_flat.new_full((n_tokens, num_dev), -1.0)
        for d in range(num_dev):
            idx = torch.nonzero(expert_owner == d, as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                dev_affinity[:, d] = p_flat.index_select(1, idx).max(dim=-1).values
        pref_dev = torch.argmax(dev_affinity, dim=-1)  # (N,)
        return expert_owner.unsqueeze(0) == pref_dev.unsqueeze(1)

    def forward(
        self,
        hidden: torch.Tensor,
        ordinal_offset: int = 0,
        support_restriction_override: Optional[bool] = None,
        owner_access_penalty_vec: Optional[torch.Tensor] = None,
    ) -> RoutingOutput:
        """执行完整路由前向：打分 -> 候选支持集 -> 有界精确-k 修复 -> 权重计算 -> 辅助损失与诊断。"""
        n_tokens = hidden.shape[0]
        h_norm = self.input_norm(hidden)
        cfg = self.config

        # 1. 计算专家 logits、组选择器概率 pi 与平坦教师分布
        logits = self.expert_router(h_norm)
        p_flat = F.softmax(logits, dim=-1)
        pi = F.softmax(self.group_selector(h_norm), dim=-1)

        # 聚合组质量 m_g 与组内条件分布 rho_{i|g}
        m_group = p_flat.new_zeros((n_tokens, self.num_groups))
        m_group.scatter_add_(1, self.expert_to_group.unsqueeze(0).expand(n_tokens, -1), p_flat)
        rho = torch.empty_like(p_flat)
        for g in range(self.num_groups):
            idx = torch.nonzero(self.expert_to_group == g, as_tuple=False).squeeze(-1)
            rho.index_copy_(1, idx, F.softmax(logits.index_select(1, idx), dim=-1))

        # 2. 构造循环保底与候选支持集掩码 S
        reserve = compute_cyclic_reserve(
            n_tokens, self.num_experts, self.top_k, ordinal_offset=ordinal_offset, device=hidden.device
        )
        eff_support_on = cfg.support_restriction if support_restriction_override is None else support_restriction_override

        if cfg.routing_mode == "device_preferred":
            cand_mask = self._build_device_preferred_mask(p_flat)
        elif cfg.routing_mode == "flat" or not eff_support_on:
            cand_mask = torch.ones((n_tokens, self.num_experts), dtype=torch.bool, device=hidden.device)
        else:
            cand_mask = self._build_group_support_mask(pi, reserve)

        # 3. 计算支持集概率 q^S（Eq. 7）或公版非归一化原生概率（norm_topk_prob=False）
        if cfg.norm_topk_prob:
            if cfg.routing_mode == "grouped":
                pi_per_exp = pi.index_select(1, self.expert_to_group)
                unnorm_q = torch.where(cand_mask, pi_per_exp * rho, torch.zeros_like(rho))
            else:
                unnorm_q = torch.where(cand_mask, p_flat, torch.zeros_like(p_flat))
            z_s = unnorm_q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            q_s = unnorm_q / z_s
        else:
            # 公版大模型（OLMoE / Qwen1.5-MoE）：始终使用全专家原生 softmax 权重，不重归一化
            q_s = p_flat

        # 若为 Penalized-Independent 对照传入额外访问代价惩罚，调整修复评分（不改变输出权重语义）
        repair_scores = q_s
        if owner_access_penalty_vec is not None:
            penalty = owner_access_penalty_vec.to(device=q_s.device, dtype=q_s.dtype).unsqueeze(0)
            repair_scores = q_s * torch.exp(-penalty)

        # 4. 执行零丢弃有界精确-k 容量修复
        capacity = math.ceil(cfg.capacity_factor * self.top_k * n_tokens / float(self.num_experts))
        accepted_experts, rounds_used, overflow_rate = bounded_exact_k_repair(
            scores=repair_scores,
            allowed_mask=cand_mask,
            reserve_experts=reserve,
            capacity=capacity,
            max_rounds=cfg.repair_max_rounds,
            bucket_limit=cfg.repair_bucket_limit,
        )

        # 5. 计算最终接受权重 q_hat（对离散分配 A_{ti} 施加 stop-gradient）
        gathered = torch.gather(q_s, 1, accepted_experts)  # (N, k)
        if cfg.norm_topk_prob:
            accepted_weights = gathered / gathered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            # norm_topk_prob=False：直接采用原生全 softmax 概率，不做重归一化
            accepted_weights = gathered

        # 6. 计算 Eq. (1) 全部四项辅助损失
        a_ind = torch.zeros_like(p_flat)
        a_ind.scatter_(1, accepted_experts, 1.0)

        p_i = q_s.mean(dim=0)
        f_i = a_ind.sum(dim=0) / float(max(1, self.top_k * n_tokens))
        l_lb_e = float(self.num_experts) * torch.sum(p_i * f_i)

        p_g = pi.mean(dim=0)
        # C_{tg} = 1{exists i in G_g: A_{ti} = 1}
        c_tg = torch.zeros((n_tokens, self.num_groups), dtype=p_flat.dtype, device=hidden.device)
        acc_groups = self.expert_to_group.index_select(0, accepted_experts.reshape(-1)).view(n_tokens, self.top_k)
        c_tg.scatter_(1, acc_groups, 1.0)
        f_g = c_tg.sum(dim=0) / c_tg.sum().clamp_min(1.0)
        l_lb_g = float(self.num_groups) * torch.sum(p_g * f_g)

        # L_conf: 向 stop-gradient 最高概率组锐化
        top_g = torch.argmax(pi.detach(), dim=-1)
        b0 = F.one_hot(top_g, num_classes=self.num_groups).to(dtype=pi.dtype)
        l_conf = (1.0 - (b0 * pi).sum(dim=-1)).mean()

        # L_mass: 在 1/64 教师子采样上计算 KL(sg(m) || pi)
        sub_n = max(1, int(round(n_tokens * cfg.teacher_subsample_ratio)))
        stride = max(1, n_tokens // sub_n)
        m_sub = m_group[::stride].detach().clamp_min(1e-12)
        pi_sub = pi[::stride].clamp_min(1e-12)
        l_mass = (m_sub * (torch.log(m_sub) - torch.log(pi_sub))).sum(dim=-1).mean()

        aux_loss = (
            cfg.lambda_lb_expert * l_lb_e
            + cfg.lambda_lb_group * l_lb_g
            + cfg.lambda_conf * l_conf
            + cfg.lambda_mass * l_mass
        )

        # 7. 计算 Eq. (9) TV 分解诊断与 Eq. (14) 非归一化精确子概率 L1 恒等式诊断
        with torch.no_grad():
            q_hat_full = torch.zeros_like(p_flat)
            q_hat_full.scatter_(1, accepted_experts, accepted_weights)

            omitted_mass = (1.0 - torch.where(cand_mask, p_flat, torch.zeros_like(p_flat)).sum(dim=-1)).clamp_min(0.0)
            p_tilde = torch.where(cand_mask, p_flat, torch.zeros_like(p_flat))
            p_tilde = p_tilde / p_tilde.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            router_tv = 0.5 * torch.abs(p_tilde - q_s).sum(dim=-1)
            hard_tv = 0.5 * torch.abs(q_s - q_hat_full).sum(dim=-1)
            total_tv = 0.5 * torch.abs(p_flat - q_hat_full).sum(dim=-1)

            # 非归一化子概率 L1 精确恒等式 (Eq. 14): ||r_t - r_hat_t||_1 = p(T\A) + p(A\T)
            topk_teacher = torch.topk(p_flat, k=self.top_k, dim=-1).indices
            t_mask = torch.zeros_like(cand_mask)
            t_mask.scatter_(1, topk_teacher, True)
            a_mask = a_ind > 0.5
            r_teacher = torch.where(t_mask, p_flat, torch.zeros_like(p_flat))
            r_accept = torch.where(a_mask, p_flat, torch.zeros_like(p_flat))
            l1_direct = torch.abs(r_teacher - r_accept).sum(dim=-1).mean().item()
            l1_symdiff = (
                torch.where(t_mask & (~a_mask), p_flat, torch.zeros_like(p_flat)).sum(dim=-1)
                + torch.where(a_mask & (~t_mask), p_flat, torch.zeros_like(p_flat)).sum(dim=-1)
            ).mean().item()

            # 教师 top-k 召回率（被候选支持集覆盖的教师 top-k 比例）
            topk_covered = torch.gather(cand_mask, 1, topk_teacher).float().mean().item()

            # 统计每 token 打开组数与触达设备所有者数
            open_groups_cnt = cand_mask.float().sum(dim=-1).mean().item() / float(self.group_size)
            acc_owners = self.group_to_owner.index_select(0, acc_groups.reshape(-1)).view(n_tokens, self.top_k)
            sorted_ow, _ = torch.sort(acc_owners, dim=1)
            if self.top_k > 1:
                uniq_owners = 1 + (sorted_ow[:, 1:] != sorted_ow[:, :-1]).sum(dim=1)
            else:
                uniq_owners = torch.ones(n_tokens, dtype=torch.long, device=hidden.device)
            owners_per_token = uniq_owners.float().mean().item()

        return RoutingOutput(
            accepted_experts=accepted_experts,
            accepted_weights=accepted_weights,
            aux_loss=aux_loss,
            loss_components={
                "l_lb_e": float(l_lb_e.item()),
                "l_lb_g": float(l_lb_g.item()),
                "l_conf": float(l_conf.item()),
                "l_mass": float(l_mass.item()),
            },
            diagnostics={
                "omitted_mass": float(omitted_mass.mean().item()),
                "router_tv": float(router_tv.mean().item()),
                "hard_tv": float(hard_tv.mean().item()),
                "total_tv": float(total_tv.mean().item()),
                "l1_subprob_direct": float(l1_direct),
                "l1_subprob_identity": float(l1_symdiff),
                "topk_recall": float(topk_covered),
                "retained_mass": float(1.0 - omitted_mass.mean().item()),
                "open_groups_per_token": float(open_groups_cnt),
                "owners_per_token": float(owners_per_token),
                "repair_rounds": float(rounds_used),
                "overflow_rate": float(overflow_rate),
            },
            candidate_mask=cand_mask,
            supported_probs=q_s,
        )
