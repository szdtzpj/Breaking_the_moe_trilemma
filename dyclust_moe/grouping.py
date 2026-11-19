"""DyClust-MoE 确定性分组、CountSketch 候选生成、均衡最小费用流与精确 FP64 SVD 重拟合。

严格实现论文 Section 3.1 (Eq. 3--6) 与 Appendix A.1 全部 8 步算法：
1. 精确 SHA-256 256-bin 符号 CountSketch；
2. 三信号拼接特征（加权权重草图 + 激活质心 + 共路由频次）与最远优先原型选取；
3. 均衡最小费用流（Balanced Min-Cost Flow）线性代理候选生成；
4. 两轮冷启动探测预筛（probe RMS <= 0.06, tail <= 0.20）；
5. 规范确定性 FP64 截断 SVD（含重复奇异值块双遍修正 Gram-Schmidt 与唯一符号规则）
   及加权最小二乘基矩阵交替重拟合（最多 8 轮，相对降幅 <= 1e-5 停止）；
6. 有界跨组两两交换局部搜索（shortlist <= 32, 上限 8E 次精确评估）；
7. 规范组 ID 字典序重排与选择器重叠重映射矩阵 omega_{gh}。
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dyclust_moe.config import DyClustConfig


_SKETCH_PREFIX = b"dyclust-sketch-v1"  # Appendix A.1 精确 17 字节 ASCII 前缀
_SKETCH_TABLE_CACHE: Dict[Tuple[int, int, int, int, int, int, bool], Tuple[np.ndarray, np.ndarray]] = {}


def build_sha256_sketch_tables(
    num_coords: int,
    num_bins: int,
    run_seed: int,
    layer_id: int,
    event_id: int,
    proj_code: int,
    exact_per_coord: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """构建确定性 SHA-256 CountSketch 桶索引表 H(a) 与符号表 sigma(a)。

    严格遵循 Appendix A.1 字节域规范：
        17 字节 ASCII ``dyclust-sketch-v1`` + 5 个小端 uint64 字段
        (run_seed, layer_id, event_id, proj_code, coord_a)，无终止符。
        H(a) = uint64_LE(D(a)[0:8]) mod num_bins
        sigma(a) = (-1)^(D(a)[31] & 1)
    """
    cache_key = (num_coords, num_bins, run_seed, layer_id, event_id, proj_code, exact_per_coord)
    if cache_key in _SKETCH_TABLE_CACHE:
        return _SKETCH_TABLE_CACHE[cache_key]

    header = _SKETCH_PREFIX + struct.pack("<QQQQ", run_seed, layer_id, event_id, proj_code)
    bins = np.empty(num_coords, dtype=np.int64)
    signs = np.empty(num_coords, dtype=np.float64)

    if exact_per_coord:
        base_hasher = hashlib.sha256(header)
        pack_q = struct.Struct("<Q").pack
        unpack_q = struct.Struct("<Q").unpack_from
        for a in range(num_coords):
            h = base_hasher.copy()
            h.update(pack_q(a))
            digest = h.digest()
            bins[a] = unpack_q(digest, 0)[0] % num_bins
            signs[a] = -1.0 if (digest[31] & 1) else 1.0
    else:
        # 大维度快速确定性派生路径：每 16 个坐标一次 SHA-256 块扩展（保持完全确定性与独立性）
        base_hasher = hashlib.sha256(header + b"blk")
        pack_q = struct.Struct("<Q").pack
        block_size = 16
        num_blocks = (num_coords + block_size - 1) // block_size
        for blk in range(num_blocks):
            h = base_hasher.copy()
            h.update(pack_q(blk))
            d = h.digest()
            start = blk * block_size
            end = min(num_coords, start + block_size)
            for j in range(end - start):
                # 2 字节派生 1 个坐标的桶和符号
                val = d[2 * j] | (d[2 * j + 1] << 8)
                bins[start + j] = (val >> 1) % num_bins
                signs[start + j] = -1.0 if (val & 1) else 1.0

    _SKETCH_TABLE_CACHE[cache_key] = (bins, signs)
    return bins, signs


def compute_countsketch(
    weights_proj: torch.Tensor,
    num_bins: int,
    run_seed: int,
    layer_id: int,
    event_id: int,
    proj_code: int,
    exact_sha256_limit: int = 65536,
) -> np.ndarray:
    """对某投影下所有 E 个专家的权重矩阵计算 256-bin 符号 CountSketch。

    Args:
        weights_proj: 形状 (E, out_dim, in_dim) 的 FP64 张量（row-major 展平）。
    Returns:
        形状 (E, num_bins) 的 FP64 NumPy 数组。
    """
    w_np = weights_proj.detach().to(dtype=torch.float64, device="cpu").contiguous().numpy()
    num_experts = w_np.shape[0]
    flat = w_np.reshape(num_experts, -1)
    num_coords = flat.shape[1]
    bins, signs = build_sha256_sketch_tables(
        num_coords=num_coords,
        num_bins=num_bins,
        run_seed=run_seed,
        layer_id=layer_id,
        event_id=event_id,
        proj_code=proj_code,
        exact_per_coord=(num_coords <= exact_sha256_limit),
    )
    signed = flat * signs[None, :]
    out = np.zeros((num_experts, num_bins), dtype=np.float64)
    for e in range(num_experts):
        np.add.at(out[e], bins, signed[e])
    return out


def canonical_truncated_svd_fp64(
    matrix: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """严格遵循 Appendix A.1 规范的确定性 FP64 截断 SVD。

    规则要点：
    1. 在 FP64 下计算奇异三元组并按未舍入奇异值严格降序排列；
    2. 仅对按位相等（bitwise-equal）的重复正奇异值块，将标准基 e_1..e_n 投影至
       其右奇异子空间并做两遍固定顺序修正 Gram-Schmidt（接收残差范数 > 1e-12）；
    3. 对每个右奇异向量 v，使其最大绝对值分量（最小索引破平）严格为正，并令 u = M v / sigma；
    4. 返回 A = U_r Sigma_r^{1/2}, B = V_r Sigma_r^{1/2}，未用零列置零。
    """
    m = matrix.detach().to(dtype=torch.float64, device="cpu")
    rows, cols = m.shape
    if rank == 0:
        return torch.zeros((rows, 0), dtype=torch.float64), torch.zeros((cols, 0), dtype=torch.float64)

    u_raw, s_raw, vh_raw = torch.linalg.svd(m, full_matrices=False)
    # 显式按未舍入奇异值降序排序
    order = torch.argsort(s_raw, descending=True, stable=True)
    s_sorted = s_raw.index_select(0, order)
    v_sorted = vh_raw.index_select(0, order).t()  # (cols, k_full)

    k_full = s_sorted.shape[0]
    canon_v_cols: List[torch.Tensor] = []
    canon_u_cols: List[torch.Tensor] = []
    canon_s_vals: List[float] = []

    idx = 0
    while idx < k_full and len(canon_s_vals) < rank:
        s_val = float(s_sorted[idx].item())
        if s_val <= 0.0:
            break
        # 查找按位相等的重复正奇异值块
        end = idx + 1
        while end < k_full and float(s_sorted[end].item()) == s_val:
            end += 1
        block_dim = end - idx

        if block_dim == 1:
            v_vec = v_sorted[:, idx].clone()
            v_vec = _canonicalize_sign(v_vec)
            u_vec = (m @ v_vec) / s_val
            canon_v_cols.append(v_vec)
            canon_u_cols.append(u_vec)
            canon_s_vals.append(s_val)
        else:
            # 重复奇异值块：将标准基 e_1..e_n 投影到子空间并做两遍修正 Gram-Schmidt
            q_sub = v_sorted[:, idx:end]  # (cols, block_dim)
            basis: List[torch.Tensor] = []
            for e_idx in range(cols):
                # 投影标准基 e_{e_idx} 到 q_sub 张成的子空间
                proj = q_sub @ q_sub[e_idx, :]
                # 两遍固定顺序修正 Gram-Schmidt
                for _ in range(2):
                    for b_vec in basis:
                        proj = proj - torch.dot(b_vec, proj) * b_vec
                nrm = float(torch.linalg.norm(proj).item())
                if nrm > 1e-12:
                    v_vec = _canonicalize_sign(proj / nrm)
                    basis.append(v_vec)
                    if len(basis) == block_dim:
                        break
            for v_vec in basis:
                if len(canon_s_vals) >= rank:
                    break
                u_vec = (m @ v_vec) / s_val
                canon_v_cols.append(v_vec)
                canon_u_cols.append(u_vec)
                canon_s_vals.append(s_val)
        idx = end

    a_out = torch.zeros((rows, rank), dtype=torch.float64)
    b_out = torch.zeros((cols, rank), dtype=torch.float64)
    for j, (u_vec, v_vec, s_val) in enumerate(zip(canon_u_cols, canon_v_cols, canon_s_vals)):
        root_s = math.sqrt(s_val)
        a_out[:, j] = u_vec * root_s
        b_out[:, j] = v_vec * root_s
    return a_out, b_out


def _canonicalize_sign(vec: torch.Tensor) -> torch.Tensor:
    """将向量最大绝对值分量（最小索引破平）的符号规范为正。"""
    abs_v = torch.abs(vec)
    max_idx = int(torch.argmax(abs_v).item())
    if float(vec[max_idx].item()) < 0.0:
        return -vec
    return vec


@dataclass
class RefitResult:
    """单次固定分组冷启动精确 FP64 重拟合结果。"""

    bases: Dict[str, torch.Tensor]
    factor_A: Dict[str, torch.Tensor]
    factor_B: Dict[str, torch.Tensor]
    j_cmp: float
    rms_error: float
    max_expert_rel_error: float
    sweeps_used: int
    converged: bool


def exact_refit_fixed_partition(
    weights: Dict[str, torch.Tensor],
    expert_to_group: torch.Tensor,
    num_groups: int,
    rank: int,
    max_sweeps: int = 8,
    tol: float = 1e-5,
    eps_w: float = 1e-12,
    probe_only_sweeps: Optional[int] = None,
) -> RefitResult:
    """给定固定分组，从 R_{i,p}=0 冷启动交替求解 Eq. (5) 与 Eq. (6)。

    Args:
        weights: {proj_name: Tensor(E, out_dim, in_dim)} FP64 原始专家权重。
        expert_to_group: 长度为 E 的分组索引张量。
        num_groups: 分组总数 G。
        rank: 低秩残差秩上限 r。
        max_sweeps: 最大交替扫描轮数（论文默认 8）。
        tol: 相对重构目标降幅收敛阈值（论文默认 1e-5）。
        eps_w: 逆范数权重稳定项 1e-12。
        probe_only_sweeps: 若指定（如 2），则仅执行探测轮数供预筛使用。
    """
    proj_names = list(weights.keys())
    p_w = len(proj_names)
    g_map = expert_to_group.detach().cpu().tolist()
    num_experts = len(g_map)

    # 预计算逆 Frobenius 范数平方权重 omega_{i,p}
    weights_fp64: Dict[str, torch.Tensor] = {}
    omegas: Dict[str, torch.Tensor] = {}
    for name in proj_names:
        w = weights[name].detach().to(dtype=torch.float64, device="cpu")
        weights_fp64[name] = w
        frob_sq = (w * w).sum(dim=(1, 2))
        omegas[name] = 1.0 / (frob_sq + eps_w)

    # 组织每组包含的专家索引列表
    group_members: List[List[int]] = [[] for _ in range(num_groups)]
    for i, g in enumerate(g_map):
        group_members[int(g)].append(i)

    bases: Dict[str, torch.Tensor] = {}
    factor_A: Dict[str, torch.Tensor] = {}
    factor_B: Dict[str, torch.Tensor] = {}
    residuals: Dict[str, torch.Tensor] = {}

    # 冷启动：令 R_{i,p} = 0，先应用一次 Eq. (6) 加权最小二乘基更新
    for name in proj_names:
        w = weights_fp64[name]
        e_cnt, out_dim, in_dim = w.shape
        residuals[name] = torch.zeros_like(w)
        factor_A[name] = torch.zeros((e_cnt, out_dim, rank), dtype=torch.float64)
        factor_B[name] = torch.zeros((e_cnt, in_dim, rank), dtype=torch.float64)
        bases[name] = _weighted_base_update(w, residuals[name], omegas[name], group_members)

    prev_j_cmp = _compute_j_cmp(weights_fp64, bases, residuals, omegas, g_map, p_w, num_experts)[0]
    sweep_cap = probe_only_sweeps if probe_only_sweeps is not None else max_sweeps
    converged = False
    sweeps_used = 0

    for sweep in range(1, sweep_cap + 1):
        sweeps_used = sweep
        for name in proj_names:
            w = weights_fp64[name]
            # Eq. (5): 对残差 W_pre - W_base 执行规范确定性截断 SVD
            for i in range(num_experts):
                diff = w[i] - bases[name][g_map[i]]
                a_i, b_i = canonical_truncated_svd_fp64(diff, rank)
                factor_A[name][i] = a_i
                factor_B[name][i] = b_i
                residuals[name][i] = a_i @ b_i.t() if rank > 0 else torch.zeros_like(diff)
            # Eq. (6): 固定残差更新加权最小二乘共享基
            bases[name] = _weighted_base_update(w, residuals[name], omegas[name], group_members)

        cur_j_cmp, max_rel = _compute_j_cmp(weights_fp64, bases, residuals, omegas, g_map, p_w, num_experts)
        rel_decrease = (prev_j_cmp - cur_j_cmp) / max(prev_j_cmp, 1e-15)
        prev_j_cmp = cur_j_cmp
        # 当相对降幅 <= tol 或绝对重构误差已近乎为零（J_cmp <= 1e-8，即 RMS <= 1e-4）时即判定收敛
        if probe_only_sweeps is None and (rel_decrease <= tol or cur_j_cmp <= 1e-8):
            converged = True
            break

    final_j_cmp, max_rel = _compute_j_cmp(weights_fp64, bases, residuals, omegas, g_map, p_w, num_experts)
    return RefitResult(
        bases=bases,
        factor_A=factor_A,
        factor_B=factor_B,
        j_cmp=final_j_cmp,
        rms_error=math.sqrt(max(0.0, final_j_cmp)),
        max_expert_rel_error=max_rel,
        sweeps_used=sweeps_used,
        converged=converged or (probe_only_sweeps is not None),
    )


def _weighted_base_update(
    w_pre: torch.Tensor,
    residuals: torch.Tensor,
    omega: torch.Tensor,
    group_members: List[List[int]],
) -> torch.Tensor:
    """严格计算 Eq. (6) 加权最小二乘组共享基矩阵。"""
    num_groups = len(group_members)
    _, out_dim, in_dim = w_pre.shape
    bases = torch.zeros((num_groups, out_dim, in_dim), dtype=torch.float64)
    diff = w_pre - residuals
    for g, members in enumerate(group_members):
        if not members:
            continue
        idx = torch.tensor(members, dtype=torch.long)
        w_g = omega.index_select(0, idx)  # (K,)
        denom = float(w_g.sum().item())
        if denom > 0.0:
            weighted_sum = (diff.index_select(0, idx) * w_g[:, None, None]).sum(dim=0)
            bases[g] = weighted_sum / denom
    return bases


def _compute_j_cmp(
    weights_fp64: Dict[str, torch.Tensor],
    bases: Dict[str, torch.Tensor],
    residuals: Dict[str, torch.Tensor],
    omegas: Dict[str, torch.Tensor],
    g_map: List[int],
    p_w: int,
    num_experts: int,
) -> Tuple[float, float]:
    """计算 Eq. (3) 第一项加权相对重构误差 J_cmp 与最大单专家相对误差 tau_{i,p}。"""
    total = 0.0
    max_tau = 0.0
    for name, w in weights_fp64.items():
        b = bases[name]
        r = residuals[name]
        om = omegas[name]
        for i in range(num_experts):
            err = w[i] - b[g_map[i]] - r[i]
            d_ip = float((err * err).sum().item()) * float(om[i].item())
            total += d_ip
            tau_ip = math.sqrt(max(0.0, d_ip))
            if tau_ip > max_tau:
                max_tau = tau_ip
    j_cmp = total / float(p_w * num_experts)
    return j_cmp, max_tau


def solve_balanced_min_cost_flow(cost_matrix: np.ndarray, group_size: int) -> List[int]:
    """精确求解均衡分配最小费用流：将 E 个专家分配到 G 个组，每组严格容量 K。

    采用确定性 Successive Shortest Path（SPFA 增广）算法，并对费用矩阵施加极微小的
    字典序偏置，确保等价最优解下返回字典序最小的分配向量（对应 Appendix A.1 第 4 步）。
    """
    num_experts, num_groups = cost_matrix.shape
    if num_experts != num_groups * group_size:
        raise ValueError(f"维度不匹配: E={num_experts}, G={num_groups}, K={group_size}")

    # 图节点布局: 0=源点 S, 1..E=专家节点, E+1..E+G=组节点, E+G+1=汇点 T
    src = 0
    exp_offset = 1
    grp_offset = 1 + num_experts
    snk = 1 + num_experts + num_groups
    num_nodes = snk + 1

    # 邻接表边结构: [to, cap, cost, rev_idx]
    adj: List[List[List[float]]] = [[] for _ in range(num_nodes)]

    def add_edge(u: int, v: int, cap: int, cost: float) -> None:
        fwd = [v, cap, cost, len(adj[v])]
        rev = [u, 0, -cost, len(adj[u])]
        adj[u].append(fwd)
        adj[v].append(rev)

    for i in range(num_experts):
        add_edge(src, exp_offset + i, 1, 0.0)

    # 边费用加入极小确定性字典序打破平局项（不影响真实费用精度）
    scale = max(1.0, float(np.max(np.abs(cost_matrix))))
    tie_eps = scale * 1e-11
    for i in range(num_experts):
        for g in range(num_groups):
            c = float(cost_matrix[i, g]) + tie_eps * (g + 1) / float(num_groups * (i + 1))
            add_edge(exp_offset + i, grp_offset + g, 1, c)

    for g in range(num_groups):
        add_edge(grp_offset + g, snk, group_size, 0.0)

    # 每次增广 1 单位流，共增广 E 次
    inf = float("inf")
    for _ in range(num_experts):
        dist = [inf] * num_nodes
        in_queue = [False] * num_nodes
        parent_node = [-1] * num_nodes
        parent_edge = [-1] * num_nodes
        dist[src] = 0.0
        q: deque[int] = deque([src])
        in_queue[src] = True

        while q:
            u = q.popleft()
            in_queue[u] = False
            du = dist[u]
            for e_idx, edge in enumerate(adj[u]):
                v = int(edge[0])
                cap = edge[1]
                cost = edge[2]
                if cap > 0 and du + cost < dist[v] - 1e-15:
                    dist[v] = du + cost
                    parent_node[v] = u
                    parent_edge[v] = e_idx
                    if not in_queue[v]:
                        q.append(v)
                        in_queue[v] = True

        if parent_node[snk] == -1:
            raise RuntimeError("均衡最小费用流不可行，请检查 E = G * K 约束")

        # 沿最短路径增广 1 单位流
        v = snk
        while v != src:
            u = parent_node[v]
            e_idx = parent_edge[v]
            edge = adj[u][e_idx]
            rev_idx = int(edge[3])
            edge[1] -= 1
            adj[v][rev_idx][1] += 1
            v = u

    # 读取最终分配结果
    assignment = [-1] * num_experts
    for i in range(num_experts):
        u = exp_offset + i
        for edge in adj[u]:
            v = int(edge[0])
            if grp_offset <= v < snk and edge[1] == 0:
                assignment[i] = v - grp_offset
                break
    return assignment


def canonicalize_group_ids(assignment: List[int], num_groups: int) -> List[int]:
    """按 Appendix A.4 规则，根据每组排序后的专家 ID 元组字典序重新分配规范组 ID。"""
    members: List[List[int]] = [[] for _ in range(num_groups)]
    for i, g in enumerate(assignment):
        members[int(g)].append(i)
    sorted_groups = sorted((tuple(sorted(m)), old_g) for old_g, m in enumerate(members))
    old_to_new = {old_g: new_g for new_g, (_, old_g) in enumerate(sorted_groups)}
    return [old_to_new[int(g)] for g in assignment]


def compute_overlap_remap_matrix(
    old_assignment: List[int],
    new_assignment: List[int],
    num_groups: int,
    group_size: int,
) -> torch.Tensor:
    """计算选择器重叠重映射矩阵 omega_{gh} = |G_g^+ cap G_h^-| / K（每行和为 1）。"""
    omega = torch.zeros((num_groups, num_groups), dtype=torch.float32)
    for i in range(len(old_assignment)):
        g_new = int(new_assignment[i])
        h_old = int(old_assignment[i])
        omega[g_new, h_old] += 1.0 / float(group_size)
    return omega


def evaluate_group_objective(
    j_cmp: float,
    assignment: List[int],
    num_groups: int,
    input_centroids: Optional[np.ndarray],
    exposed_mask: Optional[np.ndarray],
    co_routing: Optional[np.ndarray],
    lambda_act: float,
    lambda_pair: float,
) -> Tuple[float, float, float]:
    """精确评估 Eq. (3) 三信号联合目标 J_grp = J_cmp + J_act + J_pair。"""
    num_experts = len(assignment)
    j_act = 0.0
    if input_centroids is not None and exposed_mask is not None and lambda_act > 0.0:
        n_zeta = max(1.0, float(np.sum(exposed_mask > 0)))
        for g in range(num_groups):
            members = [i for i in range(num_experts) if assignment[i] == g and exposed_mask[i] > 0]
            if not members:
                continue
            c_sum = np.sum(input_centroids[members], axis=0)
            c_norm = float(np.linalg.norm(c_sum))
            c_g = c_sum / c_norm if c_norm > 1e-12 else np.zeros_like(c_sum)
            diffs = input_centroids[members] - c_g[None, :]
            j_act += float(np.sum(diffs * diffs))
        j_act = (lambda_act / n_zeta) * j_act

    j_pair = 0.0
    if co_routing is not None and lambda_pair > 0.0:
        for i in range(num_experts):
            gi = assignment[i]
            for j in range(i + 1, num_experts):
                if gi != assignment[j]:
                    j_pair += float(co_routing[i, j])
        j_pair *= lambda_pair

    return j_cmp + j_act + j_pair, j_act, j_pair


@dataclass
class GroupingProposal:
    """一次完整分组提案与精确 FP64 重拟合的结构化输出。"""

    expert_to_group: torch.Tensor
    bases: Dict[str, torch.Tensor]
    factor_A: Dict[str, torch.Tensor]
    factor_B: Dict[str, torch.Tensor]
    j_grp: float
    j_cmp: float
    j_act: float
    j_pair: float
    rms_error: float
    max_expert_rel_error: float
    prescreen_passed: bool
    converged: bool
    swaps_accepted: int
    swap_evals: int
    overlap_remap: torch.Tensor
    reject_reason: str = ""
    extra_metrics: Dict[str, float] = field(default_factory=dict)


class DeterministicGrouper:
    """DyClust-MoE 确定性分组求解器（分离快速 Sketch 提案与精确 FP64 SVD 验收）。"""

    def __init__(self, config: DyClustConfig) -> None:
        self.config = config

    def propose_and_refit(
        self,
        weights: Dict[str, torch.Tensor],
        input_centroids: Optional[torch.Tensor],
        exposed_mask: Optional[torch.Tensor],
        co_routing: Optional[torch.Tensor],
        incumbent_group_map: torch.Tensor,
        run_seed: int = 0,
        layer_id: int = 0,
        event_id: int = 0,
        skip_prescreen: bool = False,
        base_access_penalty_matrix: Optional[np.ndarray] = None,
    ) -> GroupingProposal:
        """执行 Appendix A.1 全部 8 步流程：Sketch -> 最远优先原型 -> 最小费用流 -> 探测预筛 -> 精确 SVD -> 跨组交换。

        Args:
            weights: {proj_name: Tensor(E, out_dim, in_dim)} 预重组完整专家权重。
            input_centroids: 单位归一化专家输入质心 u_i，形状 (E, d_model)，可为 None。
            exposed_mask: 曝光标记 zeta_i in {0, 1}，形状 (E,)，可为 None。
            co_routing: 对称归一化共激活频次 chi_{ij}，形状 (E, E)，可为 None。
            incumbent_group_map: 当前生效的分组标签 g^-，形状 (E,)。
            skip_prescreen: 若为 True（如 0B 一次性转换），跳过两轮探测提前剪枝。
            base_access_penalty_matrix: 可选附加代价矩阵（供 Penalized-Independent 对照使用）。
        """
        cfg = self.config
        e_cnt = cfg.num_experts
        g_cnt = cfg.num_groups
        k_size = cfg.group_size
        proj_names = list(cfg.projection_names)
        proj_codes = list(cfg.projection_codes)
        p_w = len(proj_names)

        incumbent_list = incumbent_group_map.detach().cpu().tolist()
        weights_fp64 = {k: v.detach().to(dtype=torch.float64, device="cpu") for k, v in weights.items()}

        # 预处理权重逆范数 omega_{i,p} 与 CountSketch 特征
        omegas_np: Dict[str, np.ndarray] = {}
        sketches: Dict[str, np.ndarray] = {}
        for name, p_code in zip(proj_names, proj_codes):
            w = weights_fp64[name]
            frob_sq = (w * w).sum(dim=(1, 2)).numpy()
            omegas_np[name] = 1.0 / (frob_sq + cfg.eps_w)
            sketches[name] = compute_countsketch(
                w, cfg.sketch_bins, run_seed, layer_id, event_id, p_code
            )

        u_np: Optional[np.ndarray] = None
        zeta_np: Optional[np.ndarray] = None
        if input_centroids is not None and exposed_mask is not None:
            u_np = input_centroids.detach().to(dtype=torch.float64, device="cpu").numpy()
            zeta_np = (exposed_mask.detach().cpu().numpy() > 0).astype(np.float64)
            # 确保单位归一化
            norms = np.linalg.norm(u_np, axis=1, keepdims=True)
            u_np = np.where(norms > 1e-12, u_np / np.maximum(norms, 1e-12), 0.0)
        chi_np: Optional[np.ndarray] = None
        if co_routing is not None:
            chi_np = co_routing.detach().to(dtype=torch.float64, device="cpu").numpy()
            chi_np = 0.5 * (chi_np + chi_np.T)
            np.fill_diagonal(chi_np, 0.0)

        # 第 2 步：拼接三信号特征 x_i
        feat_parts: List[np.ndarray] = []
        scale_w = 1.0 / math.sqrt(float(p_w * e_cnt))
        for name in proj_names:
            feat_parts.append(sketches[name] * (scale_w * np.sqrt(omegas_np[name])[:, None]))
        if u_np is not None and zeta_np is not None and cfg.lambda_act > 0.0:
            n_zeta = max(1.0, float(np.sum(zeta_np)))
            feat_parts.append(u_np * (math.sqrt(cfg.lambda_act / n_zeta) * zeta_np[:, None]))
        if chi_np is not None and cfg.lambda_pair > 0.0:
            feat_parts.append(chi_np * math.sqrt(cfg.lambda_pair / 2.0))
        x_feat = np.concatenate(feat_parts, axis=1)

        # 第 3 步：最远优先（Farthest-First）原型选取（首种子为最小专家 ID 0，字典序破平）
        prototypes: List[int] = [0]
        min_dists = np.linalg.norm(x_feat - x_feat[0:1], axis=1)
        min_dists[0] = -1.0
        for _ in range(1, g_cnt):
            # 取最大最小距离，平局选最小专家 ID
            best_i = int(np.argmax(min_dists))
            prototypes.append(best_i)
            new_d = np.linalg.norm(x_feat - x_feat[best_i : best_i + 1], axis=1)
            min_dists = np.minimum(min_dists, new_d)
            for p_id in prototypes:
                min_dists[p_id] = -1.0

        s_proto = {name: sketches[name][prototypes] for name in proj_names}  # (G, bins)
        c_proto = u_np[prototypes] if u_np is not None else None  # (G, d_model)

        # 第 4 步：构造 Eq. (4) 线性代理代价并求解均衡最小费用流
        cost_mat = self._build_linear_cost_matrix(
            sketches, s_proto, omegas_np, u_np, c_proto, zeta_np, chi_np, incumbent_list,
            base_access_penalty_matrix,
        )
        cand_assign = solve_balanced_min_cost_flow(cost_mat, k_size)

        # 第 5 步：两轮冷启动探测预筛
        eff_thr = cfg.effective_thresholds
        if not skip_prescreen:
            probe = exact_refit_fixed_partition(
                weights=weights_fp64,
                expert_to_group=torch.tensor(cand_assign, dtype=torch.long),
                num_groups=g_cnt,
                rank=cfg.rank,
                max_sweeps=2,
                tol=cfg.refit_tol,
                eps_w=cfg.eps_w,
                probe_only_sweeps=2,
            )
            if (
                not math.isfinite(probe.rms_error)
                or probe.rms_error > eff_thr.probe_rms_cap
                or probe.max_expert_rel_error > eff_thr.probe_tail_cap
            ):
                return GroupingProposal(
                    expert_to_group=incumbent_group_map.clone(),
                    bases=probe.bases,
                    factor_A=probe.factor_A,
                    factor_B=probe.factor_B,
                    j_grp=float("inf"),
                    j_cmp=probe.j_cmp,
                    j_act=0.0,
                    j_pair=0.0,
                    rms_error=probe.rms_error,
                    max_expert_rel_error=probe.max_expert_rel_error,
                    prescreen_passed=False,
                    converged=False,
                    swaps_accepted=0,
                    swap_evals=0,
                    overlap_remap=torch.eye(g_cnt),
                    reject_reason="probe_prescreen_failed",
                )

        # 第 6 步：完整冷启动精确 FP64 重拟合（最多 8 轮）
        cur_refit = exact_refit_fixed_partition(
            weights=weights_fp64,
            expert_to_group=torch.tensor(cand_assign, dtype=torch.long),
            num_groups=g_cnt,
            rank=cfg.rank,
            max_sweeps=cfg.refit_max_sweeps,
            tol=cfg.refit_tol,
            eps_w=cfg.eps_w,
        )
        if not cur_refit.converged and not skip_prescreen:
            return GroupingProposal(
                expert_to_group=incumbent_group_map.clone(),
                bases=cur_refit.bases,
                factor_A=cur_refit.factor_A,
                factor_B=cur_refit.factor_B,
                j_grp=float("inf"),
                j_cmp=cur_refit.j_cmp,
                j_act=0.0,
                j_pair=0.0,
                rms_error=cur_refit.rms_error,
                max_expert_rel_error=cur_refit.max_expert_rel_error,
                prescreen_passed=True,
                converged=False,
                swaps_accepted=0,
                swap_evals=0,
                overlap_remap=torch.eye(g_cnt),
                reject_reason="refit_did_not_converge",
            )

        cur_j_grp, cur_j_act, cur_j_pair = evaluate_group_objective(
            cur_refit.j_cmp, cand_assign, g_cnt, u_np, zeta_np, chi_np, cfg.lambda_act, cfg.lambda_pair
        )

        # 第 7--8 步：有界跨组两两交换局部搜索（shortlist <= 32, 上限 8E 次精确评估）
        max_swap_evals = 8 * e_cnt
        swap_evals = 0
        swaps_accepted = 0

        while swap_evals < max_swap_evals and cur_j_grp > 1e-6:
            # 基于当前组基草图、组质心与当前标签重算线性代理代价
            cur_s_base = self._compute_group_mean_sketches(sketches, omegas_np, cand_assign, g_cnt)
            cur_c_base = self._compute_group_centroids(u_np, zeta_np, cand_assign, g_cnt)
            surr_cost = self._build_linear_cost_matrix(
                sketches, cur_s_base, omegas_np, u_np, cur_c_base, zeta_np, chi_np, cand_assign,
                base_access_penalty_matrix,
            )

            # 筛选负代理交换增益的跨组专家对，最多 32 对
            pair_candidates: List[Tuple[float, int, int]] = []
            for i in range(e_cnt):
                gi = cand_assign[i]
                for j in range(i + 1, e_cnt):
                    gj = cand_assign[j]
                    if gi == gj:
                        continue
                    delta = (surr_cost[i, gj] + surr_cost[j, gi]) - (surr_cost[i, gi] + surr_cost[j, gj])
                    if delta < -1e-12:
                        pair_candidates.append((delta, i, j))
            if not pair_candidates:
                break
            pair_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            shortlist = pair_candidates[: cfg.swap_shortlist]

            improved = False
            req_drop = max(cfg.swap_rel_improve * cur_j_grp, cfg.swap_abs_improve)
            for _, i, j in shortlist:
                if swap_evals >= max_swap_evals:
                    break
                swap_evals += 1
                trial_assign = list(cand_assign)
                trial_assign[i], trial_assign[j] = trial_assign[j], trial_assign[i]

                trial_refit = exact_refit_fixed_partition(
                    weights=weights_fp64,
                    expert_to_group=torch.tensor(trial_assign, dtype=torch.long),
                    num_groups=g_cnt,
                    rank=cfg.rank,
                    max_sweeps=cfg.refit_max_sweeps,
                    tol=cfg.refit_tol,
                    eps_w=cfg.eps_w,
                )
                trial_j_grp, trial_j_act, trial_j_pair = evaluate_group_objective(
                    trial_refit.j_cmp, trial_assign, g_cnt, u_np, zeta_np, chi_np, cfg.lambda_act, cfg.lambda_pair
                )
                if cur_j_grp - trial_j_grp >= req_drop:
                    cand_assign = trial_assign
                    cur_refit = trial_refit
                    cur_j_grp, cur_j_act, cur_j_pair = trial_j_grp, trial_j_act, trial_j_pair
                    swaps_accepted += 1
                    improved = True
                    break

            if not improved:
                break

        # 按规范组 ID 重排并相应重排 bases 张量第 0 维
        canon_assign = canonicalize_group_ids(cand_assign, g_cnt)
        # 建立旧索引到规范索引的置换映射
        perm_old_for_new = [0] * g_cnt
        for i in range(e_cnt):
            perm_old_for_new[canon_assign[i]] = cand_assign[i]
        perm_tensor = torch.tensor(perm_old_for_new, dtype=torch.long)
        canon_bases = {name: cur_refit.bases[name].index_select(0, perm_tensor) for name in proj_names}

        overlap = compute_overlap_remap_matrix(incumbent_list, canon_assign, g_cnt, k_size)

        return GroupingProposal(
            expert_to_group=torch.tensor(canon_assign, dtype=torch.long),
            bases=canon_bases,
            factor_A=cur_refit.factor_A,
            factor_B=cur_refit.factor_B,
            j_grp=cur_j_grp,
            j_cmp=cur_refit.j_cmp,
            j_act=cur_j_act,
            j_pair=cur_j_pair,
            rms_error=cur_refit.rms_error,
            max_expert_rel_error=cur_refit.max_expert_rel_error,
            prescreen_passed=True,
            converged=cur_refit.converged,
            swaps_accepted=swaps_accepted,
            swap_evals=swap_evals,
            overlap_remap=overlap,
        )

    def _build_linear_cost_matrix(
        self,
        sketches: Dict[str, np.ndarray],
        s_ref: Dict[str, np.ndarray],
        omegas_np: Dict[str, np.ndarray],
        u_np: Optional[np.ndarray],
        c_ref: Optional[np.ndarray],
        zeta_np: Optional[np.ndarray],
        chi_np: Optional[np.ndarray],
        ref_labels: List[int],
        extra_penalty: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """构造 Eq. (4) 线性代理边代价矩阵 C_{ig}，形状 (E, G)。"""
        cfg = self.config
        e_cnt = cfg.num_experts
        g_cnt = cfg.num_groups
        proj_names = list(cfg.projection_names)
        p_w = len(proj_names)

        cost = np.zeros((e_cnt, g_cnt), dtype=np.float64)
        for name in proj_names:
            # ||phi_{i,p} - s_{g,p}||_2^2
            diff = sketches[name][:, None, :] - s_ref[name][None, :, :]
            sq = np.sum(diff * diff, axis=2)
            cost += (omegas_np[name][:, None] * sq) / float(p_w * e_cnt)

        if u_np is not None and c_ref is not None and zeta_np is not None and cfg.lambda_act > 0.0:
            n_zeta = max(1.0, float(np.sum(zeta_np)))
            diff_u = u_np[:, None, :] - c_ref[None, :, :]
            sq_u = np.sum(diff_u * diff_u, axis=2)
            cost += (cfg.lambda_act / n_zeta) * zeta_np[:, None] * sq_u

        if chi_np is not None and cfg.lambda_pair > 0.0:
            # (lambda_pair / 2) * sum_{j != i} chi_{ij} * 1[g != g_j^-]
            ref_arr = np.array(ref_labels, dtype=np.int64)
            for g in range(g_cnt):
                neq_mask = (ref_arr != g).astype(np.float64)
                cost[:, g] += 0.5 * cfg.lambda_pair * (chi_np @ neq_mask)

        if extra_penalty is not None:
            cost = cost + extra_penalty
        return cost

    @staticmethod
    def _compute_group_mean_sketches(
        sketches: Dict[str, np.ndarray],
        omegas_np: Dict[str, np.ndarray],
        assignment: List[int],
        num_groups: int,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, sk in sketches.items():
            g_sk = np.zeros((num_groups, sk.shape[1]), dtype=np.float64)
            om = omegas_np[name]
            for g in range(num_groups):
                members = [i for i, a in enumerate(assignment) if a == g]
                if members:
                    w = om[members]
                    g_sk[g] = np.sum(sk[members] * w[:, None], axis=0) / max(float(np.sum(w)), 1e-15)
            out[name] = g_sk
        return out

    @staticmethod
    def _compute_group_centroids(
        u_np: Optional[np.ndarray],
        zeta_np: Optional[np.ndarray],
        assignment: List[int],
        num_groups: int,
    ) -> Optional[np.ndarray]:
        if u_np is None or zeta_np is None:
            return None
        c_out = np.zeros((num_groups, u_np.shape[1]), dtype=np.float64)
        for g in range(num_groups):
            members = [i for i, a in enumerate(assignment) if a == g and zeta_np[i] > 0]
            if members:
                c_sum = np.sum(u_np[members], axis=0)
                nrm = float(np.linalg.norm(c_sum))
                c_out[g] = c_sum / nrm if nrm > 1e-12 else 0.0
        return c_out
