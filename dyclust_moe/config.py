"""DyClust-MoE 统一配置与经验准入预算体系。

严格对应最新版论文（ICLR 2027）Section 3、Appendix A、Appendix B 与 Appendix J：
- 支持完整版 Full DyClust-MoE 与精简版 DyClust-MoE-Lite；
- 支持小模型两矩阵 GELU FFN（up/down）与公版大模型三矩阵 SwiGLU FFN（gate/up/down + 可选共享分支）；
- 支持独立 pilot 标定的经验准入门控及其全局误差裕度缩放 alpha in {0.5, 1.0, 2.0}；
- 支持所有受控对比基线与正交重放开关。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class AdmissionThresholds:
    """在线重组的经验质量-开销准入预算（pilot 标定）。

    所有阈值均作为可控的经验预算参数，支持通过 ``scaled(alpha)`` 按论文 Table 38
    统一缩放允许误差裕度（上限类阈值乘以 alpha，下限类阈值 tau 按 1 - alpha * (1 - tau) 缩放）。
    """

    # 探测预筛（2 轮冷启动探针）
    probe_rms_cap: float = 0.06
    probe_tail_cap: float = 0.20

    # 精确 FP64 重拟合失真门控与逐层累计失真账本
    rms_cap: float = 0.030          # 单事件相对重构 RMS 上限 b_{l,e}
    tail_cap: float = 0.120         # 单专家最坏相对重构误差上限 tau_{i,p}
    cum_budget: float = 0.100       # 逐层累计重构失真预算 B_l + b_{l,e}
    output_drift_cap: float = 0.020 # 筛选视图相对 FFN 输出漂移上限 D_out

    # 64 步影子路由预检门控
    selector_kl_cap: float = 0.020
    retained_mass_floor: float = 0.980
    topk_recall_floor: float = 0.990

    # 64 切片冻结路由参数插值桥门控
    bridge_nll_cap: float = 0.020
    bridge_ppl_cap: float = 0.250

    def scaled(self, alpha: float) -> "AdmissionThresholds":
        """按论文 Table 38 规则对所有允许误差裕度施加全局缩放因子 alpha。"""
        if alpha <= 0:
            raise ValueError(f"缩放因子 alpha 必须大于 0，当前为 {alpha}")
        return AdmissionThresholds(
            probe_rms_cap=self.probe_rms_cap * alpha,
            probe_tail_cap=self.probe_tail_cap * alpha,
            rms_cap=self.rms_cap * alpha,
            tail_cap=self.tail_cap * alpha,
            cum_budget=self.cum_budget * alpha,
            output_drift_cap=self.output_drift_cap * alpha,
            selector_kl_cap=self.selector_kl_cap * alpha,
            retained_mass_floor=max(0.0, 1.0 - alpha * (1.0 - self.retained_mass_floor)),
            topk_recall_floor=max(0.0, 1.0 - alpha * (1.0 - self.topk_recall_floor)),
            bridge_nll_cap=self.bridge_nll_cap * alpha,
            bridge_ppl_cap=self.bridge_ppl_cap * alpha,
        )


@dataclass
class DyClustConfig:
    """DyClust-MoE 单层与全模型统一配置。"""

    # 模型与专家几何结构
    d_model: int = 768
    d_ff: int = 2688
    num_experts: int = 128
    group_size: int = 8
    rank: int = 8
    top_k: int = 2
    ffn_type: Literal["gelu_2mat", "swiglu_3mat"] = "gelu_2mat"
    retain_bias: bool = True
    has_shared_expert: bool = False
    shared_expert_d_ff: int = 0

    # 路由与容量修复参数（Section 3.2）
    routing_mode: Literal["grouped", "flat", "device_preferred"] = "grouped"
    norm_topk_prob: bool = True
    selector_bias: bool = True
    support_restriction: bool = True
    selector_mass_threshold: float = 0.98
    max_extra_groups: int = 4
    capacity_factor: float = 1.25
    repair_max_rounds: int = 16
    repair_bucket_limit: int = 8
    teacher_subsample_ratio: float = 1.0 / 64.0

    # 训练损失权重（Eq. 1）
    lambda_lb_expert: float = 0.01
    lambda_lb_group: float = 0.01
    lambda_conf: float = 0.05
    lambda_mass: float = 0.10

    # 分组目标与搜索参数（Section 3.1 & Appendix A.1）
    lambda_act: float = 0.25
    lambda_pair: float = 0.10
    sketch_bins: int = 256
    refit_max_sweeps: int = 8
    refit_tol: float = 1e-5
    swap_shortlist: int = 32
    swap_rel_improve: float = 0.005
    swap_abs_improve: float = 1e-7
    eps_w: float = 1e-12

    # 执行路径与分布式设备拓扑（Section 3.3）
    execution_mode: Literal["shared_base", "materialized"] = "shared_base"
    num_devices: int = 8
    gpus_per_node: int = 4
    coalesce_tokens: bool = True
    placement_time_limit_s: float = 2.0

    # 在线重组、门控与 Lite 模式（Section 3.4）
    mode: Literal["full", "lite"] = "full"
    enable_quality_gates: bool = True
    enable_cum_ledger: bool = True
    enable_output_drift_gate: bool = True
    enable_shadow_preflight: bool = True
    enable_bridge: bool = True
    shadow_steps: int = 64
    bridge_slices: int = 64
    local_lr_ramp_steps: int = 256
    threshold_alpha: float = 1.0
    thresholds: AdmissionThresholds = field(default_factory=AdmissionThresholds)

    # 状态耦合控制（用于复现论文 Table 6 & Appendix J.2）
    state_coupling_mode: Literal[
        "single",
        "decoupled_cyclic",
        "matched_schedule",
        "partial_route_place",
        "penalized_independent",
    ] = "single"
    base_access_penalty: float = 0.15
    owner_access_penalty: float = 0.15

    # 随机性与运行标识
    seed: int = 0
    layer_id: int = 0

    def __post_init__(self) -> None:
        if self.num_experts <= 0 or self.group_size <= 0:
            raise ValueError("num_experts 与 group_size 必须为正整数")
        if self.num_experts % self.group_size != 0:
            raise ValueError(
                f"均衡分组要求 num_experts ({self.num_experts}) 能被 group_size ({self.group_size}) 整除"
            )
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) 必须在 [1, {self.num_experts}] 范围内")
        if self.rank < 0:
            raise ValueError("rank 不能为负")
        if self.shared_expert_d_ff == 0:
            self.shared_expert_d_ff = self.d_ff

        # 若选择 DyClust-MoE-Lite 模式，自动关闭辅助预检、影子门控与插值桥
        if self.mode == "lite":
            self.enable_output_drift_gate = False
            self.enable_shadow_preflight = False
            self.enable_bridge = False

    @property
    def num_groups(self) -> int:
        """分组数 G = E / K。"""
        return self.num_experts // self.group_size

    @property
    def projection_names(self) -> Tuple[str, ...]:
        """当前 FFN 结构包含的专家矩阵投影名称。"""
        if self.ffn_type == "gelu_2mat":
            return ("up", "down")
        return ("gate", "up", "down")

    @property
    def projection_codes(self) -> Tuple[int, ...]:
        """与 Appendix A.1 对齐的投影整数编码（gate=0, up=1, down=2）。"""
        if self.ffn_type == "gelu_2mat":
            return (1, 2)
        return (0, 1, 2)

    @property
    def effective_thresholds(self) -> AdmissionThresholds:
        """返回经 alpha 缩放后的生效准入预算。"""
        if self.threshold_alpha == 1.0:
            return self.thresholds
        return self.thresholds.scaled(self.threshold_alpha)


@dataclass
class TransformerLMConfig:
    """与论文 Appendix B.1 对齐的 Pre-LN RoPE Transformer 语言模型配置。"""

    vocab_size: int = 32000
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 2688
    max_seq_len: int = 2048
    layernorm_eps: float = 1e-5
    moe_layers: Tuple[int, ...] = (1, 2, 4, 5, 7, 8, 10, 11)  # 对应论文 1-based {2,3,5,6,8,9,11,12}
    tie_embeddings: bool = True
    moe_config: DyClustConfig = field(default_factory=DyClustConfig)
