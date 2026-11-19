"""DyClust-MoE: One Grouping State for Expert Sharing, Routing, and Placement.

最新版 ICLR 2027 论文对应的完整 Python / PyTorch 参考实现框架。
"""

from dyclust_moe.baselines import (
    ControlledSwitchMoELayer,
    FullExpertGroupedMoELayer,
    Replay2x2Result,
    build_sub_moe_merged_baseline,
    compute_max_merged_experts_within_budget,
    run_2x2_orthogonal_replay,
)
from dyclust_moe.config import AdmissionThresholds, DyClustConfig, TransformerLMConfig
from dyclust_moe.grouping import (
    DeterministicGrouper,
    GroupingProposal,
    RefitResult,
    canonical_truncated_svd_fp64,
    canonicalize_group_ids,
    compute_countsketch,
    compute_overlap_remap_matrix,
    exact_refit_fixed_partition,
    solve_balanced_min_cost_flow,
)
from dyclust_moe.layer import DyClustMoELayer, MoELayerOutput
from dyclust_moe.lifecycle import (
    EpochPhase,
    EpochSnapshotRecord,
    EpochSnapshotStore,
    HistoryWindowCollector,
    OnlineRegroupingController,
    RegroupEventReport,
    run_same_state_counterfactual_fork,
)
from dyclust_moe.model import DyClustTransformerLM, LMForwardOutput, PublicMoEAdapter
from dyclust_moe.placement import (
    CommunicationStats,
    DestinationCoalescedDispatcher,
    FixedChargePlacementSolver,
    LinkCostCalibrator,
    LinkCostModel,
    PlacementResult,
)
from dyclust_moe.representation import ScalarLedger, SharedBaseLowRankFFN, SharedBaseLowRankProjection
from dyclust_moe.routing import (
    MassDistilledGroupRouter,
    RoutingOutput,
    bounded_exact_k_repair,
    compute_cyclic_reserve,
)
from dyclust_moe.trainer import (
    DyClustTrainer,
    WallClockLedger,
    build_matrix_only_adamw,
    paired_t_bonferroni_summary,
)

__all__ = [
    "AdmissionThresholds",
    "DyClustConfig",
    "TransformerLMConfig",
    "SharedBaseLowRankProjection",
    "SharedBaseLowRankFFN",
    "ScalarLedger",
    "DeterministicGrouper",
    "GroupingProposal",
    "RefitResult",
    "canonical_truncated_svd_fp64",
    "compute_countsketch",
    "exact_refit_fixed_partition",
    "solve_balanced_min_cost_flow",
    "canonicalize_group_ids",
    "compute_overlap_remap_matrix",
    "MassDistilledGroupRouter",
    "RoutingOutput",
    "compute_cyclic_reserve",
    "bounded_exact_k_repair",
    "LinkCostModel",
    "LinkCostCalibrator",
    "FixedChargePlacementSolver",
    "PlacementResult",
    "DestinationCoalescedDispatcher",
    "CommunicationStats",
    "EpochPhase",
    "EpochSnapshotRecord",
    "EpochSnapshotStore",
    "HistoryWindowCollector",
    "OnlineRegroupingController",
    "RegroupEventReport",
    "run_same_state_counterfactual_fork",
    "DyClustMoELayer",
    "MoELayerOutput",
    "DyClustTransformerLM",
    "LMForwardOutput",
    "PublicMoEAdapter",
    "ControlledSwitchMoELayer",
    "FullExpertGroupedMoELayer",
    "build_sub_moe_merged_baseline",
    "compute_max_merged_experts_within_budget",
    "run_2x2_orthogonal_replay",
    "Replay2x2Result",
    "DyClustTrainer",
    "WallClockLedger",
    "build_matrix_only_adamw",
    "paired_t_bonferroni_summary",
]
