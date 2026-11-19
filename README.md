# DyClust-MoE: One Grouping State for Expert Sharing, Routing, and Placement

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official Python / PyTorch implementation of **DyClust-MoE** (*ICLR 2027*, upgraded from the early `DEGUC-main` prototype). DyClust-MoE coordinates three tightly coupled MoE systems interfaces—**expert parameter sharing**, **candidate routing support**, and **physical device placement**—through one versioned, layer-local balanced grouping state.

---

## Key Upgrades Over Early `DEGUC-main` Prototype

| Dimension | Early `DEGUC-main` Prototype | Upgraded `dyclust_moe/` Framework (ICLR 2027) |
| :--- | :--- | :--- |
| **Representation & Execution** | Single matrix, materialized `W_base + A@B^T`, Python token loop | True **GELU-2mat (`up/down`) & SwiGLU-3mat (`gate/up/down`) FFN** with per-expert bias & optional shared branch; **grouped shared-base batched execution** without materializing experts (`max diff < 1e-5` vs. materialized) |
| **Online Grouping** | 5-step unbalanced K-means on residual means | **Deterministic SHA-256 CountSketch** + farthest-first prototypes + **balanced min-cost flow** + 2-sweep probe pre-screen + **canonical FP64 truncated SVD / WLS alternating refits** + bounded pair-swap search |
| **Routing & Capacity** | Per-token Python loop, no drop-free guarantee | **Mass-distilled group selector** (`1/64` teacher subsample, Eq. 1), group support restriction, **zero-drop cyclic reserve + bounded exact-$k$ parallel capacity repair** (single-edge + 2-token exchanges), plus `norm_topk_prob=False` $L_1$ identity support |
| **Placement & Dispatch** | Empty `DistributedCommunicatorPlaceholder` | **Active-Set NNLS link calibration** ($R^2\ge0.95$, p95 $\le5\%$ gate), **fixed-charge critical-owner MIP placement** (Eq. 10–12), and **destination-coalesced dispatcher** with exact `F+B wire` accounting |
| **Online Lifecycle & Safety** | Random re-init of bases/routers on regrouping | **Pilot-calibrated empirical admission budgets** ($b_{\ell,e}, \tau_{i,p}, B_\ell, D_{\mathrm{out}}$) with global $\alpha$ scaling, overlap-remapped **64-step shadow preflight**, **64-slice frozen-routing interpolation bridge**, **immutable SHA-256 epoch snapshots + linearizable CAS + crash recovery**, and streamlined **`DyClust-MoE-Lite`** |
| **Controlled Baselines & Replays** | None | `ControlledSwitch`, `Full-Expert+Grouped`, `Device-Preferred` (DeepSeek-V2 style), `Sub-MoE` budget-matched merging, 5 state-coupling modes (`Decoupled-3State`, `Matched-Schedule`, `Partial`, `Penalized-Independent`, `Single`), `Support-on/off` & $2\times2$ orthogonal replay, same-state 64-step counterfactual fork, and `PublicMoEAdapter` (`OLMoE-1B-7B` / `Qwen1.5-MoE-A2.7B`) |

---

## Repository Layout

```text
Breaking_the_moe_trilemma/
├── dyclust_moe/                 # Core DyClust-MoE (ICLR 2027) framework
│   ├── config.py                # DyClustConfig, TransformerLMConfig, AdmissionThresholds (alpha scaling)
│   ├── representation.py        # SharedBaseLowRankFFN (shared-base fused & materialized paths), ScalarLedger
│   ├── grouping.py              # SHA-256 CountSketch, balanced MCF, canonical FP64 SVD refit, pair-swap
│   ├── routing.py               # MassDistilledGroupRouter, cyclic reserve, bounded exact-k repair, Device-Preferred
│   ├── placement.py             # NNLS LinkCostCalibrator, FixedChargePlacementSolver, DestinationCoalescedDispatcher
│   ├── lifecycle.py             # EpochSnapshotStore (CAS/crash recovery), OnlineRegroupingController, counterfactual fork
│   ├── layer.py                 # Unified DyClustMoELayer + 5 state-coupling controllers
│   ├── baselines.py             # ControlledSwitch, FullExpertGrouped, Sub-MoE merging, 2x2 orthogonal replay
│   ├── model.py                 # Pre-LN RoPE DyClustTransformerLM + PublicMoEAdapter (OLMoE / Qwen1.5-MoE)
│   └── trainer.py               # Matrix-only AdamW, DyClustTrainer, WallClockLedger, paired Bonferroni stats
├── configs/                     # Reference YAML configurations (Full E=128 & DyClust-MoE-Lite)
├── scripts/
│   └── verify_paper_mechanisms.py  # End-to-end mechanism & baseline benchmark script
├── tests/
│   └── test_dyclust_moe.py      # Comprehensive unit & invariant test suite
├── DEGUC-main/                  # Legacy prototype (syntax errors fixed for archival compatibility)
├── pyproject.toml
└── README.md
```

---

## Quick Start

### 1. Installation & Verification

```bash
pip install -e ".[dev]"

# Run the full unit and mathematical-invariant test suite
python -m pytest -v tests/test_dyclust_moe.py

# Run the end-to-end mechanism verification report
python scripts/verify_paper_mechanisms.py
```

### 2. Minimal Usage Example (Full vs. `DyClust-MoE-Lite`)

```python
import torch
from dyclust_moe import DyClustConfig, DyClustMoELayer, run_2x2_orthogonal_replay

# Full DyClust-MoE or streamlined DyClust-MoE-Lite (mode="lite")
cfg = DyClustConfig(
    d_model=768,
    d_ff=2688,
    num_experts=128,
    group_size=8,
    rank=8,
    top_k=2,
    mode="full",                  # Switch to "lite" to omit D_out, shadow gates, and bridge
    state_coupling_mode="single", # Or "penalized_independent" / "partial_route_place" / "matched_schedule"
    num_devices=8,
    gpus_per_node=4,
)
layer = DyClustMoELayer(cfg)

x = torch.randn(32, 768)
out = layer(x)
print("Output shape:", out.output.shape)
print("Bases/token:", out.bases_per_token, "| Owners/token:", out.comm_stats.mean_owners_per_token)
print("Storage reduction:", f"{layer.scalar_ledger(out.bases_per_token).storage_reduction_ratio * 100:.2f}%")

# Trigger an atomic online regrouping opportunity
report = layer.step_online_regrouping(shadow_batch=x)
print("Accepted:", report.accepted, "| RMS:", report.rms_error, "| Active epoch:", layer.active_epoch)

# Verify shared-base vs. materialized parity and Support-on/off 2x2 replay
replay = run_2x2_orthogonal_replay(layer, x, num_iters=5)
print("Max numerical diff (shared vs. materialized):", replay.max_abs_diff_on)
```

### 3. Public MoE Transfer Adapter (`OLMoE-1B-7B` / `Qwen1.5-MoE-A2.7B`)

```python
import torch
from dyclust_moe import PublicMoEAdapter

qwen_cfg = PublicMoEAdapter.build_qwen_adapter_config(
    d_model=2048, d_ff=1408, num_experts=60, group_size=6, rank=8, top_k=4
)
# Preserves norm_topk_prob=False, bias-free selector, and uncompressed shared branch
```

---

## License

MIT License (see [LICENSE](LICENSE)).
