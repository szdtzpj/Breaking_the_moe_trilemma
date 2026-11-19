"""DyClust-MoE 训练器、互斥关键路径耗时账本与配对统计推断工具。

严格对齐论文 Appendix B.1 & Appendix J：
- AdamW 优化器（仅对矩阵参数施加 weight decay，bias 与 LayerNorm 权重衰减为 0）；
- 余弦学习率调度 + 1% 线性预热 + 梯度裁剪 1.0 + 重组提交后局部学习率爬坡；
- 互斥关键路径 wall-clock 账本（对应 Table 39：普通训练步 / 提案预筛与 SVD / 桥接额外开销 / 其他重组发布 / 验证保存）；
- 配对 t 置信区间与单侧 Bonferroni 界计算（零 scipy 依赖，内置精确 t 分位数表）。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dyclust_moe.lifecycle import RegroupEventReport
from dyclust_moe.model import DyClustTransformerLM


# 精确双侧/单侧 t 分布分位数查找表（df -> {alpha_two_sided: t_crit}），避免强依赖 scipy
_T_CRIT_TWO_SIDED_95: Dict[int, float] = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
    6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
    15: 2.1314, 20: 2.0860, 30: 2.0423,
}


def _t_quantile_one_sided(df: int, alpha_one_sided: float) -> float:
    """计算自由度 df、单侧显著性 alpha 的 t 临界值（支持 Bonferroni 多重检验）。"""
    try:
        from scipy.stats import t as scipy_t  # type: ignore

        return float(scipy_t.ppf(1.0 - alpha_one_sided, df))
    except Exception:
        # 常用自由度与 Bonferroni 水平（m=3,4,5 下 alpha=0.05/m）精确表值 + Cornish-Fisher 展开兜底
        exact_table: Dict[Tuple[int, float], float] = {
            (4, 0.0125): 3.4954,    # 5 种子, m=4 Bonferroni
            (4, 0.025): 2.7764,
            (5, 0.0125): 3.1634,    # 6 种子, m=4 Bonferroni
            (5, 0.025): 2.5706,
            (9, 0.016667): 2.5083,  # 10 对, m=3 Bonferroni
            (9, 0.0125): 2.6850,    # 10 对, m=4 Bonferroni
            (9, 0.01): 2.8214,      # 10 对, m=5 Bonferroni
            (9, 0.025): 2.2622,
        }
        for (k_df, k_a), val in exact_table.items():
            if k_df == df and abs(k_a - alpha_one_sided) < 1e-4:
                return val
        if abs(alpha_one_sided - 0.025) < 1e-4 and df in _T_CRIT_TWO_SIDED_95:
            return _T_CRIT_TWO_SIDED_95[df]
        # 标准正态近似 + Cornish-Fisher 一阶/二阶修正
        z = _norm_ppf(1.0 - alpha_one_sided)
        g1 = (z**3 + z) / (4.0 * max(1, df))
        g2 = (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * max(1, df) ** 2)
        return z + g1 + g2


def _norm_ppf(p: float) -> float:
    """Acklam 高精度标准正态分布分位数近似。"""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def paired_t_bonferroni_summary(
    diffs: List[float],
    num_family_tests: int = 4,
    alpha: float = 0.05,
    log_ratio_mode: bool = False,
) -> Dict[str, float]:
    """计算配对样本均值、SD、双侧 95% CI 及单侧 Bonferroni 界（对应论文 Table 40 & 48）。"""
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    if n < 2:
        raise ValueError("配对推断至少需要 2 个独立外层单元（种子/配对）")
    df = n - 1
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(float(n))

    t_95 = _t_quantile_one_sided(df, alpha / 2.0)
    t_bonf = _t_quantile_one_sided(df, alpha / float(num_family_tests))

    ci_lo = mean - t_95 * se
    ci_hi = mean + t_95 * se
    bonf_lcb = mean - t_bonf * se
    bonf_ucb = mean + t_bonf * se

    if log_ratio_mode:
        # 吞吐比率在对数空间计算区间，再还原为百分比增益 exp(x) - 1
        return {
            "estimate": math.exp(mean) - 1.0,
            "ci95_low": math.exp(ci_lo) - 1.0,
            "ci95_high": math.exp(ci_hi) - 1.0,
            "bonferroni_lcb": math.exp(bonf_lcb) - 1.0,
            "bonferroni_ucb": math.exp(bonf_ucb) - 1.0,
            "log_sd": sd,
        }
    return {
        "estimate": mean,
        "sd": sd,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "bonferroni_lcb": bonf_lcb,
        "bonferroni_ucb": bonf_ucb,
    }


@dataclass
class WallClockLedger:
    """互斥关键路径耗时分项账本（精确对应论文 Table 39 五大时间桶）。"""

    ordinary_steps_sec: float = 0.0
    proposal_refit_sec: float = 0.0
    extra_bridge_sec: float = 0.0
    other_regroup_pub_sec: float = 0.0
    init_val_save_sec: float = 0.0
    total_effective_tokens: int = 0
    total_aux_reread_tokens: int = 0

    @property
    def total_sec(self) -> float:
        return (
            self.ordinary_steps_sec
            + self.proposal_refit_sec
            + self.extra_bridge_sec
            + self.other_regroup_pub_sec
            + self.init_val_save_sec
        )

    @property
    def full_run_tokens_per_sec(self) -> float:
        return float(self.total_effective_tokens) / max(self.total_sec, 1e-9)

    def as_dict(self) -> Dict[str, float]:
        return {
            "ordinary_steps_sec": self.ordinary_steps_sec,
            "proposal_refit_sec": self.proposal_refit_sec,
            "extra_bridge_sec": self.extra_bridge_sec,
            "other_regroup_pub_sec": self.other_regroup_pub_sec,
            "init_val_save_sec": self.init_val_save_sec,
            "total_sec": self.total_sec,
            "total_effective_tokens": float(self.total_effective_tokens),
            "total_aux_reread_tokens": float(self.total_aux_reread_tokens),
            "full_run_tokens_per_sec": self.full_run_tokens_per_sec,
        }


def build_matrix_only_adamw(
    model: nn.Module,
    lr: float = 2e-4,
    weight_decay: float = 0.10,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """构造与论文 Appendix B.1 一致的矩阵参数衰减 AdamW（bias 与 1D 归一化参数衰减为 0）。"""
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim >= 2 and "embed" not in name and "ln" not in name:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        eps=eps,
    )


class DyClustTrainer:
    """DyClust-MoE 端到端训练器，集成轮询在线重组、学习率爬坡与互斥耗时账本。"""

    def __init__(
        self,
        model: DyClustTransformerLM,
        total_steps: int,
        peak_lr: float = 2e-4,
        weight_decay: float = 0.10,
        warmup_ratio: float = 0.01,
        grad_clip: float = 1.0,
        regroup_every_n_steps: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.total_steps = max(1, total_steps)
        self.peak_lr = peak_lr
        self.warmup_steps = max(1, int(round(self.total_steps * warmup_ratio)))
        self.grad_clip = grad_clip
        self.regroup_every_n_steps = regroup_every_n_steps

        t_init = time.perf_counter()
        self.optimizer = build_matrix_only_adamw(model, lr=peak_lr, weight_decay=weight_decay)
        self.ledger = WallClockLedger()
        self.ledger.init_val_save_sec += time.perf_counter() - t_init

        self.step_count: int = 0
        self.regroup_reports: List[RegroupEventReport] = []
        self.step_metrics: List[Dict[str, float]] = []

    def _compute_scheduled_lr(self, step: int) -> float:
        """1% 线性预热 + 余弦衰减至 0（Appendix B.1）。"""
        if step <= self.warmup_steps:
            return self.peak_lr * float(step) / float(self.warmup_steps)
        progress = min(1.0, float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps)))
        return 0.5 * self.peak_lr * (1.0 + math.cos(math.pi * progress))

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        shadow_input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """执行单步训练，并在预设游标处触发层轮询在线重组。"""
        self.model.train()
        self.step_count += 1
        input_ids = input_ids.to(self.device)
        labels = input_ids if labels is None else labels.to(self.device)

        # 应用全局调度学习率与重组后局部爬坡倍率
        base_lr = self._compute_scheduled_lr(self.step_count)
        local_mult = 1.0
        for _, m_layer in self.model.get_moe_layers():
            local_mult = min(local_mult, m_layer.controller.get_local_lr_multiplier())
        for pg in self.optimizer.param_groups:
            pg["lr"] = base_lr * local_mult

        # 普通前向/反向与优化器更新计时
        t0 = time.perf_counter()
        self.optimizer.zero_grad()
        out = self.model(input_ids, labels=labels, record_history=True)
        assert out.total_loss is not None and out.task_loss is not None
        out.total_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.ledger.ordinary_steps_sec += time.perf_counter() - t0
        self.ledger.total_effective_tokens += int(input_ids.numel())

        # 在预设游标位置执行轮询在线重组，并将耗时记入互斥关键路径账本
        regroup_accepted = 0
        if self.regroup_every_n_steps > 0 and (self.step_count % self.regroup_every_n_steps == 0):
            s_ids = (shadow_input_ids if shadow_input_ids is not None else input_ids).to(self.device)
            rep = self.model.trigger_round_robin_regrouping(
                shadow_input_ids=s_ids,
                data_cursor=self.ledger.total_effective_tokens,
                rng_counter=self.step_count,
            )
            if rep is not None:
                self.regroup_reports.append(rep)
                self.ledger.proposal_refit_sec += rep.proposal_refit_sec
                self.ledger.extra_bridge_sec += rep.bridge_sec
                self.ledger.other_regroup_pub_sec += rep.other_pub_sec
                self.ledger.total_aux_reread_tokens += rep.auxiliary_reread_tokens
                regroup_accepted = 1 if rep.accepted else 0

        metrics = {
            "step": float(self.step_count),
            "task_loss": float(out.task_loss.item()),
            "aux_loss": float(out.aux_loss.item()),
            "total_loss": float(out.total_loss.item()),
            "ppl": math.exp(min(20.0, float(out.task_loss.item()))),
            "bases_per_token": out.mean_bases_per_token,
            "owners_per_token": out.mean_owners_per_token,
            "fb_wire_bytes": float(out.total_fb_wire_bytes),
            "regroup_accepted": float(regroup_accepted),
        }
        self.step_metrics.append(metrics)
        return metrics

    @torch.no_grad()
    def evaluate_nll(self, batches: Iterable[torch.Tensor]) -> float:
        """在固定验证批上评估平均交叉熵 NLL，并将评估时间计入 init_val_save_sec。"""
        t0 = time.perf_counter()
        self.model.eval()
        losses: List[float] = []
        for batch in batches:
            b = batch.to(self.device)
            out = self.model(b, labels=b, record_history=False)
            assert out.task_loss is not None
            losses.append(float(out.task_loss.item()))
        self.ledger.init_val_save_sec += time.perf_counter() - t0
        return sum(losses) / max(1, len(losses))
