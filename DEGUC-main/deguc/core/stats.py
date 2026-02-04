from dataclasses import dataclass, field
import torch
from typing import Dict, Any

@dataclass
class ExpertRuntimeStats:
    activations: int = 0
    tokens: int = 0
    routed_batches: int = 0
    avg_input_mean: float = 0.0

    def update(self, batch_inputs: torch.Tensor):
        bsz = batch_inputs.shape[0]
        self.activations += bsz
        self.tokens += bsz
        self.routed_batches += 1
        current_mean = batch_inputs.mean().item()
        self.avg_input_mean = 0.9 * self.avg_input_mean + 0.1 * current_mean if self.routed_batches > 1 else current_mean

@dataclass
class GlobalStats:
    expert_stats: Dict[int, ExpertRuntimeStats] = field(default_factory=dict)
    offloaded_experts: int = 0
    reloaded_experts: int = 0
    clustering_round: int = 0
    last_group_map: Dict[int, list] = field(default_factory=dict)
    group_stability_score: float = 0.0

    def ensure_expert(self, exp_id: int):
        if exp_id not in self.expert_stats:
            self.expert_stats[exp_id] = ExpertRuntimeStats()

    def activation_rate(self, exp_id: int):
        if exp_id not in self.expert_stats:
            return 0.0
        data = self.expert_stats[exp_id]
        if data.tokens == 0:
            return 0.0
        return data.activations / max(1, data.tokens)

    def compute_group_stability(self, new_map: Dict[int, list]):
        if not self.last_group_map:
            self.group_stability_score = 1.0
        else:
            keys = set(new_map.keys()) | set(self.last_group_map.keys())
            scores = []
            for k in keys:
                a = set(new_map.get(k, []))
                b = set(self.last_group_map.get(k, []))
                if not a and not b:
                    continue
                inter = len(a & b)
                union = len(a | b) if (a | b) else 1
                scores.append(inter / union)
            self.group_stability_score = sum(scores) / len(scores) if scores else 1.0
        self.last_group_map = {k: list(v) for k, v in new_map.items()}

global_stats = GlobalStats()