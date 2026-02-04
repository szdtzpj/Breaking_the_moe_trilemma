import torch
from typing import Dict, Any, List


class Int8EffectiveWeightCache:
    def __init__(self, weight_only: bool = True):
        self.weight_only = weight_only
        self.cache: Dict[int, Dict[str, Any]] = {}
        self.enabled = False
        self.input_dim = None
        self.output_dim = None

    def clear(self):
        self.cache.clear()
        self.enabled = False

    @staticmethod
    def _quantize_per_tensor(t: torch.Tensor):
        min_v = t.min()
        max_v = t.max()
        if (max_v - min_v) < 1e-8:
            scale = torch.tensor(1.0, device=t.device, dtype=torch.float32)
            zp = torch.tensor(0.0, device=t.device, dtype=torch.float32)
            q = torch.zeros_like(t, dtype=torch.uint8)
        else:
            scale = (max_v - min_v) / 255.0
            zp = (-min_v / scale).clamp(0, 255)
            q = ((t / scale) + zp).round().clamp(0, 255).to(torch.uint8)
        return q, scale, zp

    def build_from_module(self, compression_module, expert_group_map: Dict[int, List[int]]):
        self.clear()
        base0 = next(iter(compression_module.group_bases.values()))
        self.input_dim, self.output_dim = base0.shape
        with torch.no_grad():
            for g, exps in expert_group_map.items():
                Wg = compression_module.group_bases[str(g)].data
                for e in exps:
                    A = compression_module.expert_residuals.get(f"{e}_A", None)
                    Bp = compression_module.expert_residuals.get(f"{e}_B", None)
                    if A is None or Bp is None:
                        continue
                    delta = torch.matmul(A.data, Bp.data.t())
                    W_eff = (Wg + delta).to(torch.float32)
                    q, scale, zp = self._quantize_per_tensor(W_eff)
                    self.cache[e] = {"q_weight": q.contiguous(),
                                     "scale": scale,
                                     "zero_point": zp,
                                     "orig_dtype": W_eff.dtype}
        self.enabled = True

    def update_single_expert(self, e: int, compression_module, expert_group_map):
        if not self.enabled:
            return
        expert_group = {}
        for g, exps in expert_group_map.items():
            for eid in exps:
                expert_group[eid] = g
        if e not in expert_group:
            return
        g = expert_group[e]
        Wg = compression_module.group_bases[str(g)].data
        A = compression_module.expert_residuals.get(f"{e}_A", None)
        Bp = compression_module.expert_residuals.get(f"{e}_B", None)
        if A is None or Bp is None:
            return
        with torch.no_grad():
            delta = torch.matmul(A.data, Bp.data.t())
            W_eff = (Wg + delta).to(torch.float32)
            q, scale, zp = self._quantize_per_tensor(W_eff)
            self.cache[e] = {"q_weight": q.contiguous(),
                             "scale": scale,
                             "zero_point": zp,
                             "orig_dtype": W_eff.dtype}

    def effective_weight_dequant(self, e: int):
        rec = self.cache[e]
        q = rec["q_weight"].to(torch.float32)
        return (q - rec["zero_point"]) * rec["scale"]

    def report(self):
        if not self.enabled:
            return {}
        total_fp16_bytes = 0
        total_int8_bytes = 0
        for e, rec in self.cache.items():
            q = rec["q_weight"]
            total_fp16_bytes += q.numel() * 2
            total_int8_bytes += q.numel()
        return {
            "experts_cached": len(self.cache),
            "fp16_MB_equiv": total_fp16_bytes / (1024**2),
            "int8_MB": total_int8_bytes / (1024**2),
            "ratio": total_int8_bytes / max(1, total_fp16_bytes)
        }

class SimpleWeightQuantizer:
    def __init__(self, use_int8_cache=True, weight_only=True, try_full_int8=False):
        self.scales = {}
        self.zero_points = {}
        self.quantized_tensors = {}
        self.enabled = False
        self.use_int8_cache = use_int8_cache
        self.int8_cache = Int8EffectiveWeightCache(weight_only=weight_only)
        self.try_full_int8 = try_full_int8
        self.full_int8_available = False

    def _quantize_tensor(self, name: str, t: torch.Tensor):
        min_v = t.min()
        max_v = t.max()
        if (max_v - min_v) < 1e-8:
            scale = torch.tensor(1.0, device=t.device, dtype=torch.float32)
            zp = torch.tensor(0, device=t.device, dtype=torch.int32)
            q = torch.zeros_like(t, dtype=torch.int8)
        else:
            scale = (max_v - min_v) / 255.0
            zp = (-min_v / scale).round().clamp(0, 255)
            q = ((t / scale) + zp).round().clamp(0, 255).to(torch.uint8)
        self.scales[name] = scale
        self.zero_points[name] = zp
        self.quantized_tensors[name] = q

    def quantize_module(self, module):
        for g_key, base in module.group_bases.items():
            self._quantize_tensor(f"base_{g_key}", base.data)
        for k, p in module.expert_residuals.items():
            self._quantize_tensor(f"res_{k}", p.data)
        self.enabled = True

    def dequant_weight(self, name: str, device):
        q = self.quantized_tensors[name]
        scale = self.scales[name]
        zp = self.zero_points[name]
        return (q.to(torch.float32) - zp) * scale

    def replace_forward_weights(self, module):
        if not self.enabled: return
        with torch.no_grad():
            for g_key in list(module.group_bases.keys()):
                name = f"base_{g_key}"
                if name in self.quantized_tensors:
                    module.group_bases[g_key].data = self.dequant_weight(name, module.group_bases[g_key].device).to(module.group_bases[g_key].dtype)
            for k in list(module.expert_residuals.keys()):
                name = f"res_{k}"
                if name in self.quantized_tensors:
                    module.expert_residuals[k].data = self.dequant_weight(name, module.expert_residuals[k].device).to(module.expert_residuals[k].dtype)

    def build_int8_cache(self, compression_module):
        if not self.use_int8_cache:
            return
        self.int8_cache.build_from_module(compression_module, compression_module.group_expert_map)
        if self.try_full_int8:
            try:
                _ = torch.ops.quantized.linear
                self.full_int8_available = True
            except Exception:
                self.full_int8_available = False

    @property
    def int8_enabled(self):
        return self.use_int8_cache and self.int8_cache.enabled

    def update_single_expert(self, expert_id, compression_module):
        self.int8_cache.update_single_expert(expert_id, compression_module, compression_module.group_expert_map)

    def forward_experts_int8(self, hidden: torch.Tensor, routing, compression_module):
        device = hidden.device
        B = hidden.shape[0]
        out = torch.zeros(B, compression_module.output_dim, device=device, dtype=hidden.dtype)
        if not routing:
            return out
        expert_to_tokens = {}
        token_weights = {}
        for t_idx, lst in routing:
            for e, w in lst:
                if e not in self.int8_cache.cache:
                    self.update_single_expert(e, compression_module)
                if e not in self.int8_cache.cache:
                    continue
                expert_to_tokens.setdefault(e, []).append(t_idx)
                token_weights[(t_idx, e)] = w
        if not expert_to_tokens:
            return out
        for e, token_indices in expert_to_tokens.items():
            rec = self.int8_cache.cache[e]
            q_w = rec["q_weight"]
            scale = rec["scale"]
            zp = rec["zero_point"]
            W = (q_w.to(torch.float32) - zp) * scale
            X = hidden[token_indices].to(torch.float32)
            Y = X @ W
            Y = Y.to(hidden.dtype)
            for idx_local, tok in enumerate(token_indices):
                w = token_weights[(tok, e)]
                out[tok] += Y[idx_local] * w
        return out

    def compression_report(self, module):
        original_bytes = 0
        quant_bytes = 0
        for g_key, base in module.group_bases.items():
            original_bytes += base.numel() * 2
            quant_bytes += base.numel()
        for k, p in module.expert_residuals.items():
            original_bytes += p.numel() * 2
            quant_bytes += p.numel()
        r1 = {
            "original_MB": original_bytes / (1024**2),
            "quant_MB": quant_bytes / (1024**2),
            "ratio": quant_bytes / max(1, original_bytes)
        }
        if self.int8_cache.enabled:
            r2 = self.int8_cache.report()
            r1["int8_cache_ratio"] = r2.get("ratio", None)
            r1["int8_cache_experts"] = r2.get("experts_cached", 0)
        return r1