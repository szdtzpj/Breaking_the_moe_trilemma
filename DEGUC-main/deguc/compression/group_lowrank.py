import torch
import torch.nn as nn
from typing import Dict, List

class GroupSharedLowRank(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, group_expert_map: Dict[int, List[int]],
                 rank: int = 16, device=None, dtype=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.group_expert_map = group_expert_map
        self.device = device or torch.device("cpu")

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = torch.float16 if (self.device.type == "cuda") else torch.float32

        self.group_bases = nn.ParameterDict()
        self.expert_residuals = nn.ParameterDict()
        self._init_params()

    def _init_params(self):
        for g, experts in self.group_expert_map.items():
            base = torch.empty(self.input_dim, self.output_dim, dtype=self.dtype, device=self.device)
            nn.init.xavier_uniform_(base)
            self.group_bases[str(g)] = nn.Parameter(base)
            for e in experts:
                A = torch.empty(self.input_dim, self.rank, dtype=self.dtype, device=self.device)
                B = torch.empty(self.output_dim, self.rank, dtype=self.dtype, device=self.device)
                nn.init.xavier_uniform_(A)
                nn.init.xavier_uniform_(B)
                self.expert_residuals[f"{e}_A"] = nn.Parameter(A)
                self.expert_residuals[f"{e}_B"] = nn.Parameter(B)

    def update_group_map(self, new_map):
        old_groups = set(self.group_bases.keys())
        new_groups = set(str(g) for g in new_map.keys())
        for og in old_groups - new_groups:
            del self.group_bases[og]

        for g, experts in new_map.items():
            key = str(g)
            if key not in self.group_bases:
                base = torch.empty(self.input_dim, self.output_dim, dtype=self.dtype, device=self.device)
                nn.init.xavier_uniform_(base)
                self.group_bases[key] = nn.Parameter(base)

        existing_experts = {int(k[:-2]) for k in self.expert_residuals if k.endswith("_A")}
        new_experts = set(e for exps in new_map.values() for e in exps)

        for e in existing_experts - new_experts:
            del self.expert_residuals[f"{e}_A"]
            del self.expert_residuals[f"{e}_B"]

        for e in new_experts - existing_experts:
            A = torch.empty(self.input_dim, self.rank, dtype=self.dtype, device=self.device)
            B = torch.empty(self.output_dim, self.rank, dtype=self.dtype, device=self.device)
            nn.init.xavier_uniform_(A)
            nn.init.xavier_uniform_(B)
            self.expert_residuals[f"{e}_A"] = nn.Parameter(A)
            self.expert_residuals[f"{e}_B"] = nn.Parameter(B)

        self.group_expert_map = new_map

    def forward_experts(self, hidden: torch.Tensor, routing):
        """
        Forward pass through experts with robust error handling.
        
        Args:
            hidden: Input tensor of shape (batch_size, input_dim)
            routing: List of (token_idx, [(expert_id, weight), ...])
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Input validation
        if hidden.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {hidden.shape}")
        if hidden.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {hidden.shape[1]}")
        
        orig_dtype = hidden.dtype
        device = hidden.device
        B = hidden.shape[0]
        
        # Handle empty group_bases
        if not self.group_bases:
            import warnings
            warnings.warn("No group bases found, returning zeros")
            return torch.zeros(B, self.output_dim, device=device, dtype=orig_dtype)
        
        weight_dtype = next(iter(self.group_bases.values())).dtype
        if hidden.dtype != weight_dtype:
            hidden = hidden.to(weight_dtype)

        out = torch.zeros(B, self.output_dim, device=device, dtype=hidden.dtype)

        expert_to_tokens = {}
        token_weights = {}
        for token_idx, exp_list in routing:
            # Validate token index
            if token_idx >= B or token_idx < 0:
                import warnings
                warnings.warn(f"Invalid token index {token_idx} for batch size {B}, skipping")
                continue
            for e, w in exp_list:
                expert_to_tokens.setdefault(e, []).append(token_idx)
                token_weights[(token_idx, e)] = w

        if not expert_to_tokens:
            return out.to(orig_dtype)

        # Build expert to group mapping
        expert_group = {}
        for g, exps in self.group_expert_map.items():
            for e in exps:
                expert_group[e] = g

        for e, token_indices in expert_to_tokens.items():
            # Check if expert exists
            if f"{e}_A" not in self.expert_residuals or f"{e}_B" not in self.expert_residuals:
                import warnings
                warnings.warn(f"Expert {e} not found in residuals, skipping")
                continue
            
            if e not in expert_group:
                import warnings
                warnings.warn(f"Expert {e} not found in group mapping, skipping")
                continue
            
            try:
            A = self.expert_residuals[f"{e}_A"]
            Bp = self.expert_residuals[f"{e}_B"]
            g = expert_group[e]
                
                if str(g) not in self.group_bases:
                    import warnings
                    warnings.warn(f"Group {g} not found in bases, skipping expert {e}")
                    continue
                
            Wg = self.group_bases[str(g)]
                
                # Ensure tensors are on the same device
                if A.device != device:
                    A = A.to(device)
                if Bp.device != device:
                    Bp = Bp.to(device)
                if Wg.device != device:
                    Wg = Wg.to(device)
                
                # Compute effective weight with numerical stability
            delta = torch.matmul(A, Bp.t())
            W_eff = Wg + delta
                
                # Process tokens
            X = hidden[token_indices]
            Y = torch.matmul(X, W_eff)
                
                # Accumulate weighted outputs
            for idx_local, tok in enumerate(token_indices):
                w = token_weights[(tok, e)]
                    # Clamp weight for numerical stability
                    w = max(0.0, min(1.0, w))
                out[tok] += Y[idx_local] * w
                    
            except Exception as ex:
                import warnings
                warnings.warn(f"Error processing expert {e}: {ex}, skipping")
                continue

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out
