import torch
import torch.nn as nn
import torch.nn.functional as F
from deguc.core.stats import global_stats

class HierarchicalRouter(nn.Module):
    def __init__(self, hidden_dim: int, group_expert_map, top_k: int = 2, group_top_g: int = 1, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.group_expert_map = group_expert_map
        self.top_k = top_k
        self.group_top_g = group_top_g
        self.device = device
        self.group_ids = sorted(group_expert_map.keys())
        self.num_groups = len(self.group_ids)
        self.group_router = nn.Linear(hidden_dim, self.num_groups)
        self.intra_group_routers = nn.ModuleDict()
        for g, experts in group_expert_map.items():
            self.intra_group_routers[str(g)] = nn.Linear(hidden_dim, len(experts))
        self.to(device) if device else None

    def update_group_map(self, new_map):
        self.group_expert_map = new_map
        self.group_ids = sorted(new_map.keys())
        self.num_groups = len(self.group_ids)
        old_state = self.group_router.state_dict()
        self.group_router = nn.Linear(self.hidden_dim, self.num_groups).to(self.group_router.weight.device)
        with torch.no_grad():
            k = min(old_state["weight"].shape[0], self.group_router.weight.shape[0])
            self.group_router.weight[:k].copy_(old_state["weight"][:k])
            self.group_router.bias[:k].copy_(old_state["bias"][:k])
        new_intra = nn.ModuleDict()
        for g, exps in new_map.items():
            new_intra[str(g)] = nn.Linear(self.hidden_dim, len(exps))
        self.intra_group_routers = new_intra.to(self.group_router.weight.device)

    def forward(self, hidden: torch.Tensor):
        """
        Forward pass with robust error handling.
        
        Args:
            hidden: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Tuple of (routing_info, balance_loss)
        """
        # Input validation
        if hidden.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {hidden.shape}")
        
        B, H = hidden.shape
        
        if H != self.hidden_dim:
            raise ValueError(f"Hidden dimension mismatch: expected {self.hidden_dim}, got {H}")
        
        # Handle empty groups
        if self.num_groups == 0:
            import warnings
            warnings.warn("No groups available, returning empty routing")
            return [(i, []) for i in range(B)], torch.tensor(0.0, device=hidden.device)
        
        # Group-level routing
        group_logits = self.group_router(hidden)
        
        # Add numerical stability
        group_logits = torch.clamp(group_logits, min=-20, max=20)
        group_probs = F.softmax(group_logits, dim=-1)
        
        # Compute balance loss with numerical stability
        mean_probs = group_probs.mean(dim=0)
        mean_probs = torch.clamp(mean_probs, min=1e-9, max=1.0)
        uniform = torch.full_like(mean_probs, 1.0 / self.num_groups)
        
        # Use log_softmax for numerical stability
        balance_loss = F.kl_div(
            torch.log(mean_probs + 1e-9), 
            uniform, 
            reduction="batchmean"
        )
        
        # Clamp balance loss to prevent explosion
        balance_loss = torch.clamp(balance_loss, min=0.0, max=10.0)

        # Select top groups
        if self.group_top_g == 1:
            top_group_scores, top_group_idx = torch.max(group_probs, dim=-1)
            selected_groups = top_group_idx.unsqueeze(-1)
            selected_group_scores = top_group_scores.unsqueeze(-1)
        else:
            k = min(self.group_top_g, self.num_groups)
            selected_group_scores, selected_groups = torch.topk(group_probs, k=k, dim=-1)

        routing_info = []
        for i in range(B):
            token_hidden = hidden[i:i+1]
            token_routes = []
            
            for j in range(selected_groups.shape[1]):
                try:
                    group_idx = selected_groups[i, j].item()
                    
                    # Validate group index
                    if group_idx < 0 or group_idx >= len(self.group_ids):
                        import warnings
                        warnings.warn(f"Invalid group index {group_idx}, skipping")
                        continue
                    
                    g_id = self.group_ids[group_idx]
                group_score = selected_group_scores[i, j].item()
                    
                    # Clamp group score
                    group_score = max(0.0, min(1.0, group_score))
                    
                    if g_id not in self.group_expert_map:
                        import warnings
                        warnings.warn(f"Group {g_id} not in expert map, skipping")
                        continue
                    
                experts = self.group_expert_map[g_id]
                if not experts:
                    continue
                    
                    if str(g_id) not in self.intra_group_routers:
                        import warnings
                        warnings.warn(f"No router for group {g_id}, skipping")
                        continue
                    
                intra_router = self.intra_group_routers[str(g_id)]
                logits = intra_router(token_hidden)
                    
                    # Add numerical stability
                    logits = torch.clamp(logits, min=-20, max=20)
                probs = F.softmax(logits, dim=-1).squeeze(0)
                    
                k = min(self.top_k, probs.shape[0])
                    if k == 0:
                        continue
                    
                exp_scores, exp_idx = torch.topk(probs, k=k)
                    
                for s, idx_e in zip(exp_scores.tolist(), exp_idx.tolist()):
                        # Validate expert index
                        if idx_e < 0 or idx_e >= len(experts):
                            import warnings
                            warnings.warn(f"Invalid expert index {idx_e} for group {g_id}, skipping")
                            continue
                        
                    expert_id = experts[idx_e]
                        combined_score = group_score * s
                        
                        # Clamp combined score
                        combined_score = max(0.0, min(1.0, combined_score))
                        
                        token_routes.append((expert_id, combined_score))
                        
                except Exception as ex:
                    import warnings
                    warnings.warn(f"Error in routing for token {i}, group {j}: {ex}")
                    continue
            
            # Normalize weights
            if token_routes:
                total = sum(x[1] for x in token_routes)
                if total > 1e-9:
                    token_routes = [(e, w / total) for e, w in token_routes]
                else:
                    # If total is too small, use uniform weights
                    uniform_weight = 1.0 / len(token_routes)
                    token_routes = [(e, uniform_weight) for e, _ in token_routes]
            
            routing_info.append((i, token_routes))
            
            # Update statistics
            for e, _ in token_routes:
                global_stats.ensure_expert(e)
        
        return routing_info, balance_loss