import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from deguc.routing.router import HierarchicalRouter
from deguc.compression.group_lowrank import GroupSharedLowRank
from deguc.quantization.quantizer import SimpleWeightQuantizer
from deguc.offload.offload_manager import ExpertOffloadManager
from deguc.core.stats import global_stats
from deguc.distributed.communicator import DistributedCommunicatorPlaceholder

class DEGUCModel(nn.Module):
    """
    DEGUC MoE Layer with LoRA/PEFT compatibility.
    
    This module can be used as a drop-in replacement for standard FFN layers
    and is compatible with LoRA, QLoRA, and other PEFT methods.
    """
    def __init__(self, input_dim=512, output_dim=512, num_initial_experts=16, init_groups=4,
                 rank=16, top_k=2, device=None,
                 enable_int8=False, weight_only_int8=True, try_full_int8=False,
                 param_dtype=None):
        super().__init__()
        
        # Input validation
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"input_dim and output_dim must be positive, got {input_dim}, {output_dim}")
        if num_initial_experts <= 0:
            raise ValueError(f"num_initial_experts must be positive, got {num_initial_experts}")
        if init_groups <= 0 or init_groups > num_initial_experts:
            raise ValueError(f"init_groups must be in (0, {num_initial_experts}], got {init_groups}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        self.device = device or torch.device("cpu")
        
        # Normalize param_dtype to torch.dtype
        if param_dtype is None:
            param_dtype = torch.float32
        elif isinstance(param_dtype, str):
            dtype_map = {
                "float32": torch.float32, "fp32": torch.float32, "32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16, "16": torch.float16, "half": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            }
            param_dtype = dtype_map.get(param_dtype.lower(), torch.float32)
        
        self.param_dtype = param_dtype
        
        experts = list(range(num_initial_experts))
        groups = {g: experts[g::init_groups] for g in range(init_groups)}
        self.compression = GroupSharedLowRank(
            input_dim, output_dim, groups, rank=rank,
            device=self.device, dtype=self.param_dtype
        )
        self.router = HierarchicalRouter(input_dim, groups, top_k=top_k, device=self.device)
        self.quantizer = SimpleWeightQuantizer(
            use_int8_cache=enable_int8,
            weight_only=weight_only_int8,
            try_full_int8=try_full_int8
        )
        self.offloader = ExpertOffloadManager()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.num_experts = num_initial_experts
        self.top_k = top_k
        self.communicator = DistributedCommunicatorPlaceholder()
        
        # For gradient checkpointing compatibility
        self.gradient_checkpointing = False
        
        self.to(self.device)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with automatic expert reloading and quantization support.
        
        Args:
            hidden: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (output, balance_loss, aux_info)
        """
        # Input validation
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(hidden)}")
        
        if hidden.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {hidden.shape}")
        
        if hidden.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {hidden.shape[1]}"
            )
        
        # Check for NaN or Inf
        if torch.isnan(hidden).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(hidden).any():
            raise ValueError("Input contains Inf values")
        
        # Support gradient checkpointing
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, hidden, use_reentrant=False
            )
        return self._forward_impl(hidden)
    
    def _forward_impl(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Internal forward implementation."""
        try:
        routing, balance_loss = self.router(hidden)
        experts_needed = set(e for _, lst in routing for e,_ in lst)
        reloaded = 0
            
            # Reload offloaded experts if needed
        for e in experts_needed:
                try:
            changed = self.offloader.reload_if_needed(
                        e, self.compression, self.input_dim, self.rank, self.output_dim, self.param_dtype
            )
            if changed:
                reloaded += 1
                global_stats.reloaded_experts += 1
                        # Update quantization cache for reloaded expert
                if self.quantizer.int8_enabled:
                    self.quantizer.update_single_expert(e, self.compression)
                except Exception as ex:
                    import warnings
                    warnings.warn(f"Failed to reload expert {e}: {ex}")
            
            # Forward through experts
        if self.quantizer.int8_enabled:
            out = self.quantizer.forward_experts_int8(hidden, routing, self.compression)
        else:
            out = self.compression.forward_experts(hidden, routing)
            
            # Update statistics
        for token_idx, lst in routing:
            for e,_ in lst:
                    try:
                global_stats.expert_stats[e].update(hidden[token_idx:token_idx+1])
                    except Exception:
                        pass  # Silently skip stat update errors
            
            # Check output for NaN/Inf
            if torch.isnan(out).any() or torch.isinf(out).any():
                import warnings
                warnings.warn("Output contains NaN or Inf, replacing with zeros")
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            
        return out, balance_loss, {"reloaded": reloaded}
            
        except Exception as ex:
            import warnings
            warnings.warn(f"Error in forward pass: {ex}, returning zeros")
            # Return safe fallback
            out = torch.zeros(hidden.shape[0], self.output_dim, device=hidden.device, dtype=hidden.dtype)
            balance_loss = torch.tensor(0.0, device=hidden.device)
            return out, balance_loss, {"reloaded": 0, "error": str(ex)}

    def apply_quantization(self, build_int8_cache=True):
        """Apply quantization to the compression module."""
        self.quantizer.quantize_module(self.compression)
        self.quantizer.replace_forward_weights(self.compression)
        if build_int8_cache and self.quantizer.use_int8_cache:
            self.quantizer.build_int8_cache(self.compression)
        return self.quantizer.compression_report(self.compression)

    def update_groups(self, new_map):
        """Update expert grouping and rebuild caches."""
        self.compression.update_group_map(new_map)
        self.router.update_group_map(new_map)
        # Rebuild quantization cache after group update
        if self.quantizer.int8_enabled:
            self.quantizer.build_int8_cache(self.compression)

    def offload_inactive(self, min_rate=0.0005):
        """Offload inactive experts to save memory."""
        from deguc.core.stats import global_stats
        offloaded_count = 0
        for exp_id, stat in list(global_stats.expert_stats.items()):
            rate = global_stats.activation_rate(exp_id)
            if rate < min_rate:
                self.offloader.offload(exp_id, self.compression)
                global_stats.offloaded_experts += 1
                offloaded_count += 1
                # Remove from quantization cache
                if self.quantizer.int8_enabled and exp_id in self.quantizer.int8_cache.cache:
                    del self.quantizer.int8_cache.cache[exp_id]
        return offloaded_count
    
    def get_num_params(self, only_trainable: bool = False) -> int:
        """Get number of parameters (for compatibility with HuggingFace)."""
        params = self.parameters() if not only_trainable else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    def save_pretrained(self, save_directory: str):
        """Save model in HuggingFace-compatible format."""
        import os
        import json
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save config
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_experts": self.num_experts,
            "rank": self.rank,
            "top_k": self.top_k,
            "param_dtype": str(self.param_dtype),
            "model_type": "deguc_moe"
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, device=None):
        """Load model from HuggingFace-compatible format."""
        import os
        import json
        
        # Load config
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            num_initial_experts=config["num_experts"],
            rank=config["rank"],
            top_k=config["top_k"],
            device=device,
            param_dtype=config.get("param_dtype", "float32")
        )
        
        # Load weights
        state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location=device or "cpu")
        model.load_state_dict(state_dict)
        
        return model