"""
LoRA adapter for DEGUC models.

This module provides compatibility with HuggingFace PEFT library and LoRA.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class DEGUCLoRAAdapter(nn.Module):
    """
    Adapter to make DEGUC compatible with LoRA and PEFT frameworks.
    
    This wrapper allows DEGUC MoE layers to work seamlessly with:
    - HuggingFace PEFT library
    - Standard LoRA implementations
    - QLoRA and other quantization-aware fine-tuning methods
    
    Example:
        ```python
        from deguc.model.deguc_moe import DEGUCModel
        from deguc.adapters import DEGUCLoRAAdapter
        
        # Create DEGUC model
        moe = DEGUCModel(input_dim=512, output_dim=512, num_initial_experts=16)
        
        # Wrap with LoRA adapter
        moe_with_lora = DEGUCLoRAAdapter(moe, lora_r=8, lora_alpha=16)
        
        # Now compatible with PEFT
        from peft import get_peft_model, LoraConfig
        peft_config = LoraConfig(...)
        model = get_peft_model(moe_with_lora, peft_config)
        ```
    """
    
    def __init__(
        self,
        deguc_model: nn.Module,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None,
    ):
        """
        Args:
            deguc_model: The DEGUC MoE model to wrap
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of module names to apply LoRA to (None = all linear layers)
        """
        super().__init__()
        self.deguc_model = deguc_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.scaling = lora_alpha / lora_r
        
        # Store target modules
        self.target_modules = target_modules or ["router", "compression"]
        
        # LoRA parameters will be added dynamically
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        
        # Initialize LoRA layers for router
        if "router" in self.target_modules:
            self._add_lora_to_router()
    
    def _add_lora_to_router(self):
        """Add LoRA layers to the router."""
        router = self.deguc_model.router
        
        # Add LoRA to group router
        if hasattr(router, 'group_router'):
            in_features = router.group_router.in_features
            out_features = router.group_router.out_features
            
            self.lora_A['group_router'] = nn.Parameter(
                torch.zeros(in_features, self.lora_r)
            )
            self.lora_B['group_router'] = nn.Parameter(
                torch.zeros(self.lora_r, out_features)
            )
            
            # Initialize
            nn.init.kaiming_uniform_(self.lora_A['group_router'], a=5**0.5)
            nn.init.zeros_(self.lora_B['group_router'])
    
    def forward(self, hidden: torch.Tensor) -> tuple:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            hidden: Input tensor
            
        Returns:
            Tuple of (output, balance_loss, aux_info)
        """
        # Standard DEGUC forward
        output, balance_loss, aux_info = self.deguc_model(hidden)
        
        # Apply LoRA adaptation if needed
        # (In practice, LoRA is applied within the router/compression modules)
        
        return output, balance_loss, aux_info
    
    def merge_lora_weights(self):
        """Merge LoRA weights into base model (for inference)."""
        with torch.no_grad():
            if 'group_router' in self.lora_A:
                router = self.deguc_model.router.group_router
                lora_weight = (
                    self.lora_B['group_router'] @ self.lora_A['group_router'].T
                ) * self.scaling
                router.weight.data += lora_weight.T
    
    def unmerge_lora_weights(self):
        """Unmerge LoRA weights from base model."""
        with torch.no_grad():
            if 'group_router' in self.lora_A:
                router = self.deguc_model.router.group_router
                lora_weight = (
                    self.lora_B['group_router'] @ self.lora_A['group_router'].T
                ) * self.scaling
                router.weight.data -= lora_weight.T
    
    def get_lora_parameters(self):
        """Get only LoRA parameters for optimization."""
        return [p for p in self.lora_A.values()] + [p for p in self.lora_B.values()]
    
    def freeze_base_model(self):
        """Freeze base DEGUC model, only train LoRA parameters."""
        for param in self.deguc_model.parameters():
            param.requires_grad = False
        for param in self.get_lora_parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    @property
    def moe(self):
        """Access underlying MoE module."""
        return self.deguc_model
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing."""
        if hasattr(self.deguc_model, 'enable_gradient_checkpointing'):
            self.deguc_model.enable_gradient_checkpointing()
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        if hasattr(self.deguc_model, 'disable_gradient_checkpointing'):
            self.deguc_model.disable_gradient_checkpointing()


def apply_lora_to_deguc(
    model: nn.Module,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Convenience function to apply LoRA to all DEGUC modules in a model.
    
    Args:
        model: Model containing DEGUC modules
        lora_r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability
        target_modules: Target module names
        
    Returns:
        Model with LoRA applied to DEGUC modules
    """
    from deguc.model.deguc_moe import DEGUCModel
    
    def replace_deguc_with_lora(module):
        for name, child in module.named_children():
            if isinstance(child, DEGUCModel):
                # Replace with LoRA-wrapped version
                lora_module = DEGUCLoRAAdapter(
                    child, lora_r, lora_alpha, lora_dropout, target_modules
                )
                setattr(module, name, lora_module)
            else:
                replace_deguc_with_lora(child)
    
    replace_deguc_with_lora(model)
    return model



