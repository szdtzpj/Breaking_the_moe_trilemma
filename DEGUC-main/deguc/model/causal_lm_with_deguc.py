import torch
import torch.nn as nn
import math
from deguc.model.deguc_moe import DEGUCModel

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask=None):
        B,T,H = x.shape
        qkv = self.qkv(x).view(B,T,3,self.n_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        causal = torch.tril(torch.ones(T,T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            m = attention_mask[:,None,None,:]
            att = att.masked_fill(m==0, float("-inf"))
        probs = att.softmax(dim=-1)
        out = (probs @ v).transpose(1,2).contiguous().view(B,T,H)
        return self.proj(out)

class CausalLMBlock(nn.Module):
    def __init__(self, d_model, n_heads, moe: DEGUCModel, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.moe = moe

    def forward(self, x, attention_mask=None):
        h = self.attn(x, attention_mask)
        x = x + self.dropout(h)
        x = self.ln1(x)
        B,T,H = x.shape
        flat = x.view(B*T, H)
        moe_out, balance_loss, _ = self.moe(flat)
        moe_out = moe_out.view(B,T,H)
        x = x + self.dropout(moe_out)
        x = self.ln2(x)
        return x, balance_loss

class CausalLMTransformerWithDEGUC(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, num_layers=6,
                 moe_kwargs=None, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        moe_kwargs = moe_kwargs or {}
        
        # 使用 DEGUCModel 内部的标准化逻辑
        param_dtype = moe_kwargs.get("param_dtype", "float32")
        
        shared_moe = DEGUCModel(
            input_dim=d_model,
            output_dim=d_model,
            num_initial_experts=moe_kwargs.get("num_experts", 32),
            init_groups=moe_kwargs.get("init_groups", 8),
            rank=moe_kwargs.get("rank", 16),
            top_k=moe_kwargs.get("top_k", 2),
            device=self.device,
            enable_int8=moe_kwargs.get("enable_int8", False),
            weight_only_int8=moe_kwargs.get("weight_only_int8", True),
            try_full_int8=moe_kwargs.get("try_full_int8", False),
            param_dtype=param_dtype
        )
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        self.layers = nn.ModuleList([
            CausalLMBlock(d_model, n_heads, shared_moe) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        
        # Store reference to shared MoE for easy access (DDP-compatible)
        self._moe_module = shared_moe
        
        # For gradient checkpointing
        self.gradient_checkpointing = False
        
        self.to(self.device)

    @property
    def moe(self):
        """Access MoE module (works with both wrapped and unwrapped models)."""
        return self._moe_module
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        if hasattr(self._moe_module, 'enable_gradient_checkpointing'):
            self._moe_module.enable_gradient_checkpointing()
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        if hasattr(self._moe_module, 'disable_gradient_checkpointing'):
            self._moe_module.disable_gradient_checkpointing()

    def forward(self, input_ids, attention_mask=None):
        B,T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        balance_losses = []
        
        # Support gradient checkpointing
        if self.gradient_checkpointing and self.training:
            for layer in self.layers:
                x, bl = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, use_reentrant=False
                )
                balance_losses.append(bl)
        else:
        for layer in self.layers:
            x, bl = layer(x, attention_mask)
            balance_losses.append(bl)
        
        x = self.ln(x)
        logits = self.lm_head(x)
        balance_loss = torch.stack(balance_losses).mean()
        return logits, balance_loss