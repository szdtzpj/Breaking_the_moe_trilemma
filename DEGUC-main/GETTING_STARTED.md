# Getting Started with DEGUC

Welcome to DEGUC! This guide will help you get started in 5 minutes.

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU support)

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd DEGUC-main

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Verify Installation

```bash
# Run environment check
python check_environment.py
```

This will verify:
- Python version
- PyTorch installation and CUDA availability
- Required dependencies
- Basic functionality

## 🚀 Quick Start Examples

### Example 1: Basic MoE Layer

```python
import torch
from deguc.model.deguc_moe import DEGUCModel

# Create a MoE layer
moe = DEGUCModel(
    input_dim=512,
    output_dim=512,
    num_initial_experts=16,
    init_groups=4,
    rank=8,
    top_k=2,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    param_dtype="float16"  # Use float16 for efficiency
)

# Forward pass
x = torch.randn(32, 512, device=moe.device)
output, balance_loss, aux_info = moe(x)

print(f"Output shape: {output.shape}")
print(f"Balance loss: {balance_loss.item():.4f}")
print(f"Reloaded experts: {aux_info['reloaded']}")
```

### Example 2: Classification Model

```python
from deguc.model.transformer_with_deguc import MiniTransformerWithDEGUC
import torch.nn as nn

# Create model
model = MiniTransformerWithDEGUC(
    vocab_size=30522,
    d_model=256,
    n_heads=8,
    num_layers=4,
    num_classes=2,
    moe_kwargs={
        "num_experts": 32,
        "rank": 16,
        "top_k": 2,
        "param_dtype": "float16"
    }
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for batch in train_loader:
    input_ids, attention_mask, labels = batch
    
    # Forward pass
    logits, balance_loss = model(input_ids, attention_mask)
    
    # Compute loss (include balance loss)
    loss = criterion(logits, labels) + 0.01 * balance_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Example 3: Using Configuration Files

```bash
# Validate configuration
python validate_config.py configs_training_config_8g_classification_Version2.yaml

# Start training
python scripts_run_classification_Version2.py \
    --config configs_training_config_8g_classification_Version2.yaml
```

## 🎯 Common Use Cases

### Use Case 1: Memory-Constrained Training

When GPU memory is limited, enable all memory optimizations:

```python
from deguc.model.transformer_with_deguc import MiniTransformerWithDEGUC
from deguc.train.task_trainer import TaskTrainer

# Create model
model = MiniTransformerWithDEGUC(...)

# 1. Enable gradient checkpointing
model.enable_gradient_checkpointing()

# 2. Use mixed precision and expert offloading
trainer = TaskTrainer(
    model=model,
    optimizer=optimizer,
    amp_enabled=True,  # Mixed precision
    clustering_interval=500,
    offload_interval=1000,  # Expert offloading
    min_offload_rate=0.001
)

# 3. Use float16 parameters
moe_kwargs = {
    "num_experts": 32,
    "param_dtype": "float16"
}
```

**Expected savings**: 50-70% memory reduction

### Use Case 2: Fine-Tuning with LoRA

Efficient fine-tuning using LoRA:

```python
from deguc.adapters import DEGUCLoRAAdapter

# Create base model
model = MiniTransformerWithDEGUC(...)

# Add LoRA to MoE layer
moe_with_lora = DEGUCLoRAAdapter(
    model.moe,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

# Freeze base model, train only LoRA
moe_with_lora.freeze_base_model()

# Optimize only LoRA parameters
lora_params = moe_with_lora.get_lora_parameters()
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
```

### Use Case 3: Distributed Training

Multi-GPU training with DDP:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed
dist.init_process_group(backend="nccl")

# Create model
model = MiniTransformerWithDEGUC(...)
unwrapped_model = model  # Keep reference for MoE access

# Wrap with DDP
model = DDP(
    model,
    device_ids=[torch.cuda.current_device()],
    find_unused_parameters=False
)

# Training
trainer = TaskTrainer(model=unwrapped_model, ...)
trainer.train_loop(..., ddp_model=model)
```

### Use Case 4: Production Deployment

Safe production deployment with error handling:

```python
from deguc.utils.safety import create_safe_training_environment, NaNDetector

# Create safe environment
safety, checkpoint_mgr = create_safe_training_environment("outputs")

# Try to resume from checkpoint
latest_ckpt = checkpoint_mgr.find_latest_checkpoint()
if latest_ckpt:
    result = checkpoint_mgr.load_checkpoint(latest_ckpt, model, optimizer)
    if result:
        start_step, extra_state = result
        print(f"Resumed from step {start_step}")

# Safe training loop
for step in range(start_step, total_steps):
    try:
        # Safe forward pass
        logits, balance_loss = safety.safe_forward(model, input_ids, attention_mask)
        
        # Check for NaN
        if not NaNDetector.check_tensor(loss, "loss"):
            print(f"Step {step}: Loss is NaN, skipping batch")
            continue
        
        # Safe backward pass
        optimizer.zero_grad()
        safety.safe_backward(loss, optimizer, scaler, grad_clip=1.0)
        
        # Periodic checkpointing
        if step % 1000 == 0:
            checkpoint_mgr.save_checkpoint(
                step, model, optimizer, scheduler,
                extra_state={"best_acc": best_acc}
            )
    
    except Exception as e:
        safety.log_error("TrainingError", str(e), {"step": step})
        # Decide whether to continue or stop
```

## 📝 Configuration Guide

### Basic Configuration

```yaml
# Model configuration
model:
  vocab_name: bert-base-uncased
  d_model: 256
  num_heads: 8
  num_layers: 4
  num_classes: 2
  
  # MoE configuration
  moe:
    num_experts: 32        # Number of experts
    init_groups: 8         # Initial number of groups
    rank: 16               # Low-rank decomposition rank
    top_k: 2               # Experts per token
    param_dtype: float16   # Parameter dtype

# Training configuration
train:
  total_steps: 10000
  lr: 3.0e-4
  grad_clip: 1.0
  balance_loss_weight: 0.02  # Balance loss weight

# DEGUC schedule
deguc_schedule:
  clustering_interval: 500    # Re-clustering interval
  offload_interval: 1000      # Offload check interval
  min_offload_rate: 0.0005    # Offload threshold
```

### Recommended Configurations

**Small Model (< 1B parameters)**
```yaml
moe:
  num_experts: 16
  rank: 8
  top_k: 2
train:
  lr: 3e-4
  balance_loss_weight: 0.02
deguc_schedule:
  clustering_interval: 500
```

**Medium Model (1B-10B parameters)**
```yaml
moe:
  num_experts: 32
  rank: 16
  top_k: 2
  param_dtype: float16
train:
  lr: 1e-4
  balance_loss_weight: 0.01
deguc_schedule:
  clustering_interval: 1000
  offload_interval: 2000
```

**Large Model (> 10B parameters)**
```yaml
moe:
  num_experts: 64
  rank: 16
  top_k: 2
  param_dtype: float16
  enable_int8: true
train:
  lr: 5e-5
  balance_loss_weight: 0.01
deguc_schedule:
  clustering_interval: 2000
  offload_interval: 1000
  min_offload_rate: 0.001
```

## 🔧 Performance Tuning

### Memory Optimization Checklist

- [ ] Enable gradient checkpointing: `model.enable_gradient_checkpointing()`
- [ ] Use mixed precision: `amp_enabled=True`
- [ ] Enable expert offloading: Set `offload_interval`
- [ ] Use float16/bfloat16: `param_dtype="float16"`
- [ ] Reduce batch size if needed

### Speed Optimization Checklist

- [ ] Adjust clustering interval (500-2000 steps)
- [ ] Increase data loader workers: `num_workers=4`
- [ ] Use `torch.compile` (PyTorch 2.0+)
- [ ] Enable `pin_memory=True` in DataLoader
- [ ] Use `persistent_workers=True` in DataLoader

### Stability Optimization Checklist

- [ ] Adjust balance_loss_weight (0.01-0.05)
- [ ] Use gradient clipping: `grad_clip=1.0`
- [ ] Use appropriate learning rate (1e-4 to 5e-4 for MoE)
- [ ] Monitor for NaN/Inf values
- [ ] Enable error logging

## 🐛 Troubleshooting

### Problem: Out of Memory

**Solution:**
```python
# 1. Enable gradient checkpointing
model.enable_gradient_checkpointing()

# 2. Reduce batch size
batch_size = 8  # or smaller

# 3. Enable expert offloading
deguc_schedule:
  offload_interval: 500
  min_offload_rate: 0.001

# 4. Use mixed precision
amp_enabled: true
```

### Problem: Training Unstable / Loss is NaN

**Solution:**
```python
# 1. Lower learning rate
lr: 1e-4  # instead of 3e-4

# 2. Use gradient clipping
grad_clip: 1.0

# 3. Adjust balance loss weight
balance_loss_weight: 0.01  # instead of 0.05

# 4. Check data preprocessing
# Ensure inputs are properly normalized
```

### Problem: Cannot Access `moe` Attribute with DDP

**Solution:**
```python
# Save unwrapped model reference
unwrapped_model = model
model = DDP(model)

# Use unwrapped model to access MoE
moe = unwrapped_model.moe
```

### Problem: Slow Training Speed

**Solution:**
```python
# 1. Increase clustering interval
deguc_schedule:
  clustering_interval: 1000  # larger value

# 2. Optimize data loading
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

# 3. Use torch.compile (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model)
```

## 📚 Next Steps

1. **Read the Documentation**
   - [API Documentation](docs/API.md) - Complete API reference
   - [Examples](docs/EXAMPLES.md) - Detailed usage examples
   - [Production Guide](docs/PRODUCTION.md) - Deployment best practices

2. **Run Tests**
   ```bash
   python tests_comprehensive.py
   ```

3. **Try Custom Configurations**
   - Modify `configs_training_config_8g_classification_Version2.yaml`
   - Validate with `python validate_config.py your_config.yaml`

4. **Explore Advanced Features**
   - LoRA integration
   - Quantization
   - Expert offloading
   - Distributed training

## 💡 Tips

1. **Start Small**: Begin with fewer experts (8-16) to test, then scale up
2. **Monitor Metrics**: Watch balance loss and expert activation rates
3. **Save Regularly**: Use `save_interval` for periodic checkpoints
4. **Visualize Groups**: Enable `save_group_map_images: true` to see expert grouping evolution
5. **Use Validation**: Always run `check_environment.py` and `validate_config.py` before training

## 🤝 Getting Help

- **Documentation**: Check the `docs/` directory
- **Tests**: Run `python tests_comprehensive.py`
- **Environment**: Run `python check_environment.py`
- **Issues**: Submit on GitHub Issues

## 🎉 You're Ready!

You now have everything you need to start using DEGUC. Happy training! 🚀

---


