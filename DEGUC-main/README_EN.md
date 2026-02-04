# DEGUC: Dynamic Expert Grouping with Unified Compression

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Production Ready](https://img.shields.io/badge/production-ready-brightgreen.svg)]()

A production-ready implementation of DEGUC (Dynamic Expert Grouping with Unified Compression) for efficient Mixture-of-Experts (MoE) models. This framework features dynamic expert clustering, shared bases with low-rank residuals, hierarchical quantization, inactive expert offloading, and lightweight two-level routing.

## ✨ Key Features

### Core Capabilities
1. **Two-Level Routing** - Group-level + intra-group top-k routing with load balancing
2. **Dynamic Online Clustering** - Activation-based filtering + similarity-based grouping with stability tracking
3. **Group Shared Bases + Low-Rank Residuals** - Efficient parameter sharing with batched computation
4. **Hierarchical Quantization** - Optional post-training INT8 quantization with forward pass integration
5. **Inactive Expert Offloading** - Threshold-based pruning with automatic reloading on routing hits
6. **Comprehensive Metrics** - Activation rates, load balancing, compression ratios, clustering adjustments
7. **Production-Ready Training** - Complete training loop with schedulers and error handling
8. **Extensible Distributed Support** - Placeholder interfaces preserving abstractions for future integration

### Framework Compatibility
- ✅ **LoRA/PEFT Integration** - Full compatibility with HuggingFace PEFT library
- ✅ **Distributed Training** - PyTorch DDP and Accelerate support
- ✅ **Gradient Checkpointing** - Memory-efficient training for large models
- ✅ **Mixed Precision** - Automatic mixed precision (AMP) support
- ✅ **HuggingFace Ecosystem** - Standard save/load interfaces

### Production Features
- ✅ **Robust Error Handling** - Comprehensive input validation and exception recovery
- ✅ **Numerical Stability** - NaN/Inf detection and prevention
- ✅ **Environment Validation** - Automated dependency and configuration checking
- ✅ **Safety Wrappers** - Automatic checkpointing and error logging
- ✅ **Comprehensive Testing** - Full test suite with edge case coverage

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd DEGUC-main

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .

# Verify installation (recommended)
python check_environment.py
```

### Basic Usage

```python
import torch
from deguc.model.deguc_moe import DEGUCModel

# Create MoE layer
moe = DEGUCModel(
    input_dim=512,
    output_dim=512,
    num_initial_experts=16,
    init_groups=4,
    rank=8,
    top_k=2,
    device=torch.device("cuda"),
    param_dtype="float16"
)

# Forward pass
x = torch.randn(32, 512, device="cuda")
output, balance_loss, aux_info = moe(x)
```

### Training a Classification Model

```bash
# Validate configuration (recommended)
python validate_config.py configs_training_config_8g_classification_Version2.yaml

# Start training
python scripts_run_classification_Version2.py \
    --config configs_training_config_8g_classification_Version2.yaml
```

### Training a Language Model

```bash
# Validate configuration (recommended)
python validate_config.py configs_training_config_8g_lm_Version2.yaml

# Start training
python scripts_run_lm_Version2.py \
    --config configs_training_config_8g_lm_Version2.yaml
```

## 📚 Documentation

- **[Getting Started Guide](GETTING_STARTED.md)** - Comprehensive tutorial for new users
- **[API Reference](docs/API.md)** - Detailed API documentation
- **[Examples](docs/EXAMPLES.md)** - Complete usage examples and integration guides
- **[Production Deployment](docs/PRODUCTION.md)** - Best practices for production environments
- **[Configuration Templates](configs_training_config_8g_classification_Version2.yaml)** - Training configuration examples

## 🔧 Advanced Features

### LoRA Integration

```python
from deguc.adapters import DEGUCLoRAAdapter

# Wrap MoE with LoRA adapter
moe_with_lora = DEGUCLoRAAdapter(
    moe,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

# Freeze base model, train only LoRA parameters
moe_with_lora.freeze_base_model()
```

### Gradient Checkpointing (Memory Efficient)

```python
# Enable gradient checkpointing to save 30-50% memory
model.enable_gradient_checkpointing()
```

### Safe Training with Auto-Recovery

```python
from deguc.utils.safety import create_safe_training_environment

# Create safe training environment
safety, checkpoint_mgr = create_safe_training_environment("outputs")

# Auto-resume from latest checkpoint
latest_ckpt = checkpoint_mgr.find_latest_checkpoint()
if latest_ckpt:
    checkpoint_mgr.load_checkpoint(latest_ckpt, model, optimizer)

# Safe forward pass with error handling
logits, balance_loss = safety.safe_forward(model, input_ids, attention_mask)
```

### Model Save/Load (HuggingFace Compatible)

```python
# Save model
moe.save_pretrained("./my_model")

# Load model
from deguc.model.deguc_moe import DEGUCModel
moe = DEGUCModel.from_pretrained("./my_model", device="cuda")
```

## 🛡️ Production Deployment

### Pre-Deployment Checklist

```bash
# 1. Environment validation
python check_environment.py

# 2. Configuration validation
python validate_config.py your_config.yaml

# 3. Run comprehensive tests
python tests_comprehensive.py
```

### Safe Training Loop

```python
from deguc.utils.safety import create_safe_training_environment, NaNDetector

safety, checkpoint_mgr = create_safe_training_environment("outputs")

for step in range(total_steps):
    try:
        # Safe forward pass
        logits, balance_loss = safety.safe_forward(model, input_ids, attention_mask)
        
        # Check for NaN
        if not NaNDetector.check_tensor(loss, "loss"):
            continue
        
        # Safe backward pass
        optimizer.zero_grad()
        safety.safe_backward(loss, optimizer, scaler, grad_clip=1.0)
        
        # Periodic checkpointing
        if step % 1000 == 0:
            checkpoint_mgr.save_checkpoint(step, model, optimizer)
    
    except Exception as e:
        safety.log_error("TrainingError", str(e), {"step": step})
```

## ⚡ Performance Optimization

### Memory Optimization

```python
# 1. Enable gradient checkpointing
model.enable_gradient_checkpointing()

# 2. Use mixed precision training
trainer = TaskTrainer(..., amp_enabled=True)

# 3. Enable expert offloading
deguc_schedule:
  offload_interval: 1000
  min_offload_rate: 0.001

# 4. Use float16
moe_kwargs:
  param_dtype: "float16"
```

### Speed Optimization

```python
# 1. Adjust clustering interval
deguc_schedule:
  clustering_interval: 500  # Larger values reduce clustering overhead

# 2. Use torch.compile (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='reduce-overhead')

# 3. Optimize data loading
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
```

## 📊 Project Structure

```
DEGUC-main/
├── deguc/                          # Core package
│   ├── adapters/                   # LoRA and other adapters
│   ├── clustering/                 # Online clustering algorithms
│   ├── compression/                # Group low-rank compression
│   ├── core/                       # Core utilities (logging, stats)
│   ├── data/                       # Data loading utilities
│   ├── distributed/                # Distributed communication
│   ├── model/                      # Model definitions
│   ├── offload/                    # Expert offloading
│   ├── quantization/               # Quantization utilities
│   ├── routing/                    # Routing mechanisms
│   ├── train/                      # Training utilities
│   └── utils/                      # General utilities
├── docs/                           # Documentation
├── data/                           # Data directory
├── logs/                           # Training logs
├── configs_*.yaml                  # Configuration files
├── scripts_*.py                    # Training and analysis scripts
├── check_environment.py            # Environment validation tool
├── validate_config.py              # Configuration validation tool
├── tests_comprehensive.py          # Comprehensive test suite
├── requirements.txt                # Dependencies
├── setup.py                        # Installation script
└── README.md                       # This file
```

## 🧪 Testing

```bash
# Run all tests
python tests_comprehensive.py

# Run specific tests
python -m pytest tests/ -v

# Check environment
python check_environment.py

# Validate configuration
python validate_config.py your_config.yaml
```

## 📈 Monitoring and Visualization

### Analyze Training Logs

```bash
python scripts_analyze_logs.py \
    --log logs/training_log.jsonl \
    --out plots \
    --ma 20
```

### Visualize Group Evolution

```bash
python scripts_visualize_group_evolution.py \
    --log logs/training_log.jsonl \
    --out plots/group_evolution.png
```

### Visualize Single Group Map

```bash
python scripts_visualize_single_groupmap.py \
    --snapshot outputs/group_maps/step_2000.json \
    --out plots/groupmap_step2000.png
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes and add tests
4. Run `tests_comprehensive.py` to ensure all tests pass
5. Update relevant documentation
6. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


