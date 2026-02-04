# Breaking the MoE Trilemma

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目实现了DEGUC（Dynamic Expert Grouping with Unified Compression）框架，旨在打破混合专家模型（MoE）的三难困境，实现高效的模型训练和推理。

## 🌟 核心特性

- **动态专家分组**：基于激活的在线聚类算法
- **统一压缩**：共享基础 + 低秩残差的参数共享机制
- **分层量化**：可选的INT8后训练量化
- **专家卸载**：基于阈值的非活跃专家自动卸载
- **两级路由**：组级 + 组内top-k路由，带负载均衡
- **生产就绪**：完整的训练循环、错误处理和监控

## 📁 项目结构

```
Breaking_the_moe_trilemma/
└── DEGUC-main/              # DEGUC核心实现
    ├── deguc/               # 核心包
    │   ├── adapters/        # LoRA等适配器
    │   ├── clustering/      # 在线聚类算法
    │   ├── compression/     # 组低秩压缩
    │   ├── model/           # 模型定义
    │   ├── routing/         # 路由机制
    │   └── train/           # 训练工具
    ├── data/                # 数据目录
    ├── configs_*.yaml       # 配置文件
    └── scripts_*.py         # 训练和分析脚本
```

## 🚀 快速开始

### 安装依赖

```bash
cd DEGUC-main
pip install -r requirements.txt

# 验证环境（推荐）
python check_environment.py
```

### 基础使用

```python
import torch
from deguc.model.deguc_moe import DEGUCModel

# 创建MoE层
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

# 前向传播
x = torch.randn(32, 512, device="cuda")
output, balance_loss, aux_info = moe(x)
```

### 训练分类模型

```bash
# 验证配置
python validate_config.py configs_training_config_8g_classification_Version2.yaml

# 开始训练
python scripts_run_classification_Version2.py \
    --config configs_training_config_8g_classification_Version2.yaml
```

### 训练语言模型

```bash
# 验证配置
python validate_config.py configs_training_config_8g_lm_Version2.yaml

# 开始训练
python scripts_run_lm_Version2.py \
    --config configs_training_config_8g_lm_Version2.yaml
```

## 📚 文档

- **[快速入门指南](DEGUC-main/GETTING_STARTED.md)** - 新用户完整教程
- **[英文文档](DEGUC-main/README_EN.md)** - English documentation

## 🔧 高级特性

### LoRA集成

```python
from deguc.adapters import DEGUCLoRAAdapter

# 使用LoRA适配器包装MoE
moe_with_lora = DEGUCLoRAAdapter(
    moe,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

# 冻结基础模型，仅训练LoRA参数
moe_with_lora.freeze_base_model()
```

### 梯度检查点（节省内存）

```python
# 启用梯度检查点可节省30-50%内存
model.enable_gradient_checkpointing()
```

### 安全训练与自动恢复

```python
from deguc.utils.safety import create_safe_training_environment

# 创建安全训练环境
safety, checkpoint_mgr = create_safe_training_environment("outputs")

# 从最新检查点自动恢复
latest_ckpt = checkpoint_mgr.find_latest_checkpoint()
if latest_ckpt:
    checkpoint_mgr.load_checkpoint(latest_ckpt, model, optimizer)
```

## 📊 监控和可视化

### 分析训练日志

```bash
python scripts_analyze_logs.py \
    --log logs/training_log.jsonl \
    --out plots \
    --ma 20
```

### 可视化分组演化

```bash
python scripts_visualize_group_evolution.py \
    --log logs/training_log.jsonl \
    --out plots/group_evolution.png
```

## 🧪 测试

```bash
# 运行所有测试
python tests_comprehensive.py

# 检查环境
python check_environment.py

# 验证配置
python validate_config.py your_config.yaml
```

## ⚡ 性能优化

### 内存优化

```python
# 1. 启用梯度检查点
model.enable_gradient_checkpointing()

# 2. 使用混合精度训练
trainer = TaskTrainer(..., amp_enabled=True)

# 3. 启用专家卸载
deguc_schedule:
  offload_interval: 1000
  min_offload_rate: 0.001

# 4. 使用float16
moe_kwargs:
  param_dtype: "float16"
```

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支
3. 进行更改并添加测试
4. 运行 `tests_comprehensive.py` 确保所有测试通过
5. 更新相关文档
6. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

感谢所有为本项目做出贡献的开发者和研究人员。

---

**注意**：本项目处于活跃开发中，API可能会发生变化。建议在生产环境使用前进行充分测试。

