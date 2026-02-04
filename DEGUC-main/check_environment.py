"""
环境和依赖检查脚本 - 确保 DEGUC 可以正常运行
"""
import sys
import warnings

def check_python_version():
    """检查 Python 版本"""
    print("=" * 60)
    print("检查 Python 版本...")
    version = sys.version_info
    print(f"当前 Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 错误: DEGUC 需要 Python 3.8 或更高版本")
        return False
    
    print("✓ Python 版本符合要求")
    return True

def check_pytorch():
    """检查 PyTorch 安装和版本"""
    print("\n" + "=" * 60)
    print("检查 PyTorch...")
    
    try:
        import torch
        print(f"✓ PyTorch 已安装: {torch.__version__}")
        
        # 检查版本
        version = torch.__version__.split('+')[0]  # 移除 +cu118 等后缀
        major, minor = map(int, version.split('.')[:2])
        
        if major < 2:
            print("⚠️  警告: 推荐使用 PyTorch 2.0+，当前版本可能缺少某些功能")
        
        # 检查 CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 版本: {torch.version.cuda}")
        else:
            print("⚠️  CUDA 不可用，将使用 CPU（性能较慢）")
        
        # 检查 bfloat16 支持
        if hasattr(torch, 'bfloat16'):
            print("✓ bfloat16 支持可用")
        else:
            print("⚠️  bfloat16 不支持（需要 PyTorch 1.10+）")
        
        return True
        
    except ImportError:
        print("❌ 错误: PyTorch 未安装")
        print("   请运行: pip install torch>=2.0.0")
        return False

def check_transformers():
    """检查 Transformers 库"""
    print("\n" + "=" * 60)
    print("检查 Transformers...")
    
    try:
        import transformers
        print(f"✓ Transformers 已安装: {transformers.__version__}")
        
        version = transformers.__version__.split('.')
        major, minor = int(version[0]), int(version[1])
        
        if major < 4 or (major == 4 and minor < 30):
            print("⚠️  警告: 推荐使用 Transformers 4.30+")
        
        return True
        
    except ImportError:
        print("❌ 错误: Transformers 未安装")
        print("   请运行: pip install transformers>=4.30.0")
        return False

def check_datasets():
    """检查 Datasets 库"""
    print("\n" + "=" * 60)
    print("检查 Datasets...")
    
    try:
        import datasets
        print(f"✓ Datasets 已安装: {datasets.__version__}")
        return True
        
    except ImportError:
        print("⚠️  警告: Datasets 未安装（训练脚本需要）")
        print("   请运行: pip install datasets>=2.12.0")
        return False

def check_yaml():
    """检查 PyYAML"""
    print("\n" + "=" * 60)
    print("检查 PyYAML...")
    
    try:
        import yaml
        print(f"✓ PyYAML 已安装")
        return True
        
    except ImportError:
        print("❌ 错误: PyYAML 未安装（配置文件需要）")
        print("   请运行: pip install pyyaml>=6.0")
        return False

def check_optional_dependencies():
    """检查可选依赖"""
    print("\n" + "=" * 60)
    print("检查可选依赖...")
    
    optional = {
        "matplotlib": ("可视化", "pip install matplotlib>=3.7.0"),
        "accelerate": ("分布式训练", "pip install accelerate>=0.20.0"),
        "pytest": ("开发测试", "pip install pytest>=7.3.0"),
    }
    
    for package, (purpose, install_cmd) in optional.items():
        try:
            __import__(package)
            print(f"✓ {package} 已安装（用于{purpose}）")
        except ImportError:
            print(f"○ {package} 未安装（可选，用于{purpose}）")
            print(f"  安装命令: {install_cmd}")

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("测试基本功能...")
    
    try:
        import torch
        from deguc.model.deguc_moe import DEGUCModel
        
        # 创建小型模型
        print("  创建 DEGUC 模型...")
        moe = DEGUCModel(
            input_dim=64,
            output_dim=64,
            num_initial_experts=4,
            init_groups=2,
            rank=4,
            device=torch.device("cpu")
        )
        print("  ✓ 模型创建成功")
        
        # 测试前向传播
        print("  测试前向传播...")
        x = torch.randn(2, 64)
        output, balance_loss, aux = moe(x)
        
        assert output.shape == (2, 64), f"输出形状错误: {output.shape}"
        assert not torch.isnan(output).any(), "输出包含 NaN"
        assert not torch.isinf(output).any(), "输出包含 Inf"
        print("  ✓ 前向传播成功")
        
        # 测试反向传播
        print("  测试反向传播...")
        loss = output.sum() + balance_loss
        loss.backward()
        print("  ✓ 反向传播成功")
        
        print("\n✓ 所有基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_cuda_memory():
    """检查 CUDA 内存"""
    print("\n" + "=" * 60)
    print("检查 CUDA 内存...")
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # GB
                print(f"  GPU {i}: {props.name}")
                print(f"    总内存: {total_memory:.2f} GB")
                
                if total_memory < 4:
                    print(f"    ⚠️  警告: 内存较小，建议使用梯度检查点和混合精度")
                elif total_memory < 8:
                    print(f"    ○ 内存适中，建议启用混合精度训练")
                else:
                    print(f"    ✓ 内存充足")
        else:
            print("  ○ 未检测到 CUDA 设备")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  无法检查 CUDA 内存: {e}")
        return True

def print_recommendations():
    """打印使用建议"""
    print("\n" + "=" * 60)
    print("使用建议:")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**3)
            
            if total_memory < 8:
                print("\n💡 内存优化建议（GPU < 8GB）:")
                print("  1. 启用梯度检查点: model.enable_gradient_checkpointing()")
                print("  2. 使用混合精度: amp_enabled=True")
                print("  3. 减小批次大小: batch_size=8 或更小")
                print("  4. 使用 float16: param_dtype='float16'")
                print("  5. 启用专家卸载: offload_interval=500")
            else:
                print("\n💡 性能优化建议:")
                print("  1. 使用混合精度训练: amp_enabled=True")
                print("  2. 使用 float16 或 bfloat16: param_dtype='float16'")
                print("  3. 适当增加批次大小以提高 GPU 利用率")
        else:
            print("\n💡 CPU 训练建议:")
            print("  1. 使用较小的模型配置")
            print("  2. 减少专家数量: num_experts=8")
            print("  3. 减小批次大小: batch_size=4")
            print("  4. 考虑使用 Google Colab 或云 GPU")
        
        print("\n📚 快速开始:")
        print("  1. 阅读文档: docs/QUICKSTART.md")
        print("  2. 运行测试: python tests_comprehensive.py")
        print("  3. 查看示例: docs/EXAMPLES.md")
        
    except Exception:
        pass

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("DEGUC 环境检查工具")
    print("=" * 60)
    
    checks = [
        ("Python 版本", check_python_version),
        ("PyTorch", check_pytorch),
        ("Transformers", check_transformers),
        ("Datasets", check_datasets),
        ("PyYAML", check_yaml),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 检查 {name} 时出错: {e}")
            results.append((name, False))
    
    # 可选依赖
    check_optional_dependencies()
    
    # CUDA 内存
    check_cuda_memory()
    
    # 功能测试
    func_test = test_basic_functionality()
    results.append(("功能测试", func_test))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("检查总结:")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("\n🎉 恭喜！您的环境已准备就绪，可以使用 DEGUC！")
        print_recommendations()
        return 0
    else:
        print("\n⚠️  部分检查未通过，请根据上述提示安装缺失的依赖")
        print("\n快速修复:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    exit(main())



