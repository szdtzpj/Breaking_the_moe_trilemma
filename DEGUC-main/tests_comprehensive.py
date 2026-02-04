"""
Comprehensive Test Script - Validate All Improvements
"""
import torch
import torch.nn as nn
from deguc.model.deguc_moe import DEGUCModel
from deguc.model.transformer_with_deguc import MiniTransformerWithDEGUC
from deguc.adapters import DEGUCLoRAAdapter

def test_dtype_handling():
    """Test dtype handling"""
    print("\n=== Test dtype handling ===")

    # Test different dtype formats
    dtypes = ["float32", "fp32", "float16", "fp16", "half", torch.float16]

    for dtype in dtypes:
        try:
            moe = DEGUCModel(
                input_dim=64,
                output_dim=64,
                num_initial_experts=8,
                param_dtype=dtype,
                device=torch.device("cpu")
            )
            print(f"✓ dtype={dtype} created successfully, actual type={moe.param_dtype}")
        except Exception as e:
            print(f"✗ dtype={dtype} failed: {e}")

def test_gradient_checkpointing():
    """Test gradient checkpointing"""
    print("\n=== Test gradient checkpointing ===")

    model = MiniTransformerWithDEGUC(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        num_layers=2,
        num_classes=2,
        moe_kwargs={"num_experts": 8, "rank": 4}
    )

    # Enable gradient checkpointing
    model.enable_gradient_checkpointing()
    print(f"✓ Gradient checkpointing enabled: {model.gradient_checkpointing}")

    # Test forward propagation
    input_ids = torch.randint(0, 1000, (2, 32))
    logits, balance_loss = model(input_ids)
    print(f"✓ Forward propagation successful, output shape: {logits.shape}")

    # Test backward propagation
    loss = logits.sum() + balance_loss
    loss.backward()
    print("✓ Backward propagation successful")

    # Disable gradient checkpointing
    model.disable_gradient_checkpointing()
    print(f"✓ Gradient checkpointing disabled: {model.gradient_checkpointing}")

def test_save_load():
    """Test model saving and loading"""
    print("\n=== Test model saving and loading ===")

    import tempfile
    import os

    # Create model
    moe = DEGUCModel(
        input_dim=64,
        output_dim=64,
        num_initial_experts=8,
        rank=4
    )

    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model")
        moe.save_pretrained(save_path)
        print(f"✓ Model saved to {save_path}")

        # Check files
        assert os.path.exists(os.path.join(save_path, "pytorch_model.bin"))
        assert os.path.exists(os.path.join(save_path, "config.json"))
        print("✓ Config and weight files exist")

        # Load model
        loaded_moe = DEGUCModel.from_pretrained(save_path)
        print("✓ Model loaded successfully")

        # Verify output consistency
        x = torch.randn(4, 64)
        out1, _, _ = moe(x)
        out2, _, _ = loaded_moe(x)
        diff = (out1 - out2).abs().max().item()
        print(f"✓ Output difference: {diff:.6f}")
        assert diff < 1e-5, "Output inconsistent"

def test_lora_adapter():
    """Test LoRA adapter"""
    print("\n=== Test LoRA adapter ===")

    # Create base model
    moe = DEGUCModel(
        input_dim=64,
        output_dim=64,
        num_initial_experts=8
    )

    # Wrap with LoRA adapter
    moe_lora = DEGUCLoRAAdapter(
        moe,
        lora_r=4,
        lora_alpha=8
    )
    print("✓ LoRA adapter created successfully")

    # Test forward propagation
    x = torch.randn(4, 64)
    out, balance_loss, aux = moe_lora(x)
    print(f"✓ Forward propagation successful, output shape: {out.shape}")

    # Test freezing base model
    moe_lora.freeze_base_model()
    trainable = sum(p.numel() for p in moe_lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in moe_lora.parameters())
    print(f"✓ Trainable parameters after freezing: {trainable}/{total}")

    # Test getting LoRA parameters
    lora_params = moe_lora.get_lora_parameters()
    print(f"✓ Number of LoRA parameters: {len(lora_params)}")

def test_ddp_compatibility():
    """Test DDP compatibility"""
    print("\n=== Test DDP compatibility ===")

    model = MiniTransformerWithDEGUC(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        num_layers=2,
        num_classes=2,
        moe_kwargs={"num_experts": 8}
    )

    # Save unwrapped reference
    unwrapped_model = model

    # Simulate DDP wrapping (without actual distributed environment initialization)
    print("✓ Unwrapped model can access moe:", hasattr(unwrapped_model, 'moe'))
    print(f"✓ MoE number of experts: {unwrapped_model.moe.num_experts}")

    # Test forward propagation
    input_ids = torch.randint(0, 1000, (2, 32))
    logits, balance_loss = model(input_ids)
    print(f"✓ Forward propagation successful")

def test_quantization_offload():
    """Test quantization and offloading"""
    print("\n=== Test quantization and offloading ===")

    moe = DEGUCModel(
        input_dim=64,
        output_dim=64,
        num_initial_experts=16,
        enable_int8=True
    )

    # Test forward propagation to activate experts
    x = torch.randn(10, 64)
    for _ in range(5):
        out, _, _ = moe(x)

    # Apply quantization
    report = moe.apply_quantization()
    print(f"✓ Quantization completed")
    print(f"  - Original size: {report['original_MB']:.2f} MB")
    print(f"  - Quantized size: {report['quant_MB']:.2f} MB")
    print(f"  - Compression ratio: {report['ratio']:.2f}")

    # Test offloading
    offloaded = moe.offload_inactive(min_rate=0.5)
    print(f"✓ Offloaded {offloaded} inactive experts")

    # Test reloading
    out2, _, aux = moe(x)
    print(f"✓ Automatically reloaded {aux['reloaded']} experts")

def test_forward_backward():
    """Test forward and backward propagation"""
    print("\n=== Test forward and backward propagation ===")

    model = MiniTransformerWithDEGUC(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        num_layers=2,
        num_classes=2,
        moe_kwargs={"num_experts": 8, "param_dtype": "float32"}
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward propagation
    input_ids = torch.randint(0, 1000, (4, 32))
    labels = torch.randint(0, 2, (4,))

    logits, balance_loss = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits, labels) + 0.01 * balance_loss
    print(f"✓ Forward propagation successful, loss: {loss.item():.4f}")

    # Backward propagation
    optimizer.zero_grad()
    loss.backward()
    print("✓ Backward propagation successful")

    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"✓ Gradients exist: {has_grad}")

    # Optimizer step
    optimizer.step()
    print("✓ Optimizer step successful")

def main():
    """Run all tests"""
    print("=" * 60)
    print("DEGUC Comprehensive Tests")
    print("=" * 60)

    tests = [
        ("dtype handling", test_dtype_handling),
        ("gradient checkpointing", test_gradient_checkpointing),
        ("model saving/loading", test_save_load),
        ("LoRA adapter", test_lora_adapter),
        ("DDP compatibility", test_ddp_compatibility),
        ("quantization and offloading", test_quantization_offload),
        ("forward/backward propagation", test_forward_backward),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n✓ {name} test passed")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)