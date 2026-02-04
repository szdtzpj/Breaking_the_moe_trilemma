import torch
from deguc.model.deguc_moe import DEGUCModel

def test_forward():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        param_dtype = torch.float16
    else:
        device = torch.device("cpu")
        param_dtype = torch.float32
    model = DEGUCModel(input_dim=64, output_dim=64, num_initial_experts=8,
                       init_groups=2, rank=4, top_k=2, device=device, param_dtype=param_dtype)
    x = torch.randn(10, 64, device=device, dtype=torch.float32)
    y, bal, aux = model(x)
    assert y.shape == (10,64)
    assert bal.requires_grad
    print("Smoke test passed. y dtype =", y.dtype, "balance_loss dtype =", bal.dtype)

if __name__ == "__main__":
    test_forward()