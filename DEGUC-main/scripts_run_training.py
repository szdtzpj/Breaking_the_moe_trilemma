import torch
from deguc.model.deguc_moe import DEGUCModel
from deguc.train.trainer import DEGUCTrainer
from deguc.utils.data import synthetic_batch
from deguc.core.logging import SimpleLogger

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DEGUCModel(
        input_dim=256,
        output_dim=256,
        num_initial_experts=24,
        init_groups=6,
        rank=8,
        top_k=2,
        device=device,
        enable_int8=True,          # Enable int8 effective weight cache
        weight_only_int8=True,     # Use weight-only mode first
        try_full_int8=False        # If True and on CPU, can try torch.ops.quantized.linear
    )
    logger = SimpleLogger()
    trainer = DEGUCTrainer(
        model,
        lr=2e-3,
        device=device,
        logger=logger,
        clustering_interval=300,
        quant_interval=400,       # Quantize more frequently to test cache reconstruction
        offload_interval=500,
        balance_loss_weight=0.02
    )
    steps = 1200
    bsz = 64
    for i in range(steps):
        batch = synthetic_batch(bsz, model.input_dim, device)
        trainer.training_step(batch)
        # Automatically enable int8 forward after the first quantization (quant_interval)
        # Already built in model.apply_quantization via build_int8_cache => forward will use it automatically
    trainer.finish()
    print("Training finished. Log saved.")
    if model.quantizer.int8_enabled:
        print("Int8 cache report:", model.quantizer.int8_cache.report())

if __name__ == "__main__":
    main()
