import argparse, yaml, torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from deguc.model.transformer_with_deguc import MiniTransformerWithDEGUC
from deguc.data.classification_dataset import build_classification_loaders
from deguc.train.task_trainer import TaskTrainer
from deguc.utils.distributed import init_distributed, get_rank, cleanup_distributed, is_distributed
from deguc.utils.schedule import QuantizationSchedule
import os, random

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    init_distributed(cfg.get("distributed", {}).get("backend", "nccl")
                     if cfg.get("distributed", {}).get("enable", False) else "nccl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist_enabled = is_distributed()

    # ---------------- Data ----------------
    loaders_cfg = cfg["classification"]
    loaders_cfg["vocab_name"] = cfg["model"]["vocab_name"]
    train_loader, val_loader, test_loader, tokenizer = build_classification_loaders(
        loaders_cfg, distributed=dist_enabled
    )

    # ---------------- Model ----------------
    model_cfg = cfg["model"]
    moe_kwargs = dict(model_cfg.get("moe", {}))   # Prevent KeyError
    # Safely handle param_dtype - let DEGUCModel handle normalization
    if "param_dtype" not in moe_kwargs:
        moe_kwargs["param_dtype"] = "float32"

    model = MiniTransformerWithDEGUC(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        num_classes=model_cfg["num_classes"],
        seq_len=loaders_cfg["max_seq_len"],
        device=device,
        moe_kwargs=moe_kwargs
    )
    
    # Store unwrapped model reference for trainer (before DDP wrapping)
    unwrapped_model = model

    if dist_enabled and device.type == "cuda":
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[torch.cuda.current_device()],
                    find_unused_parameters=False)

    # ---------------- Optimizer ----------------
    train_cfg = cfg["train"]
    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"],
                      betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
                      weight_decay=train_cfg.get("weight_decay", 0.0))

    # (Scheduler not added here; can be added later if needed)
    # ---------------- Quant Schedule ----------------
    quant_cfg = cfg.get("quantization", {})
    quant_schedule = QuantizationSchedule(
        quantize_at_step=quant_cfg.get("quantize_at_step", 10**9),
        finetune_extra_steps=quant_cfg.get("finetune_extra_steps", 0),
        freeze_after_quant=quant_cfg.get("freeze_after_quant", False),
        unfreeze_at_step=quant_cfg.get("unfreeze_at_step", None)
    )

    # ---------------- Trainer ----------------
    trainer = TaskTrainer(
        model=unwrapped_model,  # Always pass unwrapped model to trainer
        optimizer=optimizer,
        balance_loss_weight=train_cfg["balance_loss_weight"],
        clustering_interval=cfg["deguc_schedule"]["clustering_interval"],
        offload_interval=cfg["deguc_schedule"]["offload_interval"],
        min_offload_rate=cfg["deguc_schedule"]["min_offload_rate"],
        quant_schedule=quant_schedule,
        amp_enabled=cfg.get("amp", {}).get("enabled", False),
        grad_clip=train_cfg["grad_clip"],
        output_dir=cfg["output_dir"],
        print_every=1   # Adjust print frequency
    )

    # ---------------- Train ----------------
    trainer.train_loop(
        task_type="classification",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        total_steps=train_cfg["total_steps"],
        eval_interval=train_cfg["eval_interval"],
        save_interval=train_cfg["save_interval"],
        early_stop_patience=train_cfg["early_stop_patience"],
        ddp_model=model if dist_enabled else None  # Pass DDP wrapper separately
    )

    if get_rank() == 0:
        print("Training finished (classification). Best checkpoint saved in", cfg["output_dir"])

    cleanup_distributed()

if __name__ == "__main__":
    main()
