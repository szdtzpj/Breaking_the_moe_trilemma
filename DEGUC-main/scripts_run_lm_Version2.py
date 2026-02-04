import argparse, yaml, torch
from torch.optim import AdamW
from deguc.data.causal_lm_dataset import build_lm_loaders
from deguc.model.causal_lm_with_deguc import CausalLMTransformerWithDEGUC
from deguc.train.task_trainer import TaskTrainer
from deguc.utils.distributed import init_distributed, is_distributed, get_rank, cleanup_distributed
from deguc.utils.schedule import QuantizationSchedule

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    init_distributed(cfg["distributed"].get("backend","nccl") if cfg.get("distributed",{}).get("enable",False) else "nccl")
    dist_enabled = is_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm_cfg = cfg["causal_lm"]
    lm_cfg["vocab_name"] = cfg["model"]["vocab_name"]
    train_loader, val_loader, tokenizer = build_lm_loaders(lm_cfg, distributed=dist_enabled)

    model_cfg = cfg["model"]
    moe_kwargs = model_cfg["moe"]
    model = CausalLMTransformerWithDEGUC(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        moe_kwargs=moe_kwargs,
        device=device
    )

    # Store unwrapped model reference for trainer (before DDP wrapping)
    unwrapped_model = model

    if dist_enabled and device.type == "cuda":
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[torch.cuda.current_device()],
                    find_unused_parameters=False)

    train_cfg = cfg["train"]
    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"],
                      betas=tuple(train_cfg.get("betas",[0.9,0.999])),
                      weight_decay=train_cfg.get("weight_decay",0.0))

    quant_cfg = cfg["quantization"]
    quant_schedule = QuantizationSchedule(
        quantize_at_step=quant_cfg["quantize_at_step"],
        finetune_extra_steps=quant_cfg["finetune_extra_steps"],
        freeze_after_quant=quant_cfg["freeze_after_quant"],
        unfreeze_at_step=quant_cfg.get("unfreeze_at_step", None)
    )

    trainer = TaskTrainer(
        model=unwrapped_model,  # Always pass unwrapped model to trainer
        optimizer=optimizer,
        balance_loss_weight=train_cfg["balance_loss_weight"],
        clustering_interval=cfg["deguc_schedule"]["clustering_interval"],
        offload_interval=cfg["deguc_schedule"]["offload_interval"],
        min_offload_rate=cfg["deguc_schedule"]["min_offload_rate"],
        quant_schedule=quant_schedule,
        amp_enabled=cfg["amp"]["enabled"],
        grad_clip=train_cfg["grad_clip"],
        output_dir=cfg["output_dir"]
    )

    trainer.train_loop(
        task_type="causal_lm",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        total_steps=train_cfg["total_steps"],
        eval_interval=train_cfg["eval_interval"],
        save_interval=train_cfg["save_interval"],
        early_stop_patience=train_cfg["early_stop_patience"],
        ddp_model=model if dist_enabled else None  # Pass DDP wrapper separately
    )
    if get_rank()==0:
        print("Training finished (LM). Best checkpoint saved in", cfg["output_dir"])

    cleanup_distributed()

if __name__ == "__main__":
    main()