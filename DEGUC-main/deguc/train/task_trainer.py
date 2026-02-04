import torch
import torch.nn.functional as F
import os
import math
import json
from datetime import datetime

# Prioritize using the new torch.amp (PyTorch 2.0+), fall back to old interface if it fails
try:
    from torch.amp import autocast, GradScaler
    _NEW_AMP_API = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _NEW_AMP_API = False

from deguc.core.logging import SimpleLogger
from deguc.core.stats import global_stats
from deguc.clustering.online_clustering import OnlineExpertClustering
from deguc.utils.distributed import get_rank
from deguc.utils.schedule import QuantizationSchedule


class TaskTrainer:
    def __init__(self, model, optimizer, lr_scheduler=None,
                 balance_loss_weight=0.02,
                 clustering_interval=2000,
                 offload_interval=2500,
                 min_offload_rate=0.0005,
                 quant_schedule: QuantizationSchedule = None,
                 amp_enabled=True,
                 grad_clip=1.0,
                 logger=None,
                 output_dir="outputs",
                 print_every=100,
                 save_group_map_images=False,
                 max_eval_batches=None,
                 eval_progress_every=50):
        """
        Args:
            model: The unwrapped model (not DDP-wrapped)
            print_every: Print interval for normal training phases; print every step for the first 10 steps.
            save_group_map_images: Save PNGs during cluster events when True (requires matplotlib).
            max_eval_batches: Maximum number of batches to process during validation (None means all batches).
            eval_progress_every: Print progress every N batches during validation.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.balance_loss_weight = balance_loss_weight
        self.clustering_interval = clustering_interval
        self.offload_interval = offload_interval
        self.min_offload_rate = min_offload_rate
        self.quant_schedule = quant_schedule
        self.grad_clip = grad_clip
        self.logger = logger or SimpleLogger()
        self.step = 0
        self.clustering = OnlineExpertClustering()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.group_map_dir = os.path.join(output_dir, "group_maps")
        os.makedirs(self.group_map_dir, exist_ok=True)
        self.quantized = False
        self.frozen = False
        self.print_every = print_every
        self.save_group_map_images = save_group_map_images
        self.max_eval_batches = max_eval_batches
        self.eval_progress_every = eval_progress_every

        self.cuda_available = torch.cuda.is_available()
        self.use_amp = bool(amp_enabled and self.cuda_available)
        self.scaler = GradScaler(device_type='cuda', enabled=self.use_amp) if self.use_amp else None

        if get_rank() == 0:
            print(f"[TaskTrainer Init] file={__file__}")
            print(
                f"[TaskTrainer Init] cuda_available={self.cuda_available} "
                f"config_amp={amp_enabled} use_amp={self.use_amp} "
                f"new_amp_api={_NEW_AMP_API} scaler={'on' if self.scaler else 'off'}",
                flush=True
            )

    # ---------- Utilities ----------
    def _save_group_map_snapshot(self, step, group_map):
        """Save group map JSON, and optionally generate PNG."""
        if get_rank() != 0:
            return
        snap = {
            "step": step,
            "time": datetime.utcnow().isoformat(),
            "group_map": {int(k): list(v) for k, v in group_map.items()}
        }
        json_path = os.path.join(self.group_map_dir, f"step_{step}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        if self.save_group_map_images:
            try:
                import matplotlib.pyplot as plt
                all_experts = sorted(set(e for exps in group_map.values() for e in exps))
                if not all_experts:  # no experts => skip
                    return
                expert_to_col = {e: i for i, e in enumerate(all_experts)}
                rows = len(group_map)
                cols = len(all_experts)
                import numpy as np
                mat = np.full((rows, cols), -1, dtype=int)
                for gi, (g, exps) in enumerate(sorted(group_map.items(), key=lambda x: x[0])):
                    for e in exps:
                        mat[gi, expert_to_col[e]] = g
                plt.figure(figsize=(max(6, cols * 0.3), max(3, rows * 0.4)))
                im = plt.imshow(mat, aspect='auto', interpolation='nearest', cmap='tab20')
                plt.colorbar(im, shrink=0.6, label="Group ID")
                plt.title(f"Group Map at step {step}")
                plt.xlabel("Expert Index (sorted)")
                plt.ylabel("Group Row (sorted by group id)")
                plt.tight_layout()
                png_path = os.path.join(self.group_map_dir, f"step_{step}.png")
                plt.savefig(png_path)
                plt.close()
            except Exception as e:
                print(f"[WARN] Failed to create group map image at step {step}: {e}")

    # ---------- Internal Scheduling ----------
    def _maybe_quantize(self):
        if self.quant_schedule is None:
            return
        if self.quant_schedule.is_quantization_step(self.step) and not self.quantized:
            rep = self.model.moe.apply_quantization(build_int8_cache=True)
            self.quantized = True
            if self.quant_schedule.freeze_after_quant:
                for n, p in self.model.moe.compression.named_parameters():
                    if "group_bases" in n or "_A" in n or "_B" in n:
                        p.requires_grad = False
                self.frozen = True
            if get_rank() == 0:
                self.logger.log(event="quantized", step=self.step, **rep)
                print(f"[Quantize] step={self.step} report={rep}", flush=True)

        if self.quant_schedule.should_unfreeze(self.step) and self.frozen:
            for p in self.model.moe.compression.parameters():
                p.requires_grad = True
            self.frozen = False
            if get_rank() == 0:
                self.logger.log(event="unfreeze", step=self.step)
                print(f"[Unfreeze] step={self.step}", flush=True)

    def _maybe_cluster(self):
        if self.step % self.clustering_interval == 0 and self.step > 0:
            new_map = self.clustering.cluster(self.model.moe.compression)
            self.model.moe.update_groups(new_map)
            if get_rank() == 0:
                self.logger.log(
                    event="cluster",
                    step=self.step,
                    groups=len(new_map),
                    stability=global_stats.group_stability_score,
                    group_map={int(k): list(v) for k, v in new_map.items()}
                )
                print(
                    f"[Cluster] step={self.step} groups={len(new_map)} "
                    f"stability={global_stats.group_stability_score:.4f}",
                    flush=True
                )
                self._save_group_map_snapshot(self.step, new_map)

    def _maybe_offload(self):
        if self.step % self.offload_interval == 0 and self.step > 0:
            self.model.moe.offload_inactive(self.min_offload_rate)
            if get_rank() == 0:
                self.logger.log(
                    event="offload",
                    step=self.step,
                    offloaded=global_stats.offloaded_experts
                )
                print(
                    f"[Offload] step={self.step} offloaded_total={global_stats.offloaded_experts}",
                    flush=True
                )

    # ---------- Task Forward Pass ----------
    def train_step_classification(self, batch, model=None):
        """
        Args:
            batch: Input batch
            model: Model to use (can be DDP-wrapped). If None, uses self.model.
        """
        if model is None:
            model = self.model
        input_ids, attention_mask, labels = batch
        ctx = autocast(device_type='cuda', enabled=self.use_amp)
        with ctx:
            logits, balance_loss = model(input_ids, attention_mask)
            ce = F.cross_entropy(logits, labels)
            pred = logits.argmax(dim=-1)
            batch_acc = (pred == labels).float().mean()
            total_loss = ce + self.balance_loss_weight * balance_loss
        return total_loss, {
            "ce": ce.item(),
            "balance": balance_loss.item(),
            "batch_acc": batch_acc.item()
        }

    def train_step_lm(self, batch, model=None):
        """
        Args:
            batch: Input batch
            model: Model to use (can be DDP-wrapped). If None, uses self.model.
        """
        if model is None:
            model = self.model
        input_ids, attention_mask, labels = batch
        ctx = autocast(device_type='cuda', enabled=self.use_amp)
        with ctx:
            logits, balance_loss = model(input_ids, attention_mask)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_attn = attention_mask[:, 1:].contiguous()
            loss_flat = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none"
            )
            loss_masked = (loss_flat * shift_attn.view(-1)).sum() / shift_attn.sum()
            total_loss = loss_masked + self.balance_loss_weight * balance_loss
            batch_ppl = math.exp(min(loss_masked.item(), 50))  # Prevent overflow
        return total_loss, {
            "lm_loss": loss_masked.item(),
            "balance": balance_loss.item(),
            "batch_ppl": batch_ppl
        }

    # ---------- Backward Propagation ----------
    def backward_update(self, loss):
        if self.scaler:
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

    # ---------- Validation ----------
    def evaluate_classification(self, val_loader, device, model=None):
        """
        Args:
            val_loader: Validation data loader
            device: Device to use
            model: Model to use (can be DDP-wrapped). If None, uses self.model.
        """
        if model is None:
            model = self.model
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        if get_rank() == 0:
            print(f"[Eval] classification start (max_batches={self.max_eval_batches})", flush=True)
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if self.max_eval_batches is not None and bi >= self.max_eval_batches:
                    break
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                logits, balance_loss = model(input_ids, attention_mask)
                ce = F.cross_entropy(logits, labels)
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                loss_sum += ce.item() * labels.size(0)
                if get_rank() == 0 and (bi + 1) % self.eval_progress_every == 0:
                    print(f"[Eval] processed {bi + 1} batches...", flush=True)
        acc = correct / total if total > 0 else 0.0
        if get_rank() == 0:
            print(f"[Eval] done. total_batches={bi + 1 if total > 0 else 0}", flush=True)
        return {"val_loss": loss_sum / total if total > 0 else 0.0, "val_acc": acc}

    def evaluate_lm(self, val_loader, device, model=None):
        """
        Args:
            val_loader: Validation data loader
            device: Device to use
            model: Model to use (can be DDP-wrapped). If None, uses self.model.
        """
        if model is None:
            model = self.model
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        if get_rank() == 0:
            print(f"[Eval] lm start (max_batches={self.max_eval_batches})", flush=True)
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if self.max_eval_batches is not None and bi >= self.max_eval_batches:
                    break
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                logits, balance_loss = model(input_ids, attention_mask)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_mask = attention_mask[:, 1:].contiguous()
                loss_flat = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none"
                )
                masked = loss_flat * shift_mask.view(-1)
                total_loss += masked.sum().item()
                total_tokens += shift_mask.sum().item()
                if get_rank() == 0 and (bi + 1) % self.eval_progress_every == 0:
                    print(f"[Eval] processed {bi + 1} batches...", flush=True)
        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
        if get_rank() == 0:
            print(f"[Eval] done. total_batches={bi + 1 if total_tokens > 0 else 0}", flush=True)
        return {"val_loss": avg_loss, "val_ppl": ppl}

    # ---------- Checkpoint Saving ----------
    def save_checkpoint(self, path):
        if get_rank() != 0:
            return
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step
        }
        torch.save(ckpt, path)

    # ---------- Main Training Loop ----------
    def train_loop(self, task_type, train_loader, val_loader,
                   device, total_steps, eval_interval=500,
                   save_interval=2000, early_stop_patience=5, ddp_model=None):
        """
        Main training loop.
        
        Args:
            ddp_model: Optional DDP-wrapped model for forward pass. If None, uses self.model.
        """
        # Use DDP model for forward pass if provided, otherwise use unwrapped model
        forward_model = ddp_model if ddp_model is not None else self.model
        
        best_metric = None
        bad_epochs = 0
        last_eval_step = 0
        train_iter = iter(train_loader)

        while self.step < total_steps:
            forward_model.train()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            self.step += 1
            batch = [x.to(device) for x in batch]
            self._maybe_quantize()
            self._maybe_cluster()
            self._maybe_offload()

            if task_type == "classification":
                loss, detail = self.train_step_classification(batch, forward_model)
                core_loss = detail["ce"]
                core_name = "ce"
            else:
                loss, detail = self.train_step_lm(batch, forward_model)
                core_loss = detail["lm_loss"]
                core_name = "lm_loss"

            self.backward_update(loss)

            if self.step == 1:
                for n, p in self.model.named_parameters():
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        print("[After step 1] NaN in", n)

            need_print = (
                self.step <= 10 or
                self.step % self.print_every == 0 or
                self.step == total_steps
            )

            if get_rank() == 0 and need_print:
                log_payload = {
                    "event": "train",
                    "step": self.step,
                    "loss": loss.item(),
                    **detail,
                    "groups": len(self.model.moe.router.group_expert_map),
                    "offloaded": global_stats.offloaded_experts,
                    "reloaded": global_stats.reloaded_experts
                }
                self.logger.log(**log_payload)

                extra_metrics_str = ""
                if task_type == "classification":
                    extra_metrics_str = f" acc={detail['batch_acc']:.4f}"
                else:
                    extra_metrics_str = f" ppl={detail['batch_ppl']:.2f}"

                print(
                    f"[Train] step={self.step} "
                    f"total={log_payload['loss']:.4f} "
                    f"{core_name}={core_loss:.4f} "
                    f"balance={log_payload['balance']:.4f} "
                    f"groups={log_payload['groups']} "
                    f"offloaded={log_payload['offloaded']} "
                    f"reloaded={log_payload['reloaded']}"
                    f"{extra_metrics_str}",
                    flush=True
                )

            # Validation
            if self.step - last_eval_step >= eval_interval or self.step == total_steps:
                last_eval_step = self.step
                if task_type == "classification":
                    metrics = self.evaluate_classification(val_loader, device, forward_model)
                    primary = metrics["val_acc"]
                    higher_is_better = True
                else:
                    metrics = self.evaluate_lm(val_loader, device, forward_model)
                    primary = -metrics["val_loss"]   # Convert to maximizing -loss
                    higher_is_better = True

                if get_rank() == 0:
                    self.logger.log(event="validation", step=self.step, **metrics)
                    print(f"[Validation] step={self.step} {metrics}", flush=True)

                if best_metric is None or (higher_is_better and primary > best_metric):
                    best_metric = primary
                    bad_epochs = 0
                    self.save_checkpoint(os.path.join(self.output_dir, "best.pt"))
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_stop_patience:
                        if get_rank() == 0:
                            self.logger.log(event="early_stop", step=self.step)
                            print(f"[EarlyStop] step={self.step}", flush=True)
                        break

            if self.step % save_interval == 0:
                self.save_checkpoint(os.path.join(self.output_dir, f"ckpt_{self.step}.pt"))

        if get_rank() == 0:
            self.logger.dump()
            print("[Training Completed] Logs dumped.", flush=True)
