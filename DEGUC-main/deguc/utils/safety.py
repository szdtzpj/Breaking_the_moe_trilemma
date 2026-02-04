"""
生产环境错误处理和恢复机制
"""
import torch
import os
import json
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

class SafetyWrapper:
    """安全包装器 - 捕获和处理训练中的错误"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.error_log_path = os.path.join(output_dir, "error_log.jsonl")
        os.makedirs(output_dir, exist_ok=True)
    
    def log_error(self, error_type: str, error_msg: str, context: Dict[str, Any] = None):
        """记录错误到日志文件"""
        error_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "error_message": error_msg,
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        
        try:
            with open(self.error_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"无法写入错误日志: {e}")
    
    def safe_forward(self, model, *args, **kwargs):
        """安全的前向传播"""
        try:
            return model(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.log_error("OOM", str(e), {"args_shapes": [a.shape if hasattr(a, 'shape') else None for a in args]})
                # 清理 CUDA 缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU 内存不足。建议:\n"
                    "1. 减小批次大小\n"
                    "2. 启用梯度检查点: model.enable_gradient_checkpointing()\n"
                    "3. 使用混合精度训练: amp_enabled=True\n"
                    "4. 启用专家卸载"
                ) from e
            else:
                self.log_error("RuntimeError", str(e))
                raise
        except ValueError as e:
            self.log_error("ValueError", str(e))
            raise
        except Exception as e:
            self.log_error("UnknownError", str(e))
            raise
    
    def safe_backward(self, loss, optimizer, scaler=None, grad_clip=None):
        """安全的反向传播"""
        try:
            if scaler:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in optimizer.param_groups[0]['params'] if p.requires_grad],
                        grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in optimizer.param_groups[0]['params'] if p.requires_grad],
                        grad_clip
                    )
                optimizer.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.log_error("OOM_Backward", str(e))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("反向传播时 GPU 内存不足，请减小批次大小") from e
            else:
                self.log_error("BackwardError", str(e))
                raise

class CheckpointManager:
    """检查点管理器 - 自动保存和恢复"""
    
    def __init__(self, output_dir: str, keep_last_n: int = 3):
        self.output_dir = output_dir
        self.keep_last_n = keep_last_n
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, step: int, model, optimizer, scheduler=None, 
                       extra_state: Dict = None, is_best: bool = False):
        """保存检查点"""
        try:
            checkpoint = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            if scheduler:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            
            if extra_state:
                checkpoint["extra_state"] = extra_state
            
            # 保存常规检查点
            ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.pt")
            torch.save(checkpoint, ckpt_path)
            
            # 保存最佳检查点
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
                torch.save(checkpoint, best_path)
            
            # 保存最新检查点（用于快速恢复）
            latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
            torch.save(checkpoint, latest_path)
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
            return ckpt_path
            
        except Exception as e:
            print(f"保存检查点失败: {e}")
            traceback.print_exc()
            return None
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer, scheduler=None):
        """加载检查点"""
        try:
            if not os.path.exists(checkpoint_path):
                print(f"检查点不存在: {checkpoint_path}")
                return None
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if scheduler and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            step = checkpoint.get("step", 0)
            extra_state = checkpoint.get("extra_state", {})
            
            print(f"成功加载检查点: {checkpoint_path}")
            print(f"  步数: {step}")
            print(f"  时间: {checkpoint.get('timestamp', 'unknown')}")
            
            return step, extra_state
            
        except Exception as e:
            print(f"加载检查点失败: {e}")
            traceback.print_exc()
            return None
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """查找最新的检查点"""
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path):
            return latest_path
        
        # 查找所有检查点
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith("checkpoint_step_") and f.endswith(".pt"):
                try:
                    step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
                    checkpoints.append((step, os.path.join(self.checkpoint_dir, f)))
                except ValueError:
                    continue
        
        if checkpoints:
            checkpoints.sort(reverse=True)
            return checkpoints[0][1]
        
        return None
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点，只保留最近的 N 个"""
        try:
            checkpoints = []
            for f in os.listdir(self.checkpoint_dir):
                if f.startswith("checkpoint_step_") and f.endswith(".pt"):
                    try:
                        step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
                        checkpoints.append((step, os.path.join(self.checkpoint_dir, f)))
                    except ValueError:
                        continue
            
            if len(checkpoints) > self.keep_last_n:
                checkpoints.sort(reverse=True)
                for _, path in checkpoints[self.keep_last_n:]:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        except Exception as e:
            print(f"清理旧检查点失败: {e}")

class NaNDetector:
    """NaN 检测器 - 检测和处理 NaN/Inf"""
    
    @staticmethod
    def check_tensor(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """检查张量是否包含 NaN 或 Inf"""
        if torch.isnan(tensor).any():
            print(f"警告: {name} 包含 NaN")
            return False
        if torch.isinf(tensor).any():
            print(f"警告: {name} 包含 Inf")
            return False
        return True
    
    @staticmethod
    def check_model(model: torch.nn.Module) -> bool:
        """检查模型参数是否包含 NaN 或 Inf"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not NaNDetector.check_tensor(param.grad, f"grad of {name}"):
                    return False
            if not NaNDetector.check_tensor(param.data, f"param {name}"):
                return False
        return True
    
    @staticmethod
    def sanitize_tensor(tensor: torch.Tensor, replace_value: float = 0.0) -> torch.Tensor:
        """清理张量中的 NaN 和 Inf"""
        return torch.nan_to_num(tensor, nan=replace_value, posinf=replace_value, neginf=replace_value)

def create_safe_training_environment(output_dir: str = "outputs"):
    """创建安全的训练环境"""
    safety = SafetyWrapper(output_dir)
    checkpoint_mgr = CheckpointManager(output_dir)
    
    return safety, checkpoint_mgr

# 使用示例
"""
# 在训练脚本中使用

from deguc.utils.safety import create_safe_training_environment, NaNDetector

# 创建安全环境
safety, checkpoint_mgr = create_safe_training_environment("outputs")

# 尝试恢复训练
latest_ckpt = checkpoint_mgr.find_latest_checkpoint()
if latest_ckpt:
    result = checkpoint_mgr.load_checkpoint(latest_ckpt, model, optimizer)
    if result:
        start_step, extra_state = result
        print(f"从步数 {start_step} 恢复训练")

# 训练循环
for step in range(start_step, total_steps):
    try:
        # 安全的前向传播
        logits, balance_loss = safety.safe_forward(model, input_ids, attention_mask)
        
        # 计算损失
        loss = criterion(logits, labels) + 0.01 * balance_loss
        
        # 检查 NaN
        if not NaNDetector.check_tensor(loss, "loss"):
            print(f"步数 {step}: 损失为 NaN，跳过此批次")
            continue
        
        # 安全的反向传播
        optimizer.zero_grad()
        safety.safe_backward(loss, optimizer, scaler, grad_clip=1.0)
        
        # 检查模型参数
        if step % 100 == 0:
            if not NaNDetector.check_model(model):
                print(f"步数 {step}: 模型参数包含 NaN，停止训练")
                break
        
        # 定期保存检查点
        if step % 1000 == 0:
            checkpoint_mgr.save_checkpoint(
                step, model, optimizer, scheduler,
                extra_state={"best_acc": best_acc}
            )
    
    except Exception as e:
        print(f"步数 {step} 出错: {e}")
        safety.log_error("TrainingError", str(e), {"step": step})
        # 可以选择继续或停止
        break
"""



