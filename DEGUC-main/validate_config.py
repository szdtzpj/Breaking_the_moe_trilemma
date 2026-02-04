"""
Config File Validator - Ensure the config file is correct and reasonable
"""
import yaml
import sys
import os
from typing import Dict, Any, List, Tuple

class ConfigValidator:
    """Config Validator"""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_config(self, config_path: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate the config file

        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Check if file exists
        if not os.path.exists(config_path):
            self.errors.append(f"Config file does not exist: {config_path}")
            return False, self.errors, self.warnings

        # Load config
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to parse config file: {e}")
            return False, self.errors, self.warnings

        # Validate each section
        self._validate_task(config)
        self._validate_model(config)
        self._validate_moe(config)
        self._validate_train(config)
        self._validate_deguc_schedule(config)
        self._validate_quantization(config)
        self._validate_data(config)

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def _validate_task(self, config: Dict[str, Any]):
        """Validate task type"""
        if 'task' not in config:
            self.errors.append("Missing 'task' field")
            return

        task = config['task']
        valid_tasks = ['classification', 'causal_lm']

        if task not in valid_tasks:
            self.errors.append(f"Invalid task type: {task}, must be one of {valid_tasks}")

    def _validate_model(self, config: Dict[str, Any]):
        """Validate model configuration"""
        if 'model' not in config:
            self.errors.append("Missing 'model' configuration")
            return

        model = config['model']

        # Required fields
        required = ['d_model', 'num_heads', 'num_layers']
        for field in required:
            if field not in model:
                self.errors.append(f"Model is missing required field: {field}")

        # Validate numerical rationality
        if 'd_model' in model:
            d_model = model['d_model']
            if d_model <= 0:
                self.errors.append(f"d_model must be a positive number, current value: {d_model}")
            elif d_model % model.get('num_heads', 1) != 0:
                self.errors.append(f"d_model ({d_model}) must be divisible by num_heads ({model.get('num_heads')})")

        if 'num_heads' in model and model['num_heads'] <= 0:
            self.errors.append(f"num_heads must be a positive number, current value: {model['num_heads']}")

        if 'num_layers' in model:
            num_layers = model['num_layers']
            if num_layers <= 0:
                self.errors.append(f"num_layers must be a positive number, current value: {num_layers}")
            elif num_layers > 100:
                self.warnings.append(f"num_layers is too large ({num_layers}), which may cause memory issues")

    def _validate_moe(self, config: Dict[str, Any]):
        """Validate MoE configuration"""
        if 'model' not in config or 'moe' not in config['model']:
            self.errors.append("Missing 'model.moe' configuration")
            return

        moe = config['model']['moe']

        # Required fields
        required = ['num_experts', 'init_groups', 'rank', 'top_k']
        for field in required:
            if field not in moe:
                self.errors.append(f"MoE is missing required field: {field}")

        # Validate numerical rationality
        num_experts = moe.get('num_experts', 0)
        init_groups = moe.get('init_groups', 0)
        rank = moe.get('rank', 0)
        top_k = moe.get('top_k', 0)

        if num_experts <= 0:
            self.errors.append(f"num_experts must be a positive number, current value: {num_experts}")
        elif num_experts > 1024:
            self.warnings.append(f"num_experts is too large ({num_experts}), which may cause performance issues")

        if init_groups <= 0:
            self.errors.append(f"init_groups must be a positive number, current value: {init_groups}")
        elif init_groups > num_experts:
            self.errors.append(f"init_groups ({init_groups}) cannot be greater than num_experts ({num_experts})")

        if rank <= 0:
            self.errors.append(f"rank must be a positive number, current value: {rank}")
        elif rank > 256:
            self.warnings.append(f"rank is too large ({rank}), which may not bring better performance")

        if top_k <= 0:
            self.errors.append(f"top_k must be a positive number, current value: {top_k}")
        elif top_k > num_experts:
            self.warnings.append(f"top_k ({top_k}) is greater than num_experts ({num_experts}), it will be limited")

        # Validate dtype
        if 'param_dtype' in moe:
            dtype = moe['param_dtype']
            valid_dtypes = ['float32', 'fp32', 'float16', 'fp16', 'half', 'bfloat16', 'bf16']
            if dtype not in valid_dtypes:
                self.errors.append(f"Invalid param_dtype: {dtype}, must be one of {valid_dtypes}")

    def _validate_train(self, config: Dict[str, Any]):
        """Validate training configuration"""
        if 'train' not in config:
            self.errors.append("Missing 'train' configuration")
            return

        train = config['train']

        # Required fields
        required = ['total_steps', 'lr']
        for field in required:
            if field not in train:
                self.errors.append(f"Train is missing required field: {field}")

        # Validate learning rate
        if 'lr' in train:
            lr = train['lr']
            if lr <= 0:
                self.errors.append(f"lr must be a positive number, current value: {lr}")
            elif lr > 0.01:
                self.warnings.append(f"lr is too large ({lr}), MoE models usually require smaller learning rates (1e-4 to 5e-4)")

        # Validate steps
        if 'total_steps' in train:
            total_steps = train['total_steps']
            if total_steps <= 0:
                self.errors.append(f"total_steps must be a positive number, current value: {total_steps}")

        # Validate balance_loss_weight
        if 'balance_loss_weight' in train:
            weight = train['balance_loss_weight']
            if weight < 0:
                self.errors.append(f"balance_loss_weight cannot be a negative number, current value: {weight}")
            elif weight > 0.1:
                self.warnings.append(f"balance_loss_weight is too large ({weight}), recommended range: 0.01-0.05")

        # Validate gradient clipping
        if 'grad_clip' in train:
            grad_clip = train['grad_clip']
            if grad_clip <= 0:
                self.warnings.append(f"grad_clip should be a positive number, current value: {grad_clip}")

    def _validate_deguc_schedule(self, config: Dict[str, Any]):
        """Validate DEGUC schedule configuration"""
        if 'deguc_schedule' not in config:
            self.warnings.append("Missing 'deguc_schedule' configuration, default values will be used")
            return

        schedule = config['deguc_schedule']

        # Validate interval
        if 'clustering_interval' in schedule:
            interval = schedule['clustering_interval']
            if interval <= 0:
                self.errors.append(f"clustering_interval must be a positive number, current value: {interval}")
            elif interval < 100:
                self.warnings.append(f"clustering_interval is too small ({interval}), which may cause frequent clustering and affect performance")

        if 'offload_interval' in schedule:
            interval = schedule['offload_interval']
            if interval <= 0:
                self.errors.append(f"offload_interval must be a positive number, current value: {interval}")

        if 'min_offload_rate' in schedule:
            rate = schedule['min_offload_rate']
            if rate < 0 or rate > 1:
                self.errors.append(f"min_offload_rate must be in the range [0, 1], current value: {rate}")

    def _validate_quantization(self, config: Dict[str, Any]):
        """Validate quantization configuration"""
        if 'quantization' not in config:
            return  # Quantization is optional

        quant = config['quantization']

        if 'quantize_at_step' in quant:
            step = quant['quantize_at_step']
            total_steps = config.get('train', {}).get('total_steps', float('inf'))

            if step < 0:
                self.errors.append(f"quantize_at_step cannot be a negative number, current value: {step}")
            elif step >= total_steps:
                self.warnings.append(f"quantize_at_step ({step}) >= total_steps ({total_steps}), quantization will not be executed")

    def _validate_data(self, config: Dict[str, Any]):
        """Validate data configuration"""
        task = config.get('task', '')

        if task == 'classification':
            if 'classification' not in config:
                self.errors.append("Classification task is missing 'classification' configuration")
                return

            data = config['classification']

            # Check required fields
            if 'local_path' not in data:
                self.errors.append("Classification is missing 'local_path' field")
            else:
                # Check if data files exist
                local_path = data['local_path']
                required_files = ['train.tsv', 'validation.tsv', 'test.tsv']

                for file in required_files:
                    file_path = os.path.join(local_path, file)
                    if not os.path.exists(file_path):
                        self.warnings.append(f"Data file does not exist: {file_path}")

            # Validate batch size
            if 'batch_size' in data:
                batch_size = data['batch_size']
                if batch_size <= 0:
                    self.errors.append(f"batch_size must be a positive number, current value: {batch_size}")
                elif batch_size > 128:
                    self.warnings.append(f"batch_size is too large ({batch_size}), which may cause out of memory")

        elif task == 'causal_lm':
            if 'causal_lm' not in config:
                self.errors.append("Causal language model task is missing 'causal_lm' configuration")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python validate_config.py <config_file.yaml>")
        print("\nExample:")
        print("  python validate_config.py configs_training_config_8g_classification_Version2.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    print("=" * 60)
    print(f"Validating config file: {config_path}")
    print("=" * 60)

    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate_config(config_path)

    # Print errors
    if errors:
        print("\n❌ Errors found:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    # Print warnings
    if warnings:
        print("\n⚠️  Warnings:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    # Summary
    print("\n" + "=" * 60)
    if is_valid:
        if warnings:
            print("✓ Config file is valid (with warnings)")
            print("\nIt is recommended to adjust the config according to the warnings for better performance")
        else:
            print("✓ Config file is completely valid")
            print("\nYou can start training now!")
        return 0
    else:
        print("❌ Config file is invalid")
        print("\nPlease fix the above errors and try again")
        return 1

if __name__ == "__main__":
    sys.exit(main())