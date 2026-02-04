from dataclasses import dataclass
from typing import Optional

@dataclass
class QuantizationSchedule:
    quantize_at_step: int
    finetune_extra_steps: int
    freeze_after_quant: bool = True
    unfreeze_at_step: Optional[int] = None

    def in_warmup(self, step):
        return step < self.quantize_at_step

    def is_quantization_step(self, step):
        return step == self.quantize_at_step

    def in_finetune(self, step):
        return step > self.quantize_at_step and step <= (self.quantize_at_step + self.finetune_extra_steps)

    def should_unfreeze(self, step):
        if self.unfreeze_at_step is None:
            return False
        return step == self.unfreeze_at_step