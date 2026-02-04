import torch
import os
from typing import Dict

class ExpertOffloadManager:
    def __init__(self, offload_dir="offload_cache", to_disk=False):
        self.offload_dir = offload_dir
        self.to_disk = to_disk
        os.makedirs(offload_dir, exist_ok=True)
        self.offloaded = {}
        self.offloaded_disk_index = set()

    def offload(self, expert_id: int, module):
        A_key = f"{expert_id}_A"
        B_key = f"{expert_id}_B"
        if A_key not in module.expert_residuals:
            return
        A = module.expert_residuals[A_key].data.detach().cpu()
        B = module.expert_residuals[B_key].data.detach().cpu()
        if self.to_disk:
            torch.save({"A": A, "B": B}, os.path.join(self.offload_dir, f"expert_{expert_id}.pt"))
            self.offloaded_disk_index.add(expert_id)
        else:
            self.offloaded[expert_id] = {"A": A, "B": B}
        del module.expert_residuals[A_key]
        del module.expert_residuals[B_key]

    def reload_if_needed(self, expert_id: int, module, input_dim, rank, output_dim, dtype):
        A_key = f"{expert_id}_A"
        if A_key in module.expert_residuals:
            return False
        if expert_id in self.offloaded:
            rec = self.offloaded[expert_id]
        elif expert_id in self.offloaded_disk_index:
            path = os.path.join(self.offload_dir, f"expert_{expert_id}.pt")
            rec = torch.load(path, map_location="cpu") if os.path.exists(path) else None
        else:
            rec = None
        import torch.nn as nn
        if rec is None:
            A = torch.empty(input_dim, rank, dtype=dtype)
            B = torch.empty(output_dim, rank, dtype=dtype)
            torch.nn.init.xavier_uniform_(A)
            torch.nn.init.xavier_uniform_(B)
        else:
            A = rec["A"]
            B = rec["B"]
        module.expert_residuals[f"{expert_id}_A"] = nn.Parameter(A.to(dtype))
        module.expert_residuals[f"{expert_id}_B"] = nn.Parameter(B.to(dtype))
        return True