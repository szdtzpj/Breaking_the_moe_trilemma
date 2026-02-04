import os
import torch
import torch.distributed as dist

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()

def init_distributed(backend="nccl"):
    if is_distributed():
        return
    if "RANK" not in os.environ:
        return
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    dist.barrier()

def cleanup_distributed():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()