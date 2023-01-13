import psutil
import gc

import torch
from deepspeed import comm as dist

from deepspeed.utils import groups, logger
from deepspeed.runtime.constants import PIPE_REPLICATED

# pt-1.9 deprecations
if hasattr(torch.cuda, "memory_reserved"):
    torch_memory_reserved = torch.cuda.memory_reserved
else:
    torch_memory_reserved = torch.cuda.memory_allocated
if hasattr(torch.cuda, "max_memory_reserved"):
    torch_max_memory_reserved = torch.cuda.max_memory_reserved
else:
    torch_max_memory_reserved = torch.cuda.memory_cached


def see_memory_usage(message, cpu_only=False, force=False):
    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    logger.info(message)
    if not cpu_only:
        logger.info(
            f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
            Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
            CA {round(torch_memory_reserved() / (1024 * 1024 * 1024),2)} GB \
            Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logger.info(
        f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats") and not cpu_only:  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()