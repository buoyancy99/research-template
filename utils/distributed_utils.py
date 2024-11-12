from typing import Callable
import torch
import torch.distributed as dist
from lightning.pytorch.utilities.rank_zero import rank_zero_only

is_rank_zero = rank_zero_only.rank == 0

rank_zero_print = rank_zero_only(print)
