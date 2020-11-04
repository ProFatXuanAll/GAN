import random

from typing import Tuple

import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_noise(shape: Tuple[int, int], is_normal: False) -> torch.Tensor:
    if is_normal:
        return torch.randn(shape)
    return torch.rand(shape)
