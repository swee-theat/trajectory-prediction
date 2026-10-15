"""可复现性工具：固定随机种子。"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """固定 Python、NumPy、PyTorch 的随机种子，确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确定性算法（可能略微降低性能，但在小模型上影响可忽略）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
