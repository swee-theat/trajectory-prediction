"""模型参数量统计工具，用于验证 198K 参数目标。"""

import torch.nn as nn
from typing import Dict, Tuple


def count_parameters(model: nn.Module, detailed: bool = True) -> Tuple[int, Dict[str, int]]:
    """统计模型参数量。

    参数:
        model: PyTorch 模型
        detailed: 是否打印逐模块明细

    返回:
        (总参数量, {模块名: 参数量}) 元组
    """
    total = 0
    per_module = {}

    for name, param in model.named_parameters():
        num = param.numel()
        per_module[name] = num
        total += num

    if detailed:
        print(f"\n{'='*60}")
        print(f"  模型参数量统计")
        print(f"{'='*60}")
        for name, num in per_module.items():
            print(f"  {name:<50s} {num:>8,d}")
        print(f"{'='*60}")
        print(f"  总参数量: {total:>8,d}")
        print(f"  目标:     198,000")
        print(f"  差异:     {total - 198000:>+8,d}")
        print(f"{'='*60}\n")

    return total, per_module
