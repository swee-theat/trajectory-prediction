"""指标累加器：用于评估阶段批量收集和聚合指标。"""

import torch
import numpy as np
from typing import Dict, List


class MetricsTracker:
    """累积评估指标并计算均值/标准差。"""

    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """添加一个样本的指标值。"""
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(value)

    def update_batch(self, batch_metrics: Dict[str, torch.Tensor]) -> None:
        """添加一个 batch 的指标值（tensor 格式）。"""
        for key, value in batch_metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            if isinstance(value, torch.Tensor):
                self._metrics[key].extend(value.detach().cpu().tolist())
            elif isinstance(value, np.ndarray):
                self._metrics[key].extend(value.tolist())
            else:
                self._metrics[key].append(float(value))

    def compute(self) -> Dict[str, Dict[str, float]]:
        """计算所有累积指标的均值和标准差。

        返回:
            {指标名: {"mean": 均值, "std": 标准差}} 的嵌套字典
        """
        result = {}
        for key, values in self._metrics.items():
            arr = np.array(values)
            result[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
            }
        return result

    def reset(self) -> None:
        """清空所有累加值。"""
        self._metrics.clear()

    def get_summary_str(self) -> str:
        """返回格式化的指标摘要字符串。"""
        result = self.compute()
        lines = []
        for key, stats in result.items():
            lines.append(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        return "\n".join(lines)
