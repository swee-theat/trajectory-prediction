"""早停工具：监控验证指标，在停止提升时终止训练。"""

import numpy as np
from typing import Optional


class EarlyStopping:
    """早停器：监控指定指标的提升情况。

    使用示例:
        early_stop = EarlyStopping(patience=5, mode="min")
        for epoch in range(epochs):
            val_metric = validate()
            should_stop = early_stop(val_metric)
            if should_stop:
                break
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 1e-4,
    ):
        """初始化早停器。

        参数:
            patience: 容忍多少个 epoch 没有提升
            mode: "min" 表示指标越小越好，"max" 表示越大越好
            min_delta: 视为提升的最小变化量
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, metric_value: float, epoch: int) -> bool:
        """检查是否需要早停。

        参数:
            metric_value: 当前 epoch 的验证指标值
            epoch: 当前 epoch 编号

        返回:
            True 表示应该停止训练
        """
        score = metric_value

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:  # "max"
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False

    def get_best_info(self) -> dict:
        """获取最佳 epoch 信息。"""
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
        }
