"""训练日志工具：终端 tqdm + 文件日志 + TensorBoard。"""

import logging
import sys
from pathlib import Path
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


def setup_logger(log_dir: str, name: str = "trajectory") -> logging.Logger:
    """创建同时输出到终端和文件的 logger。

    参数:
        log_dir: 日志目录
        name: logger 名称

    返回:
        配置好的 logging.Logger 实例
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 文件输出
    file_handler = logging.FileHandler(path / "training.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


def create_tb_writer(log_dir: str) -> SummaryWriter:
    """创建 TensorBoard SummaryWriter。"""
    return SummaryWriter(log_dir)
