"""YAML 配置加载工具。使用 omegaconf 支持配置继承与覆盖。"""

from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str, base_path: Optional[str] = None) -> DictConfig:
    """加载 YAML 配置文件，可选地继承默认配置。

    参数:
        config_path: 配置文件路径
        base_path: 基础配置路径（用于继承，如消融实验继承默认配置）

    返回:
        OmegaConf DictConfig 对象
    """
    if base_path is not None:
        base_cfg = OmegaConf.load(base_path)
        override_cfg = OmegaConf.load(config_path)
        config = OmegaConf.merge(base_cfg, override_cfg)
    else:
        config = OmegaConf.load(config_path)

    return config


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """将 OmegaConf DictConfig 转为普通 Python dict（方便传给函数）。"""
    return OmegaConf.to_container(config, resolve=True)


def save_config(config: DictConfig, output_path: str) -> None:
    """将配置保存为 YAML 文件（用于实验复现记录）。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(config))
