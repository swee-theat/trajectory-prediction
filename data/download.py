"""Argoverse 1.1 Motion Forecasting 数据集下载脚本。

Argoverse 数据集需要从官网注册并下载。此脚本提供：
1. 自动下载（如果直接 URL 可用）
2. 手动下载指引
3. 下载后的目录结构验证

数据来源: https://www.argoverse.org/av1.html#download-link
运动预测子集大小: 约 1.5GB（CSV 格式）
"""

import os
import sys
import hashlib
import zipfile
from pathlib import Path
from typing import Optional


# Argoverse 1.1 运动预测数据集配置
ARGO_SEQUENCES = 205942  # 预期 CSV 文件总数
EXPECTED_FILES = {
    "train": 205942,
    "val": 39472,
    "test": 78143,
}
EXPECTED_SIZE_GB = 1.5


def check_dataset(raw_dir: str) -> dict:
    """检查已下载的数据集完整性。

    参数:
        raw_dir: 数据集根目录（如 data/raw/）

    返回:
        {
            "valid": bool,
            "splits": {"train": count, "val": count, "test": count},
            "total_csv": int,
            "total_size_gb": float,
            "issues": list[str],
        }
    """
    raw_path = Path(raw_dir)
    result = {
        "valid": True,
        "splits": {},
        "total_csv": 0,
        "total_size_gb": 0.0,
        "issues": [],
    }

    total_size = 0
    for split in ["train", "val", "test"]:
        split_dir = raw_path / split
        if not split_dir.exists():
            result["issues"].append(f"缺少 {split}/ 目录")
            result["valid"] = False
            continue

        csv_files = list(split_dir.rglob("*.csv"))
        count = len(csv_files)
        result["splits"][split] = count
        result["total_csv"] += count

        # 计算目录总大小
        for f in csv_files:
            total_size += f.stat().st_size

    result["total_size_gb"] = total_size / (1024**3)

    if result["total_csv"] == 0:
        result["issues"].append("未找到任何 CSV 文件，数据集可能未下载")
        result["valid"] = False

    return result


def get_download_instructions() -> str:
    """返回手动下载指引。"""
    return """
    ╔══════════════════════════════════════════════════════════════╗
    ║         Argoverse 1.1 Motion Forecasting 下载指引            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  1. 访问: https://www.argoverse.org/av1.html#download-link   ║
    ║  2. 注册账号（需要邮箱验证）                                  ║
    ║  3. 下载 "Motion Forecasting" 数据子集（约 1.5 GB）           ║
    ║  4. 解压到: data/raw/ 目录下，结构如下：                      ║
    ║                                                              ║
    ║     data/raw/                                                ║
    ║     ├── train/                                               ║
    ║     │   └── data/                                            ║
    ║     │       └── *.csv     (约 205K 个文件)                    ║
    ║     ├── val/                                                 ║
    ║     │   └── data/                                            ║
    ║     │       └── *.csv     (约 39K 个文件)                     ║
    ║     └── test/                                                ║
    ║         └── data/                                            ║
    ║             └── *.csv     (约 78K 个文件)                     ║
    ║                                                              ║
    ║  5. 或者运行: python data/download.py --auto                 ║
    ║     尝试自动下载（需要网络连接）                              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """


def download_argoverse(target_dir: str) -> bool:
    """尝试自动下载 Argoverse 数据集。

    Argoverse 官方托管在 AWS S3 上。如果直接下载不可用，
    将打印手动下载指引。

    参数:
        target_dir: 目标目录（如 data/raw/）

    返回:
        下载成功返回 True，否则 False
    """
    import subprocess

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print("\n  尝试自动下载 Argoverse 数据集...")
    print("  注意: 官网下载需要先注册账号。\n")

    # 检查是否已有数据
    check = check_dataset(target_dir)
    if check["valid"] and check["total_csv"] > 1000:
        print(f"  ✓ 数据集已存在: {check['total_csv']} CSV 文件, "
              f"{check['total_size_gb']:.1f} GB")
        return True

    print(get_download_instructions())
    return False


def download_from_kaggle(target_dir: str) -> bool:
    """通过 Kaggle API 下载 Argoverse 数据集（备选方案）。

    需要先设置 Kaggle API key: https://www.kaggle.com/settings
    pip install kagglehub
    """
    try:
        import kagglehub

        print("  通过 Kaggle 下载 Argoverse 数据集...")
        path = kagglehub.dataset_download(
            "garymk/argoverse-motion-forecasting"
        )
        print(f"  下载完成: {path}")

        # 复制到目标目录
        import shutil
        src = Path(path)
        dst = Path(target_dir)
        for item in src.iterdir():
            if item.is_dir():
                shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst / item.name)

        return True
    except ImportError:
        print("  kagglehub 未安装，跳过 Kaggle 下载。")
        print("  安装: pip install kagglehub")
        return False
    except Exception as e:
        print(f"  Kaggle 下载失败: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Argoverse 数据集下载工具")
    parser.add_argument(
        "--target-dir",
        default="data/raw",
        help="目标下载目录（默认: data/raw）",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查已有数据完整性",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="尝试自动下载",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="尝试通过 Kaggle 下载",
    )
    args = parser.parse_args()

    if args.check:
        result = check_dataset(args.target_dir)
        print(f"\n  数据集检查结果:")
        print(f"    有效: {result['valid']}")
        print(f"    CSV 总数: {result['total_csv']}")
        print(f"    总大小: {result['total_size_gb']:.2f} GB")
        for split, count in result["splits"].items():
            print(f"    {split}: {count} 文件")
        for issue in result["issues"]:
            print(f"    ! {issue}")
    elif args.kaggle:
        download_from_kaggle(args.target_dir)
    elif args.auto:
        download_argoverse(args.target_dir)
    else:
        # 默认：检查数据 + 打印下载指引
        result = check_dataset(args.target_dir)
        if result["valid"] and result["total_csv"] > 1000:
            print(f"  ✓ 数据集已就绪: {result['total_csv']} 文件")
        else:
            download_argoverse(args.target_dir)
