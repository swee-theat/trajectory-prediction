#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""02 数据诊断：统计 future 段 avg_speed 分布，为低速阈值校准找双峰谷底。

只读不改（不写任何 npz），零风险。输出：
  1. 各类别的样本数 + avg_speed 分位数
  2. 掉头类在不同低速阈值下"将被剥离"的样本占比（用于定位停车峰/行驶峰的谷底）
  3. 模拟重分类：把低于阈值的样本从掉头类移到路口类，看掉头占比如何下降

avg_speed 口径与 classify_scene_type 完全一致：future 段逐帧位移的均值（m/帧，10Hz 下 1 帧=0.1s）。

用法:
    python scripts/diagnose_low_speed.py
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

CAT_NAMES = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口/停车"}


def avg_speed_of(futures: np.ndarray) -> np.ndarray:
    """计算每样本 future 段逐帧位移均值（m/帧），与 classify_scene_type 口径一致。"""
    dx = np.diff(futures[..., 0], axis=1)   # [N, T-1]
    dy = np.diff(futures[..., 1], axis=1)
    step = np.sqrt(dx ** 2 + dy ** 2)       # [N, T-1] 每帧位移
    return step.mean(axis=1)                # [N]


def main():
    npz_path = Path("data/processed/test.npz")
    if not npz_path.exists():
        print(f"错误：找不到 {npz_path}")
        return

    data = np.load(npz_path)
    futures = data["futures"]
    categories = data["categories"]
    N = len(categories)
    avg = avg_speed_of(futures)

    print(f"样本数: {N:,}")
    print("\n各类别样本数与 avg_speed 分位数（m/帧）:")
    for cid, cname in CAT_NAMES.items():
        m = categories == cid
        if m.sum() == 0:
            continue
        a = avg[m]
        print(f"  {cname}: n={m.sum():,}  median={np.median(a):.4f}  "
              f"p5={np.percentile(a, 5):.4f}  p25={np.percentile(a, 25):.4f}  "
              f"p75={np.percentile(a, 75):.4f}  p95={np.percentile(a, 95):.4f}")

    # 掉头类专项：分位数
    ut_mask = categories == 3
    ut_avg = avg[ut_mask]
    n_ut = int(ut_mask.sum())
    print(f"\n掉头类 n={n_ut:,}，avg_speed 分位数:")
    for q in [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 99]:
        print(f"  p{q}: {np.percentile(ut_avg, q):.4f}")

    # 不同阈值下，掉头类中"低于阈值"的样本占比 + 模拟重分类后的掉头占比
    print(f"\n阈值扫描（当前掉头占比 {n_ut / N * 100:.1f}%）:")
    print(f"  {'阈值(m/帧)':>10} {'掉头内低速占比':>14} {'移出样本数':>10} {'重分类后掉头占比':>14}")
    for th in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]:
        low_in_ut = (ut_avg < th).sum()
        frac_in_ut = low_in_ut / n_ut * 100
        new_ut_frac = (n_ut - low_in_ut) / N * 100
        print(f"  {th:>10.2f} {frac_in_ut:>13.1f}% {low_in_ut:>10,} {new_ut_frac:>13.1f}%")

    # 画图：全体 + 掉头类 的 avg_speed 直方图（聚焦 0~0.3 找谷底）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(avg, bins=300, range=(0, 1.5), log=True, color="steelblue")
    axes[0].set_title(f"全体样本 avg_speed 分布 (n={N:,})")
    axes[0].set_xlabel("avg_speed (m/帧)")
    axes[0].set_ylabel("样本数 (log)")

    axes[1].hist(ut_avg, bins=300, range=(0, 0.3), log=True, color="darkorange")
    axes[1].axvline(0.05, color="r", ls="--", lw=1.5, label="0.05 参考线")
    axes[1].set_title(f"掉头类 avg_speed 分布 (0~0.3, n={n_ut:,})")
    axes[1].set_xlabel("avg_speed (m/帧)")
    axes[1].legend()

    fig.tight_layout()
    out = Path("outputs/figures/low_speed_diagnosis.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n诊断图已保存: {out}")


if __name__ == "__main__":
    main()
