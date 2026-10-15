#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模拟 02 修复后的重分类：avg_speed < 阈值的样本从各类别移到「路口/停车」类。

只读不改。输出不同阈值下各类别的样本数变化，用于评估低速前置判断的连带影响
（不仅净化掉头，也会把左转/右转里的低速污染一并剥离）。

用法:
    python scripts/simulate_reclassify.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

CAT_NAMES = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口/停车"}


def avg_speed_of(futures: np.ndarray) -> np.ndarray:
    dx = np.diff(futures[..., 0], axis=1)
    dy = np.diff(futures[..., 1], axis=1)
    return np.sqrt(dx ** 2 + dy ** 2).mean(axis=1)


def main():
    data = np.load("data/processed/test.npz")
    futures = data["futures"]
    cats = data["categories"].copy()
    avg = avg_speed_of(futures)
    N = len(cats)

    print(f"当前各类别分布 (n={N:,}):")
    for cid, cn in CAT_NAMES.items():
        m = cats == cid
        print(f"  {cn}: {m.sum():,} ({m.sum() / N * 100:.1f}%)")

    for th in [0.03, 0.05, 0.08]:
        new = cats.copy()
        low = avg < th
        new[low] = 4  # 低速 → 路口/停车
        n_low = int(low.sum())
        print(f"\n阈值 {th} m/帧 重分类后（移出 {n_low:,} 个低速样本，占 {n_low / N * 100:.1f}%）:")
        for cid, cn in CAT_NAMES.items():
            m = new == cid
            delta = int(m.sum()) - int((cats == cid).sum())
            sign = "+" if delta >= 0 else ""
            print(f"  {cn}: {m.sum():,} ({m.sum() / N * 100:.1f}%)  [{sign}{delta:,}]")


if __name__ == "__main__":
    main()
