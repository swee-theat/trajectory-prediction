"""生成模拟轨迹数据：用于在真实数据下载完成前，端到端验证完整流程。

模拟数据特征：
- 284K 样本匹配 Argoverse 规模
- 5 种场景类型（直行/左转/右转/掉头/路口）
- 包含多智能体交互特征
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import argparse

from data.preprocessing import classify_scene_type


def generate_straight_track(obs_len=20, pred_len=30):
    """生成直行轨迹。"""
    total = obs_len + pred_len
    t = np.arange(total) * 0.1
    # 匀速直线运动 + 轻微噪声
    vx = np.random.uniform(2, 8)
    vy = np.random.uniform(-0.5, 0.5)
    x = vx * t + np.random.randn(total) * 0.05
    y = vy * t + np.random.randn(total) * 0.05
    return x, y


def generate_left_turn_track(obs_len=20, pred_len=30):
    """生成左转轨迹。"""
    total = obs_len + pred_len
    t = np.arange(total) * 0.1
    v = np.random.uniform(3, 6)
    # 先直行再左转
    turn_start = np.random.randint(obs_len - 5, obs_len + 5)
    theta = np.zeros(total)
    for i in range(turn_start, total):
        theta[i] = theta[i - 1] + 0.05  # 累积转角
    x = np.zeros(total)
    y = np.zeros(total)
    for i in range(1, total):
        heading = theta[i]
        x[i] = x[i - 1] + v * np.cos(heading) * 0.1
        y[i] = y[i - 1] + v * np.sin(heading) * 0.1
    return x, y


def generate_right_turn_track(obs_len=20, pred_len=30):
    """生成右转轨迹。"""
    total = obs_len + pred_len
    t = np.arange(total) * 0.1
    v = np.random.uniform(3, 6)
    turn_start = np.random.randint(obs_len - 5, obs_len + 5)
    theta = np.zeros(total)
    for i in range(turn_start, total):
        theta[i] = theta[i - 1] - 0.05  # 负转角 = 右转
    x = np.zeros(total)
    y = np.zeros(total)
    for i in range(1, total):
        heading = theta[i]
        x[i] = x[i - 1] + v * np.cos(heading) * 0.1
        y[i] = y[i - 1] + v * np.sin(heading) * 0.1
    return x, y


def generate_u_turn_track(obs_len=20, pred_len=30):
    """生成掉头轨迹。"""
    total = obs_len + pred_len
    v = np.random.uniform(2, 4)
    turn_start = np.random.randint(obs_len - 5, obs_len + 5)
    theta = np.zeros(total)
    # 大幅转角 180°
    for i in range(turn_start, total):
        theta[i] = theta[i - 1] + np.pi / (total - turn_start)
    x = np.zeros(total)
    y = np.zeros(total)
    for i in range(1, total):
        heading = theta[i]
        x[i] = x[i - 1] + v * np.cos(heading) * 0.1
        y[i] = y[i - 1] + v * np.sin(heading) * 0.1
    return x, y


def generate_intersection_track(obs_len=20, pred_len=30):
    """生成路口轨迹（减速+转向）。"""
    total = obs_len + pred_len
    t = np.arange(total) * 0.1
    # 先快速直行再减速转向
    v_fast = np.random.uniform(5, 10)
    v_slow = np.random.uniform(1, 3)
    slow_start = np.random.randint(10, 18)
    speeds = np.full(total, v_fast)
    speeds[slow_start:] = v_slow

    theta = np.zeros(total)
    if np.random.random() > 0.5:
        for i in range(slow_start, total):
            theta[i] = theta[i - 1] + 0.03
    else:
        for i in range(slow_start, total):
            theta[i] = theta[i - 1] - 0.03

    x = np.zeros(total)
    y = np.zeros(total)
    for i in range(1, total):
        heading = theta[i]
        x[i] = x[i - 1] + speeds[i] * np.cos(heading) * 0.1
        y[i] = y[i - 1] + speeds[i] * np.sin(heading) * 0.1
    return x, y


def build_features(x, y, vx, vy, obs_len, pred_len, all_agents):
    """构建 9 维特征向量。

    返回: history [obs_len, 9], future [pred_len, 2], category (int)
    """
    total = obs_len + pred_len
    x_center = x[obs_len - 1]
    y_center = y[obs_len - 1]
    x_c = x - x_center
    y_c = y - y_center

    # 最近智能体特征（简化版）
    dist_nearest = np.full(total, 15.0, dtype=np.float32)
    rel_x_nearest = np.zeros(total, dtype=np.float32)
    rel_y_nearest = np.zeros(total, dtype=np.float32)

    t_norm = np.arange(total, dtype=np.float32) / (obs_len - 1)

    history = np.stack([
        x_c[:obs_len],
        y_c[:obs_len],
        vx[:obs_len],
        vy[:obs_len],
        t_norm[:obs_len],
        np.ones(obs_len, dtype=np.float32),
        dist_nearest[:obs_len] / 100.0,
        rel_x_nearest[:obs_len] / 100.0,
        rel_y_nearest[:obs_len] / 100.0,
    ], axis=-1)

    future = np.stack([x_c[obs_len:], y_c[obs_len:]], axis=-1)
    category = classify_scene_type(future)

    return history, future, category


def generate_dataset(num_samples=284416, obs_len=20, pred_len=30, output_dir="data/processed"):
    """生成完整模拟数据集，匹配 Argoverse 规模。

    场景分布（大致匹配真实数据）:
    - 直行: 50%
    - 左转: 15%
    - 右转: 15%
    - 掉头: 5%
    - 路口: 15%
    """
    total = obs_len + pred_len
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generators = [
        (generate_straight_track, 0.50),
        (generate_left_turn_track, 0.15),
        (generate_right_turn_track, 0.15),
        (generate_u_turn_track, 0.05),
        (generate_intersection_track, 0.15),
    ]

    # 生成轨迹
    all_histories = []
    all_futures = []
    all_categories = []
    all_seq_ids = []

    print(f"生成 {num_samples:,} 个模拟样本...")

    for i in range(num_samples):
        # 随机选择场景类型
        r = np.random.random()
        cumsum = 0
        gen_fn = generators[0][0]
        for fn, prob in generators:
            cumsum += prob
            if r < cumsum:
                gen_fn = fn
                break

        # 计算速度
        x, y = gen_fn(obs_len, pred_len)
        dt = 0.1
        vx = np.zeros_like(x)
        vy = np.zeros_like(y)
        vx[1:] = (x[1:] - x[:-1]) / dt
        vy[1:] = (y[1:] - y[:-1]) / dt
        if len(vx) > 1:
            vx[0] = vx[1]
            vy[0] = vy[1]

        # 构建特征
        history, future, category = build_features(x, y, vx, vy, obs_len, pred_len, {})

        all_histories.append(history)
        all_futures.append(future)
        all_categories.append(category)
        all_seq_ids.append(f"sim_{i:06d}")

        if (i + 1) % 50000 == 0:
            print(f"  已生成 {i+1:,}/{num_samples:,}...")

    # 数组化
    histories = np.array(all_histories, dtype=np.float32)
    futures = np.array(all_futures, dtype=np.float32)
    categories = np.array(all_categories, dtype=np.int32)
    seq_ids = np.array(all_seq_ids)

    # 7:1:2 划分
    n = len(histories)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    splits = {
        "train": (slice(0, train_end),),
        "val": (slice(train_end, val_end),),
        "test": (slice(val_end, n),),
    }

    for split_name, (idx,) in splits.items():
        s = slice(idx.start, idx.stop)
        np.savez_compressed(
            output_path / f"{split_name}.npz",
            histories=histories[s],
            futures=futures[s],
            categories=categories[s],
            seq_ids=seq_ids[s],
        )
        print(f"  {split_name}: {s.stop - s.start:,} 样本 → {output_path / f'{split_name}.npz'}")

    # 统计场景分布
    cat_names = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口"}
    print(f"\n场景分布:")
    for cat_id, cat_name in cat_names.items():
        count = np.sum(categories == cat_id)
        print(f"  {cat_name}: {count:,} ({count/len(categories)*100:.1f}%)")

    print(f"\n模拟数据生成完成！总样本: {len(histories):,}")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成模拟轨迹数据")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="生成样本数（默认50K，快速测试；完整284K需较长时间）")
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=30)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    print("=" * 60)
    print("  生成模拟轨迹数据集")
    print("=" * 60)
    generate_dataset(args.num_samples, args.obs_len, args.pred_len, args.output_dir)
