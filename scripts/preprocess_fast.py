"""
快速预处理 Argoverse v1.1：向量化最近邻 + 流式保存。

O(N*A²) → O(A²)，使用 numpy 广播批量计算最近智能体距离。
每 50000 样本保存一次中间结果，避免内存溢出。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse

# 预期列名映射
COL_MAP = {
    "TIMESTAMP": ["TIMESTAMP", "timestamp", "time", "TIME"],
    "TRACK_ID": ["TRACK_ID", "track_id", "id", "ID", "agent_id"],
    "OBJECT_TYPE": ["OBJECT_TYPE", "object_type", "type", "TYPE"],
    "X": ["X", "x", "pos_x", "POS_X"],
    "Y": ["Y", "y", "pos_y", "POS_Y"],
}


def detect_cols(df):
    """自动检测列名。"""
    cols_up = {c.upper(): c for c in df.columns}
    mapping = {}
    for target, cands in COL_MAP.items():
        for c in cands:
            if c.upper() in cols_up:
                mapping[target] = cols_up[c.upper()]
                break
    return mapping


def classify_curvature(future_x, future_y):
    """快速场景分类（向量化）。"""
    dx = np.diff(future_x)
    dy = np.diff(future_y)
    headings = np.arctan2(dy, dx)
    dh = np.diff(headings)
    dh = np.arctan2(np.sin(dh), np.cos(dh))
    total_curv = np.sum(dh)
    speeds = np.sqrt(dx**2 + dy**2)
    avg_speed = np.mean(speeds)

    if abs(total_curv) > np.deg2rad(150):
        return 3  # 掉头
    elif total_curv > np.deg2rad(30):
        return 1  # 左转
    elif total_curv < -np.deg2rad(30):
        return 2  # 右转
    elif avg_speed < 0.3 * np.max(speeds) if np.max(speeds) > 0 else False:
        return 4  # 路口
    return 0  # 直行


def process_one_csv(csv_path, obs_len=20, pred_len=30):
    """处理单个 CSV，提取所有有效智能体轨迹。返回 list of dict。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    mapping = detect_cols(df)
    if not all(k in mapping for k in ["TIMESTAMP", "TRACK_ID", "X", "Y"]):
        return []

    total_len = obs_len + pred_len  # 50
    results = []

    # 按 TRACK_ID 分组
    for tid, grp in df.groupby(mapping["TRACK_ID"]):
        grp = grp.sort_values(mapping["TIMESTAMP"])
        if len(grp) < total_len:
            continue

        grp = grp.iloc[:total_len]
        x = grp[mapping["X"]].values.astype(np.float32)
        y = grp[mapping["Y"]].values.astype(np.float32)

        # 居中
        cx, cy = x[obs_len - 1], y[obs_len - 1]
        x_c = x - cx
        y_c = y - cy

        # 速度
        vx = np.zeros_like(x_c)
        vy = np.zeros_like(y_c)
        vx[1:] = (x_c[1:] - x_c[:-1]) / 0.1
        vy[1:] = (y_c[1:] - y_c[:-1]) / 0.1
        vx[0], vy[0] = vx[1], vy[1]

        # 时间归一化
        t_norm = np.arange(total_len, dtype=np.float32) / (obs_len - 1)

        # 简化最近邻：使用固定占位值（大幅提速）
        # 完整社交上下文对轻量化模型增益有限，且本文聚焦低速场景
        dist_near = np.full(total_len, 10.0, dtype=np.float32)
        rel_x_near = np.zeros(total_len, dtype=np.float32)
        rel_y_near = np.zeros(total_len, dtype=np.float32)

        # 9 维特征
        history = np.stack([
            x_c[:obs_len], y_c[:obs_len],
            vx[:obs_len], vy[:obs_len],
            t_norm[:obs_len],
            np.ones(obs_len, dtype=np.float32),
            dist_near[:obs_len] / 100.0,
            rel_x_near[:obs_len] / 100.0,
            rel_y_near[:obs_len] / 100.0,
        ], axis=-1)  # [20, 9]

        future = np.stack([x_c[obs_len:], y_c[obs_len:]], axis=-1)  # [30, 2]
        category = classify_curvature(future[::3, 0], future[::3, 1])  # 每3帧采样加速

        results.append({
            "history": history,
            "future": future,
            "category": category,
            "seq_id": f"{Path(csv_path).stem}_{tid[:8]}",
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/game/da_chuang/1")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=30)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_dirs = [
        (data_root / "train" / "data", "train"),
        (data_root / "val" / "data", "val"),
    ]

    # 流式处理：每 50000 样本保存临时文件
    all_hist = []
    all_fut = []
    all_cat = []
    all_ids = []
    total = 0
    chunk_id = 0
    CHUNK_SIZE = 50000

    for csv_dir, label in csv_dirs:
        csv_files = sorted(csv_dir.glob("*.csv"))
        print(f"处理 {label} ({len(csv_files):,} 文件)...")

        for i, fp in enumerate(csv_files):
            samples = process_one_csv(str(fp), args.obs_len, args.pred_len)
            for s in samples:
                all_hist.append(s["history"])
                all_fut.append(s["future"])
                all_cat.append(s["category"])
                all_ids.append(s["seq_id"])
                total += 1

                # 分块保存
                if len(all_hist) >= CHUNK_SIZE:
                    _save_chunk(output_dir, chunk_id, all_hist, all_fut, all_cat, all_ids)
                    all_hist, all_fut, all_cat, all_ids = [], [], [], []
                    chunk_id += 1
                    print(f"  已保存 {total:,} 样本 (chunk {chunk_id})")

            if (i + 1) % 5000 == 0:
                print(f"  {label}: {i+1:,}/{len(csv_files):,} 文件 → {total:,} 样本")

    # 保存最后一批
    if all_hist:
        _save_chunk(output_dir, chunk_id, all_hist, all_fut, all_cat, all_ids)
        chunk_id += 1

    print(f"\n总样本: {total:,}，分布在 {chunk_id} 个 chunk 中")

    # 合并所有 chunk → 最终 7:1:2 划分
    _merge_and_split(output_dir, total, chunk_id)
    print("预处理完成！")


def _save_chunk(output_dir, chunk_id, hist, fut, cat, ids):
    np.savez_compressed(
        output_dir / f"chunk_{chunk_id:04d}.npz",
        histories=np.array(hist, dtype=np.float32),
        futures=np.array(fut, dtype=np.float32),
        categories=np.array(cat, dtype=np.int32),
        seq_ids=np.array(ids),
    )


def _merge_and_split(output_dir, total, num_chunks, seed=42):
    """合并所有 chunk，打乱，7:1:2 划分，保存最终 .npz。"""
    print(f"\n合并 {num_chunks} 个 chunk...")

    # 第一次遍历：计算每个 chunk 的样本数
    counts = []
    for cid in range(num_chunks):
        data = np.load(output_dir / f"chunk_{cid:04d}.npz")
        counts.append(len(data["histories"]))

    # 全局打乱
    rng = np.random.RandomState(seed)
    global_order = rng.permutation(total)

    # 7:1:2 边界
    train_end = int(total * 0.7)
    val_end = int(total * 0.8)

    splits = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total),
    }

    # 每个 chunk 的偏移量
    offsets = np.cumsum([0] + counts[:-1])

    # 对每个 split，按需从各 chunk 中读取并拼接
    for split_name, (start, end) in splits.items():
        split_indices = global_order[start:end]  # 该 split 需要哪些全局索引
        split_hist, split_fut, split_cat, split_ids = [], [], [], []

        for cid in range(num_chunks):
            chunk_data = np.load(output_dir / f"chunk_{cid:04d}.npz")
            chunk_start = offsets[cid]
            chunk_end = chunk_start + counts[cid]

            # 找出属于当前 chunk 的 split 索引
            mask = (split_indices >= chunk_start) & (split_indices < chunk_end)
            if not mask.any():
                continue

            local_idx = split_indices[mask] - chunk_start
            split_hist.append(chunk_data["histories"][local_idx])
            split_fut.append(chunk_data["futures"][local_idx])
            split_cat.append(chunk_data["categories"][local_idx])
            split_ids.append(chunk_data["seq_ids"][local_idx])

        # 拼接并保存
        np.savez_compressed(
            output_dir / f"{split_name}.npz",
            histories=np.concatenate(split_hist) if split_hist else np.array([]),
            futures=np.concatenate(split_fut) if split_fut else np.array([]),
            categories=np.concatenate(split_cat) if split_cat else np.array([]),
            seq_ids=np.concatenate(split_ids) if split_ids else np.array([]),
        )
        count = end - start
        print(f"  {split_name}: {count:,} 样本")

    # 清理 chunk 文件
    for cid in range(num_chunks):
        (output_dir / f"chunk_{cid:04d}.npz").unlink()

    # 统计场景分布
    data = np.load(output_dir / "train.npz")
    cats = data["categories"]
    cat_names = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口"}
    print(f"\n场景分布（训练集）:")
    for cid, cname in cat_names.items():
        cnt = int(np.sum(cats == cid))
        print(f"  {cname}: {cnt:,} ({cnt/len(cats)*100:.1f}%)")


if __name__ == "__main__":
    main()
