#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""车道线图构建脚本：轨迹聚合拟合车道中心线（第二阶段数据层）。

思路（用户拍板：不下载官方 HD map，用轨迹聚合拟合）：
    1. 按场景读原始 CSV，取所有长度足够的 agent 观测轨迹（绝对坐标，不居中）。
    2. 按「终点位置 + 平均航向」聚类成 L 条车道（轻量手写 KMeans，不引入 sklearn）。
    3. 每簇内轨迹逐帧平均 → 车道中心线。
    4. 每条中心线沿弧长均匀采样 M 个节点，场景固定 N = L×M = 32 节点（免 batch 内动态 pad）。
    5. 节点特征 6 维 [x, y, sinθ, cosθ, κ(曲率), v_limit(限速)]，坐标相对场景中心归一化。
    6. 邻接矩阵：前后边=同车道相邻节点；左右边=不同车道对应节点欧氏距离 < 3.5m。

padding 约定：
    - 不足 L 条车道的节点特征全填 0；
    - 邻接矩阵中 padding 节点的行列全置 0（不参与消息传递）；
    - node_mask 标记有效节点，GCN 输出后 padding 节点乘 mask 置 0。

输出 data/processed/lane_graphs_{split}.npz：
    nodes     [N_scene, 32, 6]   float32
    adj       [N_scene, 32, 32]  int8 (0/1)
    node_mask [N_scene, 32]      int8 (0/1)
    scene_ids [N_scene]          str   (csv_stem)

用法:
    python scripts/build_lane_graph.py --raw_dir data/raw/train --split train --limit 200
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path

from data.preprocessing import _normalize_columns, compute_velocities


# ── 固定超参数 ──────────────────────────────────────────
NUM_LANES = 4              # 最多车道数 L
NODES_PER_LANE = 8         # 每条车道采样节点数 M
NUM_NODES = NUM_LANES * NODES_PER_LANE  # 32
OBS_LEN = 20               # 观测帧数（与第一阶段一致）
LANE_WIDTH = 3.5           # 车道宽度阈值（米），判定左右边
NODE_DIM = 6               # 节点特征维度 [x, y, sinθ, cosθ, κ, v_limit]


def _simple_kmeans(features: np.ndarray, k: int, n_init: int = 5,
                   max_iter: int = 30, seed: int = 42) -> np.ndarray:
    """轻量 KMeans（k-means++ 初始化 + Lloyd 迭代），避免引入 sklearn 重依赖。

    参数:
        features: [n, d] 待聚类特征
        k: 簇数
    返回:
        labels: [n] 簇标签
    """
    n = features.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    k = min(k, n)
    if k == 1:
        return np.zeros(n, dtype=np.int32)

    rng = np.random.RandomState(seed)
    best_labels, best_inertia = None, float("inf")

    for _ in range(n_init):
        # k-means++ 初始化
        centers = [features[rng.randint(n)]]
        for _ in range(1, k):
            dist = np.min(
                [np.sum((features - c) ** 2, axis=1) for c in centers], axis=0
            )
            dist_sum = dist.sum()
            probs = dist / dist_sum if dist_sum > 0 else np.ones(n) / n
            centers.append(features[rng.choice(n, p=probs)])
        centers = np.array(centers, dtype=np.float32)

        for _ in range(max_iter):
            dist = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist, axis=1)
            new_centers = np.array([
                features[labels == c].mean(axis=0) if (labels == c).any() else centers[c]
                for c in range(k)
            ], dtype=np.float32)
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        dist = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist, axis=1)
        inertia = float(np.sum(dist[np.arange(n), labels]))
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels

    return best_labels.astype(np.int32)


def build_scene_lane_graph(df: pd.DataFrame):
    """为单个场景构建车道线图。

    参数:
        df: 标准化后的单场景 DataFrame（含 timestamp/track_id/object_type/x/y）

    返回:
        (nodes [32,6], adj [32,32], node_mask [32]) 或 None（无有效轨迹时）
    """
    # 1. 收集所有长度足够的 agent 观测轨迹（绝对坐标）
    tracks = {}  # track_id -> [obs_len, 2]
    for tid in df["track_id"].unique():
        tdf = df[df["track_id"] == tid].sort_values("timestamp")
        if len(tdf) >= OBS_LEN:
            x = tdf["x"].values[:OBS_LEN].astype(np.float32)
            y = tdf["y"].values[:OBS_LEN].astype(np.float32)
            tracks[tid] = np.stack([x, y], axis=1)

    if len(tracks) == 0:
        return None

    # 2. 场景中心 = 所有轨迹点均值（用于归一化）
    all_pts = np.concatenate(list(tracks.values()), axis=0)
    center = all_pts.mean(axis=0)

    # 3. 聚类特征：终点位置（归一化）+ 平均航向(cos, sin)
    tids = list(tracks.keys())
    feats = np.zeros((len(tids), 4), dtype=np.float32)
    for i, tid in enumerate(tids):
        tr = tracks[tid]
        dx = tr[-1, 0] - tr[0, 0]
        dy = tr[-1, 1] - tr[0, 1]
        heading = np.arctan2(dy, dx)
        feats[i, 0] = tr[-1, 0] - center[0]
        feats[i, 1] = tr[-1, 1] - center[1]
        feats[i, 2] = np.cos(heading)
        feats[i, 3] = np.sin(heading)

    n_lanes = min(NUM_LANES, len(tids))
    labels = _simple_kmeans(feats, k=n_lanes)

    # 4. 每簇：轨迹逐帧平均 → 车道中心线 → 采样 M 节点
    nodes = np.zeros((NUM_NODES, NODE_DIM), dtype=np.float32)
    adj = np.zeros((NUM_NODES, NUM_NODES), dtype=np.int8)
    node_mask = np.zeros(NUM_NODES, dtype=np.int8)

    lane_lines = []  # 每条车道 [M, 6] 节点特征
    for l in range(n_lanes):
        ids = [tids[i] for i in range(len(tids)) if labels[i] == l]
        if len(ids) == 0:
            continue

        trajs = np.stack([tracks[tid] for tid in ids], axis=0)      # [n, obs_len, 2]
        centerline = trajs.mean(axis=0)                              # [obs_len, 2]

        # 沿弧长均匀采样 M 节点（等间隔索引近似）
        sample_idx = np.linspace(0, OBS_LEN - 1, NODES_PER_LANE).astype(int)
        sampled = centerline[sample_idx]                             # [M, 2]

        lane_nodes = np.zeros((NODES_PER_LANE, NODE_DIM), dtype=np.float32)
        # 位置（归一化）
        lane_nodes[:, 0] = sampled[:, 0] - center[0]
        lane_nodes[:, 1] = sampled[:, 1] - center[1]
        # 切线方向（相邻采样点差分 → sinθ / cosθ）
        dx = np.gradient(sampled[:, 0])
        dy = np.gradient(sampled[:, 1])
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        lane_nodes[:, 2] = dy / norm  # sinθ
        lane_nodes[:, 3] = dx / norm  # cosθ
        # 曲率（切线方向角变化率，处理 ±π 环绕）
        theta = np.arctan2(lane_nodes[:, 2], lane_nodes[:, 3])
        dtheta = np.diff(theta)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        lane_nodes[1:, 4] = dtheta
        # 限速（簇平均速度）
        vx, vy = compute_velocities(sampled[:, 0], sampled[:, 1])
        lane_nodes[:, 5] = float(np.sqrt(vx ** 2 + vy ** 2).mean())

        lane_lines.append(lane_nodes)

    # 5. 填充 nodes + node_mask
    for l, ln in enumerate(lane_lines):
        start = l * NODES_PER_LANE
        nodes[start:start + NODES_PER_LANE] = ln
        node_mask[start:start + NODES_PER_LANE] = 1

    # 6. 邻接矩阵
    # 前后边：同车道相邻节点（无向）
    for l in range(len(lane_lines)):
        start = l * NODES_PER_LANE
        for i in range(NODES_PER_LANE - 1):
            adj[start + i, start + i + 1] = 1
            adj[start + i + 1, start + i] = 1

    # 左右边：不同车道对应节点欧氏距离 < 车道宽度（近似横向相邻）
    for l1 in range(len(lane_lines)):
        for l2 in range(l1 + 1, len(lane_lines)):
            for i in range(NODES_PER_LANE):
                n1 = l1 * NODES_PER_LANE + i
                for j in range(NODES_PER_LANE):
                    n2 = l2 * NODES_PER_LANE + j
                    dx = nodes[n1, 0] - nodes[n2, 0]
                    dy = nodes[n1, 1] - nodes[n2, 1]
                    if float(np.sqrt(dx ** 2 + dy ** 2)) < LANE_WIDTH:
                        adj[n1, n2] = 1
                        adj[n2, n1] = 1

    return nodes, adj, node_mask


def build_dataset(csv_items, output_path: str, limit: int = None) -> int:
    """批量构建车道线图数据集。

    参数:
        csv_items: [(split, csv_path), ...] 列表。split 为划分标签（"train"/"val"），
            用于给 scene_id 加前缀。Argoverse 的 train 与 val 目录存在同名场景文件
            （如都含 1.csv），若 scene_id 只用 csv_stem 会冲突，车道线关联错位。
        output_path: 输出 .npz 路径
        limit: 最多处理的场景数（None=全量，调试用）

    返回:
        处理的场景数
    """
    csv_items = list(csv_items)
    if limit is not None:
        csv_items = csv_items[:limit]
    print(f"  待处理 {len(csv_items)} 个 CSV 文件")

    all_nodes = []
    all_adj = []
    all_mask = []
    all_scene_ids = []
    skipped = 0

    for i, (split, csv_path) in enumerate(csv_items):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [跳过] 读取 {csv_path} 失败: {e}")
            skipped += 1
            continue

        try:
            df = _normalize_columns(df)
        except ValueError as e:
            print(f"  [跳过] 标准化 {csv_path} 失败: {e}")
            skipped += 1
            continue

        graph = build_scene_lane_graph(df)
        if graph is None:
            skipped += 1
            continue

        nodes, adj, node_mask = graph
        all_nodes.append(nodes)
        all_adj.append(adj)
        all_mask.append(node_mask)
        all_scene_ids.append(f"{split}_{csv_path.stem}")

        if (i + 1) % 1000 == 0:
            print(f"  已处理 {i+1}/{len(csv_items)} 文件，累计 {len(all_nodes)} 场景...")

    # 保存（savez_compressed 省磁盘；dataset 加载时整体 load，约 360MB 内存）
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        nodes=np.array(all_nodes, dtype=np.float32),
        adj=np.array(all_adj, dtype=np.int8),
        node_mask=np.array(all_mask, dtype=np.int8),
        scene_ids=np.array(all_scene_ids),
    )

    print(f"  完成: {len(all_nodes)} 场景 → {output_path}（跳过 {skipped}）")
    return len(all_nodes)


def main():
    parser = argparse.ArgumentParser(description="车道线图构建（轨迹聚合拟合车道中心线）")
    parser.add_argument("--raw_dir", type=str, default=None,
                        help="原始 CSV 目录（单目录模式；默认 None，走全量模式遍历 train+val）")
    parser.add_argument("--split", type=str, default="train",
                        help="单目录模式下，给 scene_id 加的前缀（train/val）")
    parser.add_argument("--train_dir", type=str, default="1/train/data",
                        help="全量模式下训练集 CSV 目录")
    parser.add_argument("--val_dir", type=str, default="1/val/data",
                        help="全量模式下验证集 CSV 目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 .npz 路径（默认 data/processed/lane_graphs.npz）")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多处理的场景数（调试用）")
    args = parser.parse_args()

    # 收集 (split, csv_path) 列表
    if args.raw_dir:
        csv_items = [
            (args.split, p) for p in sorted(Path(args.raw_dir).rglob("*.csv"))
        ]
        default_output = f"data/processed/lane_graphs_{args.split}.npz"
    else:
        csv_items = []
        for split, d in [("train", args.train_dir), ("val", args.val_dir)]:
            if not Path(d).exists():
                print(f"  [警告] 目录不存在，跳过: {d}")
                continue
            csv_items += [(split, p) for p in sorted(Path(d).rglob("*.csv"))]
        default_output = "data/processed/lane_graphs.npz"

    output_path = args.output or default_output
    n = build_dataset(csv_items, output_path, args.limit)
    print(f"\n车道线图构建完成: {n} 个场景 → {output_path}")


if __name__ == "__main__":
    main()
