"""数据预处理模块：将 Argoverse CSV 原始数据转换为 11 维特征。

特征设计（11 维）:
    [0] x:               居中的 X 坐标
    [1] y:               居中的 Y 坐标
    [2] vx:              X 方向速度
    [3] vy:              Y 方向速度
    [4] t/T:             归一化时间步
    [5] 1:               常数偏置项
    [6] dist_nearest/100: 到最近智能体的归一化距离
    [7] rel_x/100:        相对最近智能体的 X 位移
    [8] rel_y/100:        相对最近智能体的 Y 位移
    [9] heading:          航向角（rad，unwrap 后）
    [10] curvature:       曲率（rad/s，裁剪 ±3.0）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


# 预期列名映射（兼容大小写和不同命名风格）
COLUMN_MAP = {
    "timestamp": ["TIMESTAMP", "timestamp", "time", "TIME"],
    "track_id": ["TRACK_ID", "track_id", "id", "ID", "agent_id"],
    "object_type": ["OBJECT_TYPE", "object_type", "type", "TYPE"],
    "x": ["X", "x", "pos_x", "POS_X"],
    "y": ["Y", "y", "pos_y", "POS_Y"],
    "city": ["CITY_NAME", "city", "CITY"],
}


# 城市标签 → 整数 ID（第二阶段"简化城市特征/域感知"用；未知城市归 UNK=0）
CITY_TO_ID = {
    "UNK": 0,
    "ATL": 1,
    "MIA": 2,
    "PIT": 3,
    "PAO": 4,
    "AUS": 5,
    "WDC": 6,
}


def city_to_id(city: str) -> int:
    """城市字符串 → 整数 ID（0-6，未知归 0）。"""
    return CITY_TO_ID.get(str(city).upper(), 0)


def _detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """自动检测 CSV 列名映射。"""
    cols = {c.upper(): c for c in df.columns}
    mapping = {}
    for target, candidates in COLUMN_MAP.items():
        for cand in candidates:
            if cand.upper() in cols:
                mapping[target] = cols[cand.upper()]
                break
    return mapping


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 CSV DataFrame 标准化为统一列名。"""
    detected = _detect_columns(df)
    required = ["timestamp", "track_id", "object_type", "x", "y"]
    missing = [k for k in required if k not in detected]
    if missing:
        raise ValueError(
            f"CSV 缺少必要列: {missing}。已检测到列: {list(df.columns)}"
        )

    normalized = pd.DataFrame()
    normalized["timestamp"] = df[detected["timestamp"]]
    normalized["track_id"] = df[detected["track_id"]]
    normalized["object_type"] = df[detected["object_type"]]
    normalized["x"] = df[detected["x"]].astype(np.float32)
    normalized["y"] = df[detected["y"]].astype(np.float32)

    # 可选保留城市列（第二阶段"简化城市特征/域感知"用），缺失填 "UNK"
    normalized["city"] = df[detected["city"]] if "city" in detected else "UNK"

    return normalized


def compute_velocities(
    x: np.ndarray, y: np.ndarray, dt: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """计算前向差分速度。

    参数:
        x: X 坐标序列 [T]
        y: Y 坐标序列 [T]
        dt: 时间步长（秒），Argoverse 默认为 0.1s

    返回:
        (vx, vy) 元组，各为 [T]
    """
    vx = np.zeros_like(x)
    vy = np.zeros_like(y)

    # 前向差分
    vx[1:] = (x[1:] - x[:-1]) / dt
    vy[1:] = (y[1:] - y[:-1]) / dt

    # t=0 使用 t=1 的速度（避免零点问题）
    if len(vx) > 1:
        vx[0] = vx[1]
        vy[0] = vy[1]

    return vx, vy


def _build_agents_abs(df: pd.DataFrame, total_len: int) -> Tuple[Dict, Dict]:
    """一次性预构建所有长度足够智能体的绝对坐标轨迹与类型（性能优化）。

    原实现对每个 track_id 重复 mask + sort（O(N²) 次 pandas 操作），是预处理
    的耗时瓶颈。这里改为一次排序 + 遍历，将排序次数从 O(N²) 降到 O(1)。

    返回:
        (agents, agents_type)
        agents:       {track_id: np.ndarray [total_len, 2] 绝对坐标 float32}
        agents_type:  {track_id: str} 智能体类型（'AV'/'AGENT'/'OTHERS'，取众数防类型漂移）
    """
    agents = {}
    agents_type = {}
    df_sorted = df.sort_values(["track_id", "timestamp"], kind="mergesort")
    for tid, group in df_sorted.groupby("track_id", sort=False):
        if len(group) >= total_len:
            agents[tid] = group[["x", "y"]].to_numpy(dtype=np.float32)[:total_len]
            agents_type[tid] = str(group["object_type"].mode()[0])
    return agents, agents_type


def compute_nearest_agent_features(
    x: np.ndarray,
    y: np.ndarray,
    all_agents: Dict[int, Tuple[np.ndarray, np.ndarray]],
    current_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算当前智能体相对于场景中最近其他智能体的特征（向量化实现）。

    参数:
        x, y: 当前智能体的坐标 [T]
        all_agents: {track_id: (x_arr[L], y_arr[L])} 场景中所有智能体（L >= T）
        current_id: 当前智能体的 track_id

    返回:
        (dist_nearest[T], rel_x[T], rel_y[T])
    """
    T = len(x)
    dist_nearest = np.full(T, 100.0, dtype=np.float32)  # 默认远距离
    rel_x = np.zeros(T, dtype=np.float32)
    rel_y = np.zeros(T, dtype=np.float32)

    if len(all_agents) <= 1:
        return dist_nearest, rel_x, rel_y

    # 收集其他智能体的前 T 帧（截断到 T 保证形状一致；真实数据 L=total_len>=T）
    others = [
        np.stack([ax[:T], ay[:T]], axis=-1)
        for agent_id, (ax, ay) in all_agents.items()
        if agent_id != current_id
    ]
    if not others:
        return dist_nearest, rel_x, rel_y

    others = np.stack(others).astype(np.float32)        # [N, T, 2]
    cur = np.stack([x, y], axis=-1).astype(np.float32)  # [T, 2]

    # 广播计算：diff[n, t] = cur[t] - other[n, t]（等价于原逐元素 dx=x[t]-ax[t]）
    diff = cur[None, :, :] - others                     # [N, T, 2]
    dist = np.sqrt((diff * diff).sum(axis=-1))          # [N, T]
    min_idx = dist.argmin(axis=0)                       # [T]
    rows = np.arange(T)
    dist_nearest = dist[min_idx, rows].astype(np.float32)
    rel = diff[min_idx, rows, :]                        # [T, 2]
    rel_x = rel[:, 0].astype(np.float32)
    rel_y = rel[:, 1].astype(np.float32)
    return dist_nearest, rel_x, rel_y


def build_neighbor_history(
    centered_traj: np.ndarray, obs_len: int
) -> np.ndarray:
    """构造邻居车辆的 11 维观测特征（居中到 ego 末帧原点）。

    与 ego 的 history 特征同构（便于复用 TrajectoryEncoder 权重共享编码），
    但 [6,7,8]（最近智能体标量特征）置 0——邻居的"最近智能体"通常就是 ego，
    与 cross-attention 直接建模的 ego↔neighbor 关系冗余。

    参数:
        centered_traj: [total_len, 2] 已居中到 ego 末帧原点的绝对坐标
        obs_len: 观测帧数

    返回:
        np.ndarray [obs_len, 11]
    """
    x = centered_traj[:, 0]
    y = centered_traj[:, 1]
    vx, vy = compute_velocities(x, y)
    t_norm = np.arange(len(x), dtype=np.float32) / (obs_len - 1)
    heading = np.unwrap(np.arctan2(vy, vx))
    curvature = np.zeros_like(heading)
    if len(heading) > 1:
        curvature[1:] = (heading[1:] - heading[:-1]) / 0.1
        curvature[0] = curvature[1]
    curvature = np.clip(curvature, -3.0, 3.0)
    z = np.zeros(obs_len, dtype=np.float32)
    return np.stack([
        x[:obs_len],                              # 0: x
        y[:obs_len],                              # 1: y
        vx[:obs_len],                             # 2: vx
        vy[:obs_len],                             # 3: vy
        t_norm[:obs_len],                         # 4: t/T
        np.ones(obs_len, dtype=np.float32),       # 5: 常值偏置 1
        z,                                        # 6: dist_nearest/100（置 0）
        z,                                        # 7: rel_x/100（置 0）
        z,                                        # 8: rel_y/100（置 0）
        heading[:obs_len],                        # 9: 航向角（rad）
        curvature[:obs_len],                      # 10: 曲率（rad/s）
    ], axis=-1)  # [obs_len, 11]


def select_interaction_neighbors(
    agents_abs: Dict,
    agents_type: Dict,
    track_id,
    x_center: float,
    y_center: float,
    obs_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """筛选目标车的最近 2 辆 OTHERS 邻居，构建交互输入（第三阶段）。

    邻居只取 OBJECT_TYPE == 'OTHERS' 且 tid != current_id 的车辆，按「末观测帧
    (t=obs_len-1) 的绝对坐标距离」排序取前 2。所有邻居坐标居中到 ego 末帧原点，
    与 ego history/future 同坐标系（否则碰撞检测无意义）。

    返回:
        (neighbor_histories, neighbor_futures, neighbor_mask)
        neighbor_histories: [2, obs_len, 11] 前 2 近 OTHERS 观测（居中 ego 原点）
        neighbor_futures:   [pred_len, 2]    最近 1 辆 OTHERS 的未来真值（同坐标系）
        neighbor_mask:      [2] float32      slot 有效性（1=有邻居，0=填充）
    """
    candidates = []
    for tid, coords in agents_abs.items():
        if tid == track_id:
            continue
        if agents_type.get(tid) != "OTHERS":
            continue
        # 末观测帧（t=obs_len-1）的绝对坐标距离
        d = np.hypot(coords[obs_len - 1, 0] - x_center,
                     coords[obs_len - 1, 1] - y_center)
        candidates.append((d, coords))
    candidates.sort(key=lambda t: t[0])

    neighbor_histories = np.zeros((2, obs_len, 11), dtype=np.float32)
    neighbor_mask = np.zeros(2, dtype=np.float32)
    center = np.array([x_center, y_center], dtype=np.float32)
    for i, (_, coords) in enumerate(candidates[:2]):
        centered = coords - center                     # [total_len, 2] 居中到 ego 原点
        neighbor_histories[i] = build_neighbor_history(centered, obs_len)
        neighbor_mask[i] = 1.0

    neighbor_futures = np.zeros((pred_len, 2), dtype=np.float32)
    if candidates:
        nearest = candidates[0][1]                     # 最近 OTHERS 绝对坐标
        neighbor_futures = nearest[obs_len:] - center  # [pred_len, 2] 未来真值居中

    return neighbor_histories, neighbor_futures, neighbor_mask


def extract_agent_track(
    df: pd.DataFrame,
    track_id: int,
    obs_len: int = 20,
    pred_len: int = 30,
    agents_abs: Optional[Dict] = None,
    agents_types: Optional[Dict] = None,
) -> Optional[Dict]:
    """从单个场景 CSV 中提取一个智能体的完整轨迹。

    参数:
        df: 标准化后的单个 CSV DataFrame
        track_id: 目标智能体 ID
        obs_len: 观测帧数（默认 20）
        pred_len: 预测帧数（默认 30）
        agents_abs: 预构建的所有智能体绝对坐标轨迹 {track_id: [total_len, 2]}，
            由 process_csv_file 传入以避免每个 agent 重复 sort。为 None 时走旧逻辑。
        agents_types: 预构建的智能体类型 {track_id: str}（'AV'/'AGENT'/'OTHERS'），
            用于第三阶段筛选交互邻居（只取 OTHERS）。为 None 时邻居全零填充。

    返回:
        {
            "history": np.ndarray [obs_len, 11],
            "future": np.ndarray [pred_len, 2],
            "category": int (场景类型标签),
            "seq_id": str,
            "neighbor_histories": np.ndarray [2, obs_len, 11],
            "neighbor_futures": np.ndarray [pred_len, 2],
            "neighbor_mask": np.ndarray [2],
        }
        如果轨迹长度不足则返回 None
    """
    total_len = obs_len + pred_len

    # 当前智能体轨迹（绝对坐标 [total_len, 2]）
    if agents_abs is not None:
        traj = agents_abs[track_id]
    else:
        mask = df["track_id"] == track_id
        agent_df = df[mask].sort_values("timestamp")
        if len(agent_df) < total_len:
            return None
        agent_df = agent_df.iloc[:total_len]
        traj = np.stack([
            agent_df["x"].values.astype(np.float32),
            agent_df["y"].values.astype(np.float32),
        ], axis=-1)

    x = traj[:, 0]
    y = traj[:, 1]

    # 居中：以最后观测帧为原点
    x_center = x[obs_len - 1]
    y_center = y[obs_len - 1]
    x_centered = x - x_center
    y_centered = y - y_center

    # 交互邻居（第三阶段）：邻居只取 OTHERS，居中到 ego 末帧原点
    if agents_abs is not None:
        neighbor_histories, neighbor_futures, neighbor_mask = select_interaction_neighbors(
            agents_abs, agents_types or {}, track_id,
            x_center, y_center, obs_len, pred_len,
        )
    else:
        neighbor_histories = np.zeros((2, obs_len, 11), dtype=np.float32)
        neighbor_futures = np.zeros((pred_len, 2), dtype=np.float32)
        neighbor_mask = np.zeros(2, dtype=np.float32)

    # 计算速度
    vx, vy = compute_velocities(x_centered, y_centered)

    # 收集场景中所有智能体（相对当前智能体中心，用于最近距离计算）
    if agents_abs is not None:
        all_agents = {
            tid: (ttraj[:, 0] - x_center, ttraj[:, 1] - y_center)
            for tid, ttraj in agents_abs.items()
        }
    else:
        all_agents = {}
        for tid in df["track_id"].unique():
            tid_df = df[df["track_id"] == tid].sort_values("timestamp")
            if len(tid_df) >= total_len:
                tid_df = tid_df.iloc[:total_len]
                tx = tid_df["x"].values.astype(np.float32) - x_center
                ty = tid_df["y"].values.astype(np.float32) - y_center
                all_agents[tid] = (tx, ty)

    # 最近智能体特征
    dist_nearest, rel_x, rel_y = compute_nearest_agent_features(
        x_centered, y_centered, all_agents, track_id
    )

    # 时间步归一化
    t_normalized = np.arange(total_len, dtype=np.float32) / (obs_len - 1)

    # 航向角（瞬时运动方向），展开消除 ±π 跳变
    heading = np.unwrap(np.arctan2(vy, vx))  # [total_len]

    # 曲率（航向角变化率，rad/s），裁剪到合理车辆动力学范围
    curvature = np.zeros_like(heading)
    if total_len > 1:
        curvature[1:] = (heading[1:] - heading[:-1]) / 0.1
        curvature[0] = curvature[1]  # 第一帧复制第二帧
    curvature = np.clip(curvature, -3.0, 3.0)  # 车辆极限曲率 ±3 rad/s

    # 构建 11 维特征
    history_feats = np.stack([
        x_centered[:obs_len],                    # 0: x
        y_centered[:obs_len],                    # 1: y
        vx[:obs_len],                            # 2: vx
        vy[:obs_len],                            # 3: vy
        t_normalized[:obs_len],                  # 4: t/T
        np.ones(obs_len, dtype=np.float32),      # 5: 常值偏置 1
        dist_nearest[:obs_len] / 100.0,          # 6: dist/100
        rel_x[:obs_len] / 100.0,                 # 7: rel_x/100
        rel_y[:obs_len] / 100.0,                 # 8: rel_y/100
        heading[:obs_len],                       # 9: 航向角（rad）
        curvature[:obs_len],                     # 10: 曲率（rad/s）
    ], axis=-1)  # [obs_len, 11]

    # 未来轨迹（仅 x,y，已经居中）
    future = np.stack([
        x_centered[obs_len:],
        y_centered[obs_len:],
    ], axis=-1)  # [pred_len, 2]

    # 场景类型分类（基于未来轨迹曲率）
    category = classify_scene_type(future)

    # 序列 ID
    seq_id = f"{track_id}"

    return {
        "history": history_feats,
        "future": future,
        "category": category,
        "seq_id": seq_id,
        "neighbor_histories": neighbor_histories,
        "neighbor_futures": neighbor_futures,
        "neighbor_mask": neighbor_mask,
    }


def classify_scene_type(future_traj: np.ndarray) -> int:
    """根据未来轨迹曲率和速度对场景类型进行分类。

    类别:
        0: 直行
        1: 左转
        2: 右转
        3: 掉头
        4: 路口/停车（低速/静止样本）

    参数:
        future_traj: 未来轨迹 [T_pred, 2]，已居中

    返回:
        场景类型标签 (0-4)
    """
    dx = np.diff(future_traj[:, 0])
    dy = np.diff(future_traj[:, 1])

    # 低速/停车前置判断（wen_ti/02）：先于航向曲率，避免停车样本位移≈0 导致
    # arctan2 噪声随机游走被误判为掉头/左右转。低于阈值归「路口/停车」(4)。
    velocities = np.sqrt(dx**2 + dy**2)
    avg_speed = np.mean(velocities)
    LOW_SPEED_THRESH = 0.05  # m/帧（10Hz 下 0.5 m/s）
    if avg_speed < LOW_SPEED_THRESH:
        return 4  # 路口/停车

    # 计算每帧的航向角变化
    headings = np.arctan2(dy, dx)
    heading_changes = np.diff(headings)

    # 处理角度环绕（-pi 到 pi）
    heading_changes = np.arctan2(
        np.sin(heading_changes), np.cos(heading_changes)
    )

    total_curvature = np.sum(heading_changes)

    # 分类阈值
    U_TURN_THRESH = np.deg2rad(150)  # 总曲率 > 150°
    TURN_THRESH = np.deg2rad(30)     # 总曲率 > 30°

    if abs(total_curvature) > U_TURN_THRESH:
        return 3  # 掉头
    elif total_curvature > TURN_THRESH:
        return 1  # 左转
    elif total_curvature < -TURN_THRESH:
        return 2  # 右转

    # 路口判定：速度骤降到峰值速度的 30% 以下
    max_v = np.max(velocities)
    is_slow = max_v > 0 and avg_speed < 0.3 * max_v
    if is_slow:
        return 4  # 路口/交叉

    return 0  # 直行


def process_csv_file(
    csv_path: str,
    obs_len: int = 20,
    pred_len: int = 30,
    split: str = "",
) -> List[Dict]:
    """处理单个 Argoverse CSV 文件，提取所有有效智能体轨迹。

    参数:
        csv_path: CSV 文件路径
        obs_len: 观测帧数
        pred_len: 预测帧数
        split: 数据集划分标签（如 "train"/"val"）。Argoverse 的 train 与 val 目录
            存在同名场景文件（如都含 1.csv），seq_id 必须带 split 前缀才能全局唯一，
            否则车道线图（按 scene_id 关联）会错位。空字符串时保持旧格式向后兼容。

    返回:
        样本列表，每个样本为 {"history", "future", "category", "seq_id"} 字典
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        warnings.warn(f"读取 {csv_path} 失败: {e}")
        return []

    try:
        df = _normalize_columns(df)
    except ValueError as e:
        warnings.warn(f"标准化 {csv_path} 失败: {e}")
        return []

    # 城市标签（场景级，取首行；缺失则 UNK）
    city_val = str(df["city"].iloc[0]) if "city" in df.columns else "UNK"

    # 预构建所有长度足够 agent 的绝对坐标轨迹与类型，避免每个 agent 重复 sort（O(N²)→O(N)）
    agents_abs, agents_types = _build_agents_abs(df, obs_len + pred_len)

    samples = []
    for track_id in agents_abs:
        # 第三阶段：目标车范围 AV + AGENT（OTHERS 只作交互邻居，不当预测目标）
        if agents_types.get(track_id) not in ("AV", "AGENT"):
            continue
        sample = extract_agent_track(
            df, track_id, obs_len, pred_len,
            agents_abs=agents_abs, agents_types=agents_types,
        )
        if sample is not None:
            sample["seq_id"] = (
                f"{split}_{Path(csv_path).stem}_{track_id}"
                if split else f"{Path(csv_path).stem}_{track_id}"
            )
            sample["city"] = city_val
            samples.append(sample)

    return samples


def process_dataset(
    raw_dir: str,
    output_path: str,
    obs_len: int = 20,
    pred_len: int = 30,
    split: str = "train",
) -> int:
    """批量处理数据集目录下所有 CSV 文件，保存为压缩 numpy 格式。

    参数:
        raw_dir: 原始 CSV 目录
        output_path: 输出 .npz 文件路径
        obs_len: 观测帧数
        pred_len: 预测帧数
        split: 数据集划分名称

    返回:
        处理成功的样本总数
    """
    csv_dir = Path(raw_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {raw_dir}")

    csv_files = sorted(csv_dir.rglob("*.csv"))
    print(f"  找到 {len(csv_files)} 个 CSV 文件在 {raw_dir}")

    all_histories = []
    all_futures = []
    all_categories = []
    all_seq_ids = []
    all_cities = []
    all_neighbor_histories = []
    all_neighbor_futures = []
    all_neighbor_masks = []
    total_samples = 0

    for i, csv_path in enumerate(csv_files):
        samples = process_csv_file(str(csv_path), obs_len, pred_len)
        for s in samples:
            all_histories.append(s["history"])
            all_futures.append(s["future"])
            all_categories.append(s["category"])
            all_seq_ids.append(s["seq_id"])
            all_cities.append(city_to_id(s["city"]))
            all_neighbor_histories.append(s["neighbor_histories"])
            all_neighbor_futures.append(s["neighbor_futures"])
            all_neighbor_masks.append(s["neighbor_mask"])
            total_samples += 1

        if (i + 1) % 1000 == 0:
            print(f"  已处理 {i+1}/{len(csv_files)} 文件, 累计 {total_samples} 样本...")

    # 保存为压缩 numpy 文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        histories=np.array(all_histories, dtype=np.float32),
        futures=np.array(all_futures, dtype=np.float32),
        categories=np.array(all_categories, dtype=np.int32),
        seq_ids=np.array(all_seq_ids),
        cities=np.array(all_cities, dtype=np.int32),
        neighbor_histories=np.array(all_neighbor_histories, dtype=np.float32),
        neighbor_futures=np.array(all_neighbor_futures, dtype=np.float32),
        neighbor_mask=np.array(all_neighbor_masks, dtype=np.float32),
    )

    print(f"  {split} 集处理完成: {total_samples} 样本 → {output_path}")
    return total_samples
