"""PyTorch Dataset 类：加载预处理好的轨迹数据。"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional


# 车道线图常量（与 scripts/build_lane_graph.py 保持一致）
LANE_NUM_NODES = 32   # 4 车道 × 8 节点
LANE_NODE_DIM = 6     # [x, y, sinθ, cosθ, κ, v_limit]


class ArgoverseTrajectoryDataset(Dataset):
    """Argoverse 轨迹预测数据集。

    每个样本:
        history:  [T_obs, 11] 历史轨迹特征 (动态维度，取决于 preprocessing 输出)
        future:   [T_pred, 2]  未来轨迹真值 (x, y)
        category: int          场景类型 (0-4)
        seq_id:   str          序列标识（用于调试）
    """

    CATEGORY_NAMES = {
        0: "直行",
        1: "左转",
        2: "右转",
        3: "掉头",
        4: "路口/停车",
    }

    def __init__(self, npz_path: str, lane_graphs_path: Optional[str] = None):
        """从预处理好的 .npz 文件加载数据集。

        参数:
            npz_path: .npz 文件路径，包含 histories, futures, categories, seq_ids, cities
            lane_graphs_path: 车道线图 .npz 路径（第二阶段；None 则不加载车道线）
        """
        data = np.load(npz_path, allow_pickle=True)

        self.histories = data["histories"]      # [N, T_obs, input_dim]
        self.futures = data["futures"]            # [N, T_pred, 2]
        self.categories = data["categories"]      # [N]
        self.seq_ids = data["seq_ids"]            # [N]
        # 城市 ID（第二阶段域感知；旧 npz 无 cities 时填 0）
        self.cities = (
            data["cities"] if "cities" in data
            else np.zeros(len(self.histories), dtype=np.int32)
        )

        # 交互邻居（第三阶段；旧 npz 无邻居键时回退全 0，保证向后兼容）
        n_samples = len(self.histories)
        self.neighbor_histories = (
            data["neighbor_histories"] if "neighbor_histories" in data
            else np.zeros(
                (n_samples, 2, self.histories.shape[1], self.histories.shape[2]),
                dtype=np.float32,
            )
        )
        self.neighbor_futures = (
            data["neighbor_futures"] if "neighbor_futures" in data
            else np.zeros((n_samples, self.futures.shape[1], 2), dtype=np.float32)
        )
        self.neighbor_mask = (
            data["neighbor_mask"] if "neighbor_mask" in data
            else np.zeros((n_samples, 2), dtype=np.float32)
        )

        # 车道线图（可选加载）
        self.lane_nodes = None   # [N_scene, 32, 6]
        self.lane_adj = None     # [N_scene, 32, 32]
        self.lane_mask = None    # [N_scene, 32]
        self._scene_to_idx: Dict[str, int] = {}
        if lane_graphs_path is not None and Path(lane_graphs_path).exists():
            lg = np.load(lane_graphs_path, allow_pickle=True)
            self.lane_nodes = lg["nodes"]
            self.lane_adj = lg["adj"]
            self.lane_mask = lg["node_mask"]
            self._scene_to_idx = {
                str(sid): i for i, sid in enumerate(lg["scene_ids"])
            }

        # 验证数据形状
        assert self.histories.shape[1] >= 2, f"历史帧数异常: {self.histories.shape}"
        assert self.futures.shape[1] >= 2, f"未来帧数异常: {self.futures.shape}"
        assert len(self.histories) == len(self.futures), "历史和未来样本数不一致"

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        history = torch.from_numpy(self.histories[idx].copy()).float()
        future = torch.from_numpy(self.futures[idx].copy()).float()
        category = int(self.categories[idx])
        seq_id = str(self.seq_ids[idx])
        city_id = int(self.cities[idx])

        # 交互邻居（第三阶段）
        neighbor_history = torch.from_numpy(self.neighbor_histories[idx].copy()).float()
        neighbor_future = torch.from_numpy(self.neighbor_futures[idx].copy()).float()
        neighbor_mask = torch.from_numpy(self.neighbor_mask[idx].copy()).float()

        # 车道线图（按 seq_id 解析 csv_stem 查图，查不到返回全 0 图）
        lane_nodes = torch.zeros(LANE_NUM_NODES, LANE_NODE_DIM)
        lane_adj = torch.zeros(LANE_NUM_NODES, LANE_NUM_NODES, dtype=torch.long)
        lane_mask = torch.zeros(LANE_NUM_NODES, dtype=torch.long)
        if self.lane_nodes is not None:
            scene = seq_id.rsplit("_", 1)[0]  # {csv_stem}_{track_id} → csv_stem
            gi = self._scene_to_idx.get(scene)
            if gi is not None:
                lane_nodes = torch.from_numpy(self.lane_nodes[gi].copy()).float()
                lane_adj = torch.from_numpy(self.lane_adj[gi].copy()).long()
                lane_mask = torch.from_numpy(self.lane_mask[gi].copy()).long()

        return {
            "history": history,                     # [T_obs, input_dim]
            "future": future,                       # [T_pred, 2]
            "category": category,                   # int
            "seq_id": seq_id,                       # str
            "city_id": city_id,                     # int
            "lane_nodes": lane_nodes,               # [32, 6]
            "lane_adj": lane_adj,                   # [32, 32]
            "lane_mask": lane_mask,                 # [32]
            "neighbor_history": neighbor_history,   # [2, T_obs, 11]
            "neighbor_future": neighbor_future,     # [T_pred, 2]
            "neighbor_mask": neighbor_mask,         # [2]
        }

    @property
    def obs_len(self) -> int:
        return self.histories.shape[1]

    @property
    def pred_len(self) -> int:
        return self.futures.shape[1]

    @property
    def input_dim(self) -> int:
        return self.histories.shape[2]

    def get_category_stats(self) -> Dict[str, int]:
        """返回每个场景类型的样本数量统计。"""
        stats = {}
        for label, name in self.CATEGORY_NAMES.items():
            count = int(np.sum(self.categories == label))
            stats[name] = count
        return stats
