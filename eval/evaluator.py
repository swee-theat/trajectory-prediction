"""评估器：在测试集上全量评估，按场景类型分类统计，导出 CSV。"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import pandas as pd

from .metrics import compute_min_ade, compute_min_fde, compute_ade, compute_fde


CATEGORY_NAMES = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口/停车"}


class Evaluator:
    """全量测试集评估器。"""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """在测试集上全量评估。

        返回:
            {
                "overall": {指标名: 均值},
                "per_category": {类别名: {指标名: 均值}},
                "per_sample": [逐样本指标字典],
            }
        """
        self.model.eval()

        all_metrics = []
        per_category_metrics = {name: [] for name in CATEGORY_NAMES.values()}
        overall_metrics = []

        pbar = tqdm(self.test_loader, desc="评估中")

        for batch in pbar:
            history = batch["history"].to(self.device)
            future = batch["future"].to(self.device)
            categories = batch["category"]  # [B]
            seq_ids = batch["seq_id"]

            output = self.model(history)
            trajectories = output["trajectories"]

            # 批量计算逐样本指标（一次前向，同时算所有样本）
            min_ade_vals = compute_min_ade(trajectories, future)  # [B]
            min_fde_vals = compute_min_fde(trajectories, future)  # [B]
            best_idx = output["mode_probs"].argmax(dim=-1)        # [B]
            ade_vals = compute_ade(trajectories, future, best_idx)  # [B]
            fde_vals = compute_fde(trajectories, future, best_idx)  # [B]
            B = trajectories.shape[0]

            for i in range(B):
                cat = int(categories[i].item())
                cat_name = CATEGORY_NAMES.get(cat, "未知")
                metrics = {
                    "ade": ade_vals[i].item(),
                    "fde": fde_vals[i].item(),
                    "min_ade": min_ade_vals[i].item(),
                    "min_fde": min_fde_vals[i].item(),
                    "category": cat,
                    "category_name": cat_name,
                    "seq_id": seq_ids[i],
                }
                all_metrics.append(metrics)
                per_category_metrics[cat_name].append(metrics)

            pbar.set_postfix({"样本数": len(all_metrics)})

        # 汇总整体指标
        overall = self._aggregate(all_metrics)

        # 汇总各类别指标
        per_cat_agg = {}
        for cat_name, cat_metrics in per_category_metrics.items():
            if len(cat_metrics) > 0:
                per_cat_agg[cat_name] = self._aggregate(cat_metrics)
                per_cat_agg[cat_name]["样本数"] = len(cat_metrics)
            else:
                per_cat_agg[cat_name] = {"样本数": 0}

        return {
            "overall": overall,
            "per_category": per_cat_agg,
            "per_sample": all_metrics,
        }

    def _aggregate(self, metrics_list: List[Dict]) -> Dict:
        """聚合指标列表（取均值）。"""
        if not metrics_list:
            return {}

        keys = ["ade", "fde", "min_ade", "min_fde"]  # 逐样本指标不含 miss_rate/endpoint_hit
        agg = {}
        for key in keys:
            values = [m.get(key, 0.0) for m in metrics_list if key in m]
            if values:
                agg[key] = sum(values) / len(values)
        agg["样本数"] = len(metrics_list)
        return agg

    def export_csv(self, result: Dict, output_dir: str) -> None:
        """将评估结果导出为 CSV 文件。

        参数:
            result: evaluate() 返回的结果字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 表1: 整体指标（单行）
        overall_df = pd.DataFrame([result["overall"]])
        overall_df.to_csv(output_path / "overall_metrics.csv", index=False, encoding="utf-8-sig")

        # 表2: 各类别指标
        per_cat_df = pd.DataFrame(result["per_category"]).T
        per_cat_df.to_csv(output_path / "per_category_metrics.csv", encoding="utf-8-sig")

        # 逐样本指标
        sample_df = pd.DataFrame(result["per_sample"])
        sample_df.to_csv(output_path / "per_sample_metrics.csv", index=False, encoding="utf-8-sig")

        print(f"  评估结果已导出到: {output_path}")
        print(f"    整体: {overall_df.to_dict('records')[0]}")
