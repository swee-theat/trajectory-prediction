#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二阶段消融实验：车道线融合 + 城市特征（域感知）。

三组消融（均用 branch_hidden_dim=96 压缩 predictor + 独立头，只切换车道线/城市开关，
隔离「车道线」这一个变量）：
    ① 仅轨迹          use_lane=False, use_city=False
    ② +车道线         use_lane=True,  use_city=False
    ③ +车道线+城市特征 use_lane=True,  use_city=True

用法:
    python scripts/run_ablation_stage2.py --epochs 5        # 子集/快速验证
    python scripts/run_ablation_stage2.py --epochs 80       # 全量训练
"""

import sys
import os
import json
import time
import argparse
import traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from utils.config import load_config, config_to_dict
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.param_counter import count_parameters
from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from train.trainer import Trainer
from pathlib import Path

RESULTS_PATH = "outputs/results/stage2_ablation_results.json"

# 三组消融（名称, 开关）
ABLATIONS = [
    ("仅轨迹(基线)", {"use_lane": False, "use_city": False}),
    ("+车道线", {"use_lane": True, "use_city": False}),
    ("+车道线+城市特征", {"use_lane": True, "use_city": True}),
]


def load_existing_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results):
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _find_latest_checkpoint(ckpt_dir: str):
    """在 checkpoint 目录中查找最新 epoch 的 checkpoint，用于续跑。

    返回:
        最新 checkpoint_epoch_{n}.pt 的路径字符串；无则 None
    """
    d = Path(ckpt_dir)
    if not d.exists():
        return None
    ckpts = sorted(
        d.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


def _load_prev_best(ckpt_dir: str):
    """从 best_model.pt 读取历史最佳 minADE 与 epoch，供续跑时恢复。

    续跑会重建 Trainer（best_metric / early_stopping 归零），若不恢复历史最佳，
    最终报告的 minADE 会丢失 resume 之前更优的值。故从 best_model.pt 取回。

    返回:
        (best_val_min_ade, best_epoch) 元组；无 best_model.pt 则 (None, None)
    """
    best_path = Path(ckpt_dir) / "best_model.pt"
    if not best_path.exists():
        return None, None
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})
    return metrics.get("val_min_ade"), ckpt.get("epoch", 0)


def run_one(name, overrides, epochs):
    print(f"\n{'=' * 60}")
    print(f"  第二阶段消融: {name}  ({overrides})")
    print(f"{'=' * 60}")

    cfg = config_to_dict(load_config("configs/default.yaml"))
    cfg["training"]["epochs"] = epochs
    cfg["training"]["warmup_epochs"] = max(1, epochs // 10)
    for k, v in overrides.items():
        cfg["model"][k] = v
    # 三组均用压缩 predictor + 独立头（显式写入保证一致性）
    cfg["model"]["branch_hidden_dim"] = 96
    cfg["model"]["share_uncertainty"] = False

    exp_id = name.replace("(", "").replace(")", "").replace("+", "加")
    cfg["logging"]["log_dir"] = f"outputs/logs/s2_{exp_id}"
    cfg["logging"]["checkpoint_dir"] = f"outputs/checkpoints/s2_{exp_id}"

    set_seed(42)
    logger = setup_logger(cfg["logging"]["log_dir"], exp_id)

    train_ds = ArgoverseTrajectoryDataset(
        "data/processed/train.npz",
        lane_graphs_path="data/processed/lane_graphs.npz",
    )
    val_ds = ArgoverseTrajectoryDataset(
        "data/processed/val.npz",
        lane_graphs_path="data/processed/lane_graphs.npz",
    )
    # num_workers=1：Windows 下内存约束（16.9GB，train.npz 解压约 1.8GB/worker）
    # 2+ 会因每 worker 复制 dataset 而 OOM；1 已能显著缓解 CPU 数据加载瓶颈
    train_loader = create_dataloader(train_ds, batch_size=64, shuffle=True, num_workers=1)
    val_loader = create_dataloader(val_ds, batch_size=64, shuffle=False, num_workers=1)

    mc = cfg["model"]
    model = TrajectoryModel(
        input_dim=mc.get("input_dim", 11),
        hidden_dim=mc.get("hidden_dim", 128),
        num_mlp_layers=mc.get("num_mlp_layers", 3),
        dropout=mc.get("dropout", 0.1),
        num_modes=mc.get("num_modes", 5),
        pred_len=cfg["data"].get("pred_len", 30),
        pooling_type=mc.get("pooling", "attention"),
        use_uncertainty=mc.get("use_uncertainty", True),
        use_temporal_conv=mc.get("use_temporal_conv", True),
        share_uncertainty=mc.get("share_uncertainty", False),
        use_lane=mc.get("use_lane", True),
        use_city=mc.get("use_city", True),
        branch_hidden_dim=mc.get("branch_hidden_dim", 96),
    )
    total_params, _ = count_parameters(model)
    logger.info(f"参数量: {total_params:,}")

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=cfg, logger=logger, tb_writer=None,
    )

    # 续跑：检测已有 checkpoint，从最新 epoch 恢复训练，并恢复历史最佳指标
    resume_from = _find_latest_checkpoint(cfg["logging"]["checkpoint_dir"])
    if resume_from:
        prev_best_ade, prev_best_epoch = _load_prev_best(cfg["logging"]["checkpoint_dir"])
        if prev_best_ade is not None:
            trainer.best_metric = prev_best_ade
            trainer.early_stopping.best_score = prev_best_ade
            trainer.early_stopping.best_epoch = prev_best_epoch
        logger.info(
            f"检测到 checkpoint，从 epoch {Path(resume_from).stem.rsplit('_', 1)[-1]} 续跑"
        )

    result = trainer.train(resume_from=resume_from)

    return {
        "name": name,
        "use_lane": overrides["use_lane"],
        "use_city": overrides["use_city"],
        "params": total_params,
        "best_val_min_ade": result["best_val_min_ade"],
        "best_epoch": result["best_epoch"] + 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练 epoch 数（子集验证用 5，全量用 80）")
    args = parser.parse_args()

    t0 = time.time()
    results = load_existing_results()
    done_names = {r["name"] for r in results}
    print(f"已有 {len(results)} 个结果: {done_names}")

    for name, overrides in ABLATIONS:
        if name in done_names:
            print(f"跳过已完成: {name}")
            continue
        try:
            r = run_one(name, overrides, args.epochs)
            results.append(r)
            save_results(results)
            elapsed = (time.time() - t0) / 60
            print(f"  => {name}: minADE={r['best_val_min_ade']:.4f}, "
                  f"params={r['params']:,} | 耗时 {elapsed:.0f}分钟")
        except Exception as e:
            print(f"  => {name}: 失败 - {e}")
            traceback.print_exc()

    elapsed = (time.time() - t0) / 60
    print(f"\n第二阶段消融完成! 耗时 {elapsed:.0f}分钟")
    for r in results:
        print(f"  {r['name']}: minADE={r['best_val_min_ade']:.4f}, "
              f"params={r['params']:,}")


if __name__ == "__main__":
    main()
