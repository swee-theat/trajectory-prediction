#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两阶段不确定性训练脚本（支持跳过阶段 A 直接微调）。

阶段 A: lambda_unc=0.0  正常训练至收敛（轨迹预测 + 模态选择）
阶段 B: lambda_unc=0.05 冻结编码器+预测器，仅微调不确定性头

支持两种运行模式：
  1. 完整两阶段（SKIP_STAGE_A=False）：
     Stage A 从头训练 30 epoch → Stage B 微调不确定性头
  2. 仅微调（SKIP_STAGE_A=True，推荐）：
     直接从 START_CKPT（迁移后的独立头 checkpoint）加载，
     冻结编码器+预测器，仅微调不确定性头，
     让 5 个独立头在 NLL 损失下分化出各模态不同的方差。

用法:
    python scripts/train_with_uncertainty.py

输出:
    outputs/checkpoints/unc_stage_a/best_model.pt    # 阶段 A 最佳模型（完整模式）
    outputs/checkpoints/unc_stage_b/best_model.pt    # 阶段 B 最佳模型（不确定性已校准）
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from utils.config import load_config, config_to_dict
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.param_counter import count_parameters
from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from train.trainer import Trainer

# ── 运行模式开关 ──────────────────────────────────────
SKIP_STAGE_A = True   # True=跳过阶段A，直接从迁移后 checkpoint 微调不确定性头
START_CKPT = "outputs/checkpoints/s3_组B交互_scheme2c/best_model.pt"  # 阶段B起始权重（第三阶段定稿模型）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable(model):
    """统计可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_except_uncertainty(model):
    """冻结除 UncertaintyEstimator 外的所有参数。

    同时打印可训练参数名清单并断言，用于核查冻结名单无遗漏：
    可训练参数必须全部为 uncertainty.* 前缀（context_gate / city_embed / neighbor_* / interaction.* 都在冻结名单）。
    """
    frozen, trainable = 0, 0
    trainable_names = []
    for name, param in model.named_parameters():
        if name.startswith("uncertainty"):
            param.requires_grad = True
            trainable += param.numel()
            trainable_names.append(name)
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"  冻结: {frozen:,}  可训练(仅不确定性头): {trainable:,}")
    print(f"  可训练参数名: {trainable_names}")
    assert all(n.startswith("uncertainty") for n in trainable_names), \
        f"冻结名单有遗漏，非 uncertainty 前缀的可训练参数: {trainable_names}"


def main():
    t0 = time.time()

    cfg = config_to_dict(load_config("configs/default.yaml"))
    mc = cfg["model"]

    # ── 数据加载（两阶段共用）─────────────────────────
    train_ds = ArgoverseTrajectoryDataset(
        "data/processed/train.npz",
        lane_graphs_path="data/processed/lane_graphs.npz",
    )
    val_ds = ArgoverseTrajectoryDataset(
        "data/processed/val.npz",
        lane_graphs_path="data/processed/lane_graphs.npz",
    )
    train_loader = create_dataloader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # ── 模型构造（两阶段共用）─────────────────────────
    model = TrajectoryModel(
        input_dim=mc["input_dim"], hidden_dim=mc["hidden_dim"],
        num_mlp_layers=mc.get("num_mlp_layers", 3), dropout=mc.get("dropout", 0.1),
        num_modes=mc["num_modes"], pred_len=cfg["data"].get("pred_len", 30),
        pooling_type=mc.get("pooling", "attention"), use_uncertainty=True,
        use_temporal_conv=mc.get("use_temporal_conv", True),
        use_lane=mc.get("use_lane", True),
        use_city=mc.get("use_city", True),
        share_uncertainty=True,    # 第三阶段定稿：共享不确定性头
        use_interaction=True,      # 第三阶段定稿：交互模块 + 邻居独立头
        branch_hidden_dim=mc.get("branch_hidden_dim", 96),
    )
    total_params, _ = count_parameters(model)
    print(f"模型参数量: {total_params:,}")

    best_ade_a = float("nan")

    if SKIP_STAGE_A:
        # ── 跳过阶段 A：直接从迁移后 checkpoint 加载 ──
        print("=" * 58)
        print("  跳过阶段 A：从迁移后 checkpoint 加载")
        print("=" * 58)
        checkpoint = torch.load(START_CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  已加载: {START_CKPT} (epoch {checkpoint.get('epoch', '?')})")
    else:
        # ── 阶段 A：正常训练 ──────────────────────────
        print("=" * 58)
        print("  阶段 A: 正常训练 (lambda_unc=0.0)")
        print("=" * 58)

        cfg["training"]["epochs"] = 30          # 阶段 A 训练 30 epoch
        cfg["training"]["warmup_epochs"] = 3
        cfg["loss"]["lambda_unc"] = 0.0         # 不启用 NLL
        cfg["logging"]["log_dir"] = "outputs/logs/unc_stage_a"
        cfg["logging"]["checkpoint_dir"] = "outputs/checkpoints/unc_stage_a"

        set_seed(42)
        logger_a = setup_logger(cfg["logging"]["log_dir"], "stage_a")
        logger_a.info(f"参数量: {total_params:,}")

        trainer_a = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                            config=cfg, logger=logger_a, tb_writer=None)
        result_a = trainer_a.train()

        best_ade_a = result_a["best_val_min_ade"]
        print(f"\n阶段 A 完成: minADE = {best_ade_a:.4f}, epoch = {result_a['best_epoch']+1}")

        # 加载阶段 A 最佳模型（供阶段 B 微调）
        ckpt_path = "outputs/checkpoints/unc_stage_a/best_model.pt"
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # ── 阶段 B：冻结 + 不确定性微调 ──────────────────
    print("\n" + "=" * 58)
    print("  阶段 B: 不确定性 NLL 微调 (lambda_unc=0.05)")
    print("=" * 58)

    # 冻结除不确定性头外的所有参数
    freeze_except_uncertainty(model)
    trainable_a = count_trainable(model)

    # 切换配置
    cfg["training"]["epochs"] = 15          # 阶段 B 微调 15 epoch
    cfg["training"]["warmup_epochs"] = 2
    cfg["training"]["lr"] = 5e-4            # 降低学习率
    cfg["loss"]["lambda_unc"] = 0.05         # 启用 NLL
    cfg["logging"]["log_dir"] = "outputs/logs/unc_stage_b"
    cfg["logging"]["checkpoint_dir"] = "outputs/checkpoints/unc_stage_b"

    set_seed(42)
    logger_b = setup_logger(cfg["logging"]["log_dir"], "stage_b")
    logger_b.info(f"可训练参数(仅不确定性头): {trainable_a:,}")

    trainer_b = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                        config=cfg, logger=logger_b, tb_writer=None,
                        monitor_metric="val_loss")  # Stage B 冻结轨迹预测，minADE 不变，改用含 NLL 的 val_loss 监控
    result_b = trainer_b.train()

    # ── 总结 ──────────────────────────────────────────
    elapsed = (time.time() - t0) / 60
    print("\n" + "=" * 58)
    print("  不确定性微调完成" + ("（跳过阶段A）" if SKIP_STAGE_A else "（完整两阶段）"))
    print("=" * 58)
    if not SKIP_STAGE_A:
        print(f"  阶段 A minADE: {best_ade_a:.4f}")
    print(f"  阶段 B best_{result_b['monitor_metric']}: {result_b['best_metric']:.4f}")
    print(f"  不确定性头参数: {trainable_a:,}")
    print(f"  总耗时: {elapsed:.0f} 分钟")
    print(f"  模型: outputs/checkpoints/unc_stage_b/best_model.pt")


if __name__ == "__main__":
    main()
