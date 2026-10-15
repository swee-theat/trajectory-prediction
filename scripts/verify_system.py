"""系统验证脚本：在不依赖真实数据的情况下，验证模型、损失、训练流程是否正确。

用法: python scripts/verify_system.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from utils.param_counter import count_parameters
from utils.seed import set_seed
from utils.config import load_config, config_to_dict
from models.trajectory_model import TrajectoryModel
from models.attention_pooling import AttentionPooling, MeanPooling, LSTMPooling
from train.losses import CombinedLoss
from eval.metrics import (
    compute_ade, compute_fde, compute_min_ade, compute_min_fde,
    compute_miss_rate, compute_endpoint_hit_rate, compute_all_metrics,
)


def test_model_build():
    """测试 1: 模型构建与参数量验证。"""
    print("\n" + "=" * 60)
    print("  测试 1: 模型构建")
    print("=" * 60)

    cfg = config_to_dict(load_config("configs/default.yaml"))
    input_dim = cfg["model"].get("input_dim", 11)

    model = TrajectoryModel(
        input_dim=input_dim,
        hidden_dim=cfg["model"].get("hidden_dim", 128),
        num_mlp_layers=cfg["model"].get("num_mlp_layers", 3),
        dropout=cfg["model"].get("dropout", 0.1),
        num_modes=cfg["model"].get("num_modes", 3),
        pred_len=cfg["data"].get("pred_len", 30),
        pooling_type=cfg["model"].get("pooling", "attention"),
        use_uncertainty=cfg["model"].get("use_uncertainty", True),
        use_temporal_conv=cfg["model"].get("use_temporal_conv", True),
    )

    total_params, _ = count_parameters(model)
    assert total_params < 250_000, f"参数量 {total_params} 超过 250K 目标！"
    assert total_params > 50_000, f"参数量 {total_params} 过少，可能有问题"
    print(f"  [OK] 参数量: {total_params:,} (目标 < 250K)")

    return model


def test_forward_pass(model):
    """测试 2: 前向传播形状验证。"""
    print("\n" + "=" * 60)
    print("  测试 2: 前向传播")
    print("=" * 60)

    B, T_obs, T_pred = 16, 20, 30
    K = model.num_modes
    input_dim = model.input_dim
    history = torch.randn(B, T_obs, input_dim)

    model.eval()
    with torch.no_grad():
        output = model(history)

    # 验证输出形状（K 从模型读取，不硬编码）
    assert output["trajectories"].shape == (B, K, T_pred, 2), \
        f"轨迹形状错误: {output['trajectories'].shape}"
    assert output["mode_probs"].shape == (B, K), \
        f"模态概率形状错误: {output['mode_probs'].shape}"
    assert output["mode_logits"].shape == (B, K), \
        f"模态 logits 形状错误: {output['mode_logits'].shape}"
    assert output["uncertainties"].shape == (B, K, T_pred, 2), \
        f"不确定性形状错误: {output['uncertainties'].shape}"
    assert output["encoded_feature"].shape == (B, 128), \
        f"编码特征形状错误: {output['encoded_feature'].shape}"

    # 验证模态概率和为 1
    prob_sums = output["mode_probs"].sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
        f"模态概率和不为 1: {prob_sums}"

    # 验证不确定性非负
    assert (output["uncertainties"] >= 0).all(), "不确定性存在负值！"

    print("  [OK] 所有输出形状正确")
    print(f"    trajectories:    {output['trajectories'].shape}")
    print(f"    mode_probs:      {output['mode_probs'].shape}")
    print(f"    uncertainties:   {output['uncertainties'].shape}")

    return output


def test_pooling_variants():
    """测试 3: 池化变体。"""
    print("\n" + "=" * 60)
    print("  测试 3: 池化变体")
    print("=" * 60)

    B, T, H = 8, 20, 128
    x = torch.randn(B, T, H)

    for name, pool in [
        ("Attention", AttentionPooling(H)),
        ("Mean", MeanPooling()),
        ("LSTM", LSTMPooling(H)),
    ]:
        pooled, weights = pool(x, return_weights=True)
        assert pooled.shape == (B, H), f"{name}: 输出形状错误 {pooled.shape}"

        if weights is not None:
            # 验证权重和为 1
            weight_sums = weights.sum(dim=-1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
                f"{name}: 权重和不为 1"

        print(f"  [OK] {name} 池化: 输出形状 {pooled.shape}")

    print("  [OK] 全部池化变体通过")


def test_loss_function():
    """测试 4: 联合损失函数（WTA 逻辑验证）。"""
    print("\n" + "=" * 60)
    print("  测试 4: 损失函数")
    print("=" * 60)

    criterion = CombinedLoss(lambda_traj=0.7, lambda_mode=0.3)

    # 构造完美预测 + 噪声预测
    B, K, T_pred = 4, 3, 30
    gt = torch.randn(B, T_pred, 2)

    # 模态 1 完美匹配真值，模态 0/2 随机噪声
    trajs = torch.randn(B, K, T_pred, 2)
    trajs[:, 1, :, :] = gt  # 模态 1 = 真值

    logits = torch.randn(B, K)

    output = {
        "trajectories": trajs,
        "mode_logits": logits,
        "mode_probs": torch.softmax(logits, dim=-1),
    }

    loss_dict = criterion(output, gt)

    assert loss_dict["loss"].item() > 0, "损失应为正值"
    assert loss_dict["best_mode_idx"][0].item() == 1, \
        f"WTA 应选模态 1(真值)，实际选了 {loss_dict['best_mode_idx'][0].item()}"

    print(f"  [OK] WTA 正确选择了最佳模态（模态1）")
    print(f"    总损失: {loss_dict['loss'].item():.4f}")
    print(f"    轨迹损失: {loss_dict['traj_loss'].item():.4f}")
    print(f"    模态损失: {loss_dict['mode_loss'].item():.4f}")


def test_metrics():
    """测试 5: 评估指标计算。"""
    print("\n" + "=" * 60)
    print("  测试 5: 评估指标")
    print("=" * 60)

    B, K, T_pred = 8, 3, 30
    trajs = torch.randn(B, K, T_pred, 2)
    gt = torch.randn(B, T_pred, 2)

    # 验证指标计算不报错
    ade = compute_ade(trajs, gt)
    fde = compute_fde(trajs, gt)
    min_ade = compute_min_ade(trajs, gt)
    min_fde = compute_min_fde(trajs, gt)
    mr = compute_miss_rate(trajs, gt, threshold=2.0)
    hit = compute_endpoint_hit_rate(trajs, gt, threshold=3.0)
    all_m = compute_all_metrics(trajs, gt)

    assert ade.shape == (B,), f"ADE 形状错误: {ade.shape}"
    assert min_ade.shape == (B,), f"minADE 形状错误: {min_ade.shape}"
    assert 0 <= mr <= 1, f"MR 应在 [0,1]: {mr}"
    assert 0 <= hit <= 1, f"命中率应在 [0,1]: {hit}"

    # minADE 应该 <= ADE（oracle 能选到最佳模态）
    for i in range(B):
        assert min_ade[i] <= ade[i], \
            f"minADE({min_ade[i]:.4f}) > ADE({ade[i]:.4f})，这不应该发生!"

    print(f"  [OK] 所有指标计算正确")
    print(f"    ADE={ade.mean():.4f}, FDE={fde.mean():.4f}")
    print(f"    minADE={min_ade.mean():.4f}, minFDE={min_fde.mean():.4f}")
    print(f"    MR(2m)={mr:.4f}, Hit(3m)={hit:.4f}")


def test_overfit_single_batch():
    """测试 6: 单 batch 过拟合（验证梯度流和优化器工作正常）。"""
    print("\n" + "=" * 60)
    print("  测试 6: 单 batch 过拟合")
    print("=" * 60)

    cfg = config_to_dict(load_config("configs/default.yaml"))
    model = TrajectoryModel(
        input_dim=cfg["model"].get("input_dim", 11),
        hidden_dim=64, num_mlp_layers=2,
        dropout=0.0, num_modes=2, pred_len=30,
        pooling_type="attention", use_uncertainty=False,
        use_temporal_conv=False,
    )

    criterion = CombinedLoss(lambda_traj=0.7, lambda_mode=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    B, T_obs, T_pred = 16, 20, 30
    input_dim = model.input_dim
    history = torch.randn(B, T_obs, input_dim)
    future = torch.randn(B, T_pred, 2)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(200):
        optimizer.zero_grad()
        output = model(history)
        loss_dict = criterion(output, future)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 199:
            final_loss = loss.item()

    assert final_loss < initial_loss * 0.5, \
        f"过拟合失败: 初始损失 {initial_loss:.4f} → 最终损失 {final_loss:.4f}"

    print(f"  [OK] 过拟合成功！")
    print(f"    初始损失: {initial_loss:.4f} → 最终损失: {final_loss:.4f}")


def main():
    print("\n" + "█" * 60)
    print("  轻量化多模态轨迹预测系统 — 系统验证")
    print("█" * 60)

    set_seed(42)

    try:
        model = test_model_build()
        test_forward_pass(model)
        test_pooling_variants()
        test_loss_function()
        test_metrics()
        test_overfit_single_batch()

        print("\n" + "█" * 60)
        print("  [PASS] 全部测试通过！系统核心功能正常。")
        print("█" * 60)

    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
