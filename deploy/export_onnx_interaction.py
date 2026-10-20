#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将第三阶段两车交互模型导出为 ONNX 格式，用于实体小车双车交互推理。

与 deploy/export_onnx.py（单车基线）的区别：
    - 加载第三阶段方案二定稿模型（use_interaction=True，Stage B 校准后）
    - 输入增加 neighbor_history / city_id
    - 输出增加 neighbor_trajectories / neighbor_mode_probs

导出口径与 demo/app.py 完全一致：
    - checkpoint = outputs/checkpoints/unc_stage_b/best_model.pt（Stage B 校准，
      轨迹参数 = s3_组B交互_scheme2c，共享不确定性头已 NLL 校准）
    - 构造参数 share_uncertainty=True, use_lane=True, use_city=True, use_interaction=True
    - 不传 lane_nodes（实车/demo 都无车道线检测），forward 走无 lane 分支

关于 neighbor_mask：方案一（wen_ti/04）ego 预测走 `ego_feat = encoded` 不经过
interaction，邻居预测走 `neighbor_encoder → neighbor_predictor` 也不看 mask——
mask 只影响 interaction 的死路径 context（不参与任何导出输出）。故 torch 导出时
neighbor_mask 被 dead-code 消除，ONNX 输入只有 history / neighbor_history / city_id。
导出 wrapper 内部用固定 mask 仅用于触发 forward 的交互分支（需非 None）。

用法:
    python deploy/export_onnx_interaction.py
    输出: deploy/trajectory_model_interaction.onnx
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.trajectory_model import TrajectoryModel
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
INPUT_DIM = 11
HIDDEN_DIM = 128
NUM_MODES = 5
PRED_LEN = 30
OBS_LEN = 20
CITY_ID = 2  # 与 demo/app.py 一致：固定 MIA 城（训练数据最大类）

# 与 demo/app.py 的 CHECKPOINT_PATH 一致：Stage B 校准后的定稿模型
CHECKPOINT_PATH = "outputs/checkpoints/unc_stage_b/best_model.pt"
OUTPUT_PATH = "deploy/trajectory_model_interaction.onnx"


class ExportWrapper(torch.nn.Module):
    """导出 ego + 邻居两路输出，顺序与 output_names 一一对应。

    TrajectoryModel.forward 返回 dict 且含 10 个键，TorchScript 导出器按 dict 顺序
    分配 output_names，用 wrapper 显式返回 tuple 避免错配。只导出实车节点用到的 5 个。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, history, neighbor_history, city_id):
        # neighbor_mask 对导出的 5 个输出无影响（ego 走 encoded、邻居走独立头都不看
        # mask），仅需非 None 触发 forward 的交互分支；trace 固化后不成为 ONNX 输入。
        neighbor_mask = torch.ones((history.shape[0], 2), device=history.device)
        out = self.model(
            history,
            neighbor_history=neighbor_history,
            neighbor_mask=neighbor_mask,
            city_id=city_id,
        )
        return (
            out["trajectories"],           # ego 轨迹 [1, K, 30, 2]
            out["mode_probs"],             # ego 概率 [1, K]
            out["uncertainties"],          # ego 不确定性 [1, K, 30, 2]
            out["neighbor_trajectories"],  # 邻居轨迹 [1, K, 30, 2]
            out["neighbor_mode_probs"],    # 邻居概率 [1, K]
        )


def main():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型（构造参数与 demo/app.py L56-72 完全一致）
    print("加载模型...")
    model = TrajectoryModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_mlp_layers=3,
        dropout=0.1,
        num_modes=NUM_MODES,
        pred_len=PRED_LEN,
        pooling_type="attention",
        use_uncertainty=True,
        use_temporal_conv=True,
        share_uncertainty=True,
        use_lane=True,
        use_city=True,
        use_interaction=True,
        branch_hidden_dim=96,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  从 checkpoint 加载 (epoch {checkpoint.get('epoch', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print("  从 state_dict 加载")

    model.eval()
    model.cpu()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # 2. ONNX 导出（batch=1 实时推理；obs_len 固定 20）
    print("导出 ONNX...")
    wrapper = ExportWrapper(model)
    dummy = (
        torch.randn(1, OBS_LEN, INPUT_DIM),   # history
        torch.randn(1, 2, OBS_LEN, INPUT_DIM),  # neighbor_history [1, 2, 20, 11]
        torch.tensor([CITY_ID], dtype=torch.int64),  # city_id [1]（Embedding 需 int64）
    )
    torch.onnx.export(
        wrapper,
        dummy,
        OUTPUT_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["history", "neighbor_history", "city_id"],
        output_names=[
            "trajectories",
            "mode_probs",
            "uncertainties",
            "neighbor_trajectories",
            "neighbor_mode_probs",
        ],
        dynamic_axes={
            "history": {0: "batch"},
            "neighbor_history": {0: "batch"},
            "city_id": {0: "batch"},
            "trajectories": {0: "batch"},
            "mode_probs": {0: "batch"},
            "uncertainties": {0: "batch"},
            "neighbor_trajectories": {0: "batch"},
            "neighbor_mode_probs": {0: "batch"},
        },
        dynamo=False,  # 旧版 TorchScript 导出器，避免 onnxscript 依赖
    )
    print(f"  ONNX 模型已保存: {OUTPUT_PATH}")

    # 3. 验证：ONNX Runtime vs PyTorch（用同一组输入）
    print("验证 ONNX Runtime 推理...")
    import onnxruntime as ort

    history = torch.randn(1, OBS_LEN, INPUT_DIM)
    neighbor_history = torch.randn(1, 2, OBS_LEN, INPUT_DIM)
    neighbor_mask = torch.ones((1, 2), dtype=torch.float32)
    city_id = torch.tensor([CITY_ID], dtype=torch.int64)

    model.eval()  # 导出 trace 会把 model 切回 training，重新 eval 避免 dropout 随机
    with torch.no_grad():
        torch_out = model(
            history,
            neighbor_history=neighbor_history,
            neighbor_mask=neighbor_mask,
            city_id=city_id,
        )

    session = ort.InferenceSession(OUTPUT_PATH)
    onnx_outs = session.run(
        ["trajectories", "mode_probs", "uncertainties",
         "neighbor_trajectories", "neighbor_mode_probs"],
        {
            "history": history.numpy(),
            "neighbor_history": neighbor_history.numpy(),
            "city_id": city_id.numpy(),
        },
    )

    for name, torch_val, onnx_val in [
        ("trajectories",          torch_out["trajectories"],           onnx_outs[0]),
        ("mode_probs",            torch_out["mode_probs"],             onnx_outs[1]),
        ("uncertainties",         torch_out["uncertainties"],          onnx_outs[2]),
        ("neighbor_trajectories", torch_out["neighbor_trajectories"],  onnx_outs[3]),
        ("neighbor_mode_probs",   torch_out["neighbor_mode_probs"],    onnx_outs[4]),
    ]:
        diff = np.abs(torch_val.numpy() - onnx_val).max()
        status = "OK" if diff < 1e-4 else f"差异: {diff:.2e}"
        print(f"  {name}: PyTorch {torch_val.shape} vs ONNX {onnx_val.shape} — {status}")

    # 4. 性能基准（CPU）
    print("CPU 推理基准测试...")
    import time
    feed = {
        "history": history.numpy(),
        "neighbor_history": neighbor_history.numpy(),
        "city_id": city_id.numpy(),
    }
    for _ in range(10):
        session.run(None, feed)
    N = 100
    t0 = time.time()
    for _ in range(N):
        session.run(None, feed)
    elapsed = (time.time() - t0) / N * 1000
    print(f"  CPU 推理延迟: {elapsed:.2f} ms/样本 (batch=1)")

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\nONNX 文件大小: {size_mb:.2f} MB")
    print("部署就绪!")


if __name__ == "__main__":
    main()
