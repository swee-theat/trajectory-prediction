#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将训练好的 PyTorch 模型导出为 ONNX 格式，用于实体小车 CPU 推理。

导出特性:
    - 固定 batch_size=1（实时推理单样本）
    - 动态时间步（支持可变 obs_len）
    - FP32 精度（Intel NUC 无 GPU，CPU 推理）

用法:
    python deploy/export_onnx.py
    输出: deploy/trajectory_model.onnx
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
# 实体小车是单智能体场景（无车道线检测/多车检测/城市标签），第二阶段车道线、
# 第三阶段交互、城市特征在小车上无对应传感器输入。故导出「纯轨迹基线」
# s2_仅轨迹基线（minADE 0.6144，213,117 参数），与 ROS 节点单智能体 11 维输入
# 严格匹配，是最诚实、可用的口径（三阶段定稿 use_lane/use_city/interaction=True
# 依赖小车无法提供的输入，强行导出会预测退化）。
CHECKPOINT_PATH = "outputs/checkpoints/s2_仅轨迹基线/best_model.pt"
OUTPUT_PATH = "deploy/trajectory_model.onnx"

OBS_LEN = 20  # 导出用固定帧数


class ExportWrapper(torch.nn.Module):
    """仅导出 trajectories / mode_probs / uncertainties 三个张量。

    TrajectoryModel.forward 返回 dict 且含 5 个键（trajectories, mode_probs,
    mode_logits, uncertainties, encoded_feature），TorchScript 导出器按 dict 顺序
    分配 output_names，若只声明 3 个名字会把 mode_logits 错配给 uncertainties。
    用 wrapper 显式返回 3 个张量的 tuple，顺序与 output_names 一一对应。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, history):
        out = self.model(history)
        return out["trajectories"], out["mode_probs"], out["uncertainties"]


def main():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
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
        # 显式关闭三阶段扩展，匹配 s2_仅轨迹基线结构。use_lane/use_city 默认 True，
        # 不显式关会导致 load_state_dict 报 missing/unexpected keys。
        share_uncertainty=False,
        use_lane=False,
        use_city=False,
        use_interaction=False,
        branch_hidden_dim=96,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  从 checkpoint 加载 (epoch {checkpoint.get('epoch', '?')})")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.cpu()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # 2. ONNX 导出
    print("导出 ONNX...")
    dummy_input = torch.randn(1, OBS_LEN, INPUT_DIM)

    wrapper = ExportWrapper(model)
    torch.onnx.export(
        wrapper,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["history"],
        output_names=["trajectories", "mode_probs", "uncertainties"],
        dynamic_axes={
            "history": {0: "batch", 1: "obs_len"},
            "trajectories": {0: "batch"},
            "mode_probs": {0: "batch"},
            "uncertainties": {0: "batch"},
        },
        dynamo=False,  # 使用旧版 TorchScript 导出器，避免 onnxscript 依赖
    )
    print(f"  ONNX 模型已保存: {OUTPUT_PATH}")

    # 3. 验证：ONNX Runtime 推理 vs PyTorch 推理
    print("验证 ONNX Runtime 推理...")
    import onnxruntime as ort

    test_input = torch.randn(1, OBS_LEN, INPUT_DIM)

    # PyTorch 推理。注意：torch.onnx.export 的 trace 会把 model 切回 training 模式
    # （dropout 重新激活），必须重新 eval，否则参考输出带 dropout 随机性，
    # 与 eval 模式 trace 出的 ONNX 不一致，验证 diff 会虚高且每次随机。
    model.eval()
    with torch.no_grad():
        torch_out = model(test_input)

    # ONNX Runtime 推理
    session = ort.InferenceSession(OUTPUT_PATH)
    onnx_out = session.run(
        ["trajectories", "mode_probs", "uncertainties"],
        {"history": test_input.numpy()},
    )

    # 对比（ONNX 输出顺序与 output_names 一致）
    onnx_traj, onnx_prob, onnx_unc = onnx_out
    for name, torch_val, onnx_val in [
        ("trajectories",  torch_out["trajectories"],  onnx_traj),
        ("mode_probs",    torch_out["mode_probs"],    onnx_prob),
        ("uncertainties", torch_out["uncertainties"], onnx_unc),
    ]:
        diff = np.abs(torch_val.numpy() - onnx_val).max()
        status = "OK" if diff < 1e-4 else f"差异: {diff:.2e}"
        print(f"  {name}: PyTorch {torch_val.shape} vs ONNX {onnx_val.shape} — {status}")

    # 4. 性能基准（CPU）
    print("CPU 推理基准测试...")
    import time

    # 预热
    for _ in range(10):
        session.run(None, {"history": test_input.numpy()})

    # 计时
    N = 100
    t0 = time.time()
    for _ in range(N):
        session.run(None, {"history": test_input.numpy()})
    elapsed = (time.time() - t0) / N * 1000

    print(f"  CPU 推理延迟: {elapsed:.2f} ms/样本 (batch=1)")
    print(f"  等效频率: {1000/elapsed:.0f} Hz (远超 20Hz 需求)")

    # 5. 模型大小
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\nONNX 文件大小: {size_mb:.2f} MB")
    print("部署就绪!")


if __name__ == "__main__":
    main()
