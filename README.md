# 轻量化多模态轨迹预测系统

面向校园/园区无人车低速场景的轻量化多模态轨迹预测系统。支持车道线融合、城市域感知与两车交互下的碰撞预警，提供 Web 可视化 Demo 与实体小车（ONNX + ROS）端到端部署。

## 核心特性

- **三阶段渐进式架构**：轨迹预测基线 → 车道线融合 + 城市特征 → 两车交互 + 避障决策
- **轻量化**：三阶段定稿模型仅 282K 参数（≤ 288K 约束），ONNX 0.85MB，CPU 单次推理 0.49ms
- **多模态预测**：WTA（Winner-Take-All）轨迹损失 + 模态交叉熵，K=5 模态输出
- **不确定性估计**：独立不确定性头，量化预测置信度
- **车道线 GCN 融合**：图卷积编码车道拓扑，约束预测轨迹
- **两车交互**：轻量交叉注意力交互编码（约 16.8K 参数），联合预测两车轨迹 + 逐帧碰撞检测
- **端到端部署**：PyTorch → ONNX → onnxruntime → ROS 节点（Intel NUC 实体小车）

## 系统架构

```
阶段一              阶段二                阶段三
轨迹预测基线  →   +车道线 GCN     →   +两车交互 + 碰撞预警
MLP+注意力池化     +城市域感知          交叉注意力交互编码
```

模型输入 11 维轨迹特征（位置 / 速度 / 时间 / 航向 / 曲率 / 邻车信息），观测 20 帧（2 秒）→ 预测 30 帧（3 秒），输出 K=5 条候选轨迹及各自概率。

## 结果

| 指标 | 值 |
|---|---|
| 阶段二 仅轨迹基线 val minADE | 0.6144 |
| 阶段二 +车道线 val minADE | 0.6107 |
| 阶段二 +车道线+城市特征 val minADE | 0.6300 |
| 阶段三 ego minADE | 0.8312 |
| 阶段三 邻居轨迹 minADE | 5.26 |
| 交互后碰撞误报率 | 8.9% |
| 参数量（基线 / 三阶段） | 213K / 282K |
| ONNX 大小 / CPU 推理延迟 | 0.85MB / 0.49ms |
| INT8 量化（阶段四） | 权重压缩 75.4%、文件 0.30MB、minADE 增幅 +4.85%、延迟 0.71ms |

## 目录结构

```
├── configs/          # 训练与模型配置（default.yaml）
├── data/             # 数据预处理、增强、Dataset、DataLoader
├── models/           # 模型：编码器、车道线 GCN、交互模块、预测器、不确定性
├── train/            # 训练器、损失函数、优化器、早停
├── eval/             # 评估指标
├── scripts/          # 预处理 / 训练 / 评估 / 可视化 / 数据下载脚本
├── demo/             # Flask Web 可视化 Demo
├── deploy/           # ONNX 导出 + ROS 节点（实体小车部署）
├── visualization/    # 轨迹与失败案例可视化
└── utils/            # 配置、日志、checkpoint、随机种子等工具
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

使用 Argoverse v1.1 Motion Forecasting 数据集（本项目训练使用 MIA、PIT 两城），下载脚本见 `scripts/fast_download.py` / `data/download.py`。

### 3. 数据预处理

```bash
python scripts/preprocess_real_data.py --data-root <CSV数据目录>
# 输出 data/processed/train.npz / val.npz
```

### 4. 训练

```bash
# 阶段二消融（车道线融合 + 城市特征）
python scripts/run_ablation_stage2.py --epochs 80

# 阶段三两车交互（需先重建数据，见脚本 docstring）
python scripts/train_stage3_interaction.py --epochs 80 --group both

# 不确定性头校准（Stage A/B）
python scripts/train_with_uncertainty.py
```

### 5. 评估

```bash
python scripts/eval_stage2_curves.py
python scripts/eval_stage3_interaction.py
```

### 6. Web 可视化 Demo

```bash
python demo/app.py
# 浏览器打开 http://localhost:5000
# 支持：鼠标手绘轨迹 / 上传 CSV / 内置示例
```

### 7. 实体小车部署（ONNX + ROS）

```bash
python deploy/export_onnx.py     # 导出 deploy/trajectory_model.onnx (FP32)
python deploy/quantize_int8.py   # INT8 量化 → deploy/trajectory_model_int8.onnx (0.30MB)
```

ROS 节点默认加载 INT8 模型，Intel NUC 部署步骤见 [deploy/README.md](deploy/README.md)。

## 环境

- **训练**：Windows 11 + NVIDIA RTX 3060 6GB，PyTorch 2.x
- **部署**：Intel NUC（Ubuntu 20.04 + ROS Noetic + onnxruntime）
