# 轻量化轨迹预测系统 — 实体小车部署包

## 部署文件清单

```
deploy/
├── README.md                          # 本文档
├── trajectory_model.onnx              # ONNX 模型（0.85 MB，纯轨迹基线 s2_仅轨迹基线）
├── export_onnx.py                     # PyTorch → ONNX 导出脚本（Windows 端用）
├── ros_node/
│   ├── trajectory_predictor.py        # ROS 轨迹预测节点
│   └── trajectory_predictor.launch    # ROS launch 文件
└── requirements_nuc.txt               # NUC 端 Python 依赖
```

## 一、NUC 端环境准备

```bash
# 1. SSH 到小车 NUC
ssh <用户>@<小车IP>

# 2. 安装 ONNX Runtime (CPU 版)
pip3 install -r requirements_nuc.txt

# 3. 创建模型目录
mkdir -p ~/ros_space/src/ai_node_master/models/

# 4. 将以下文件拷贝到小车:
#    deploy/trajectory_model.onnx → ~/ros_space/src/ai_node_master/models/
#    deploy/ros_node/trajectory_predictor.py → ~/ros_space/src/ai_node_master/src/
#    deploy/ros_node/trajectory_predictor.launch → ~/ros_space/src/ai_node_master/launch/
#    deploy/requirements_nuc.txt → ~/ros_space/src/ai_node_master/
```

## 二、模型导出（Windows 训练端）

> **模型口径**：导出的是「纯轨迹基线」`s2_仅轨迹基线`（minADE 0.6144，213,117 参数），
> 而非三阶段定稿模型。原因：实体小车是单智能体场景（无车道线检测/多车检测/城市标签），
> 定稿模型的 `use_lane` / `use_city` / `use_interaction` 依赖小车无法提供的输入，强行导出
> 会预测退化。纯轨迹基线只吃 11 维轨迹特征，与 ROS 节点输入严格匹配。

```bash
# 在 Windows RTX 3060 上训练完成后:
cd D:\game\da_chuang
python deploy/export_onnx.py
# 输出: deploy/trajectory_model.onnx (0.85 MB)
```

## 三、运行

```bash
# 方式1: 直接运行节点
cd ~/ros_space
source devel/setup.bash
roscore &
python3 src/ai_node_master/src/trajectory_predictor.py

# 方式2: 使用 launch 文件
roslaunch ai_node_master trajectory_predictor.launch

# 方式3: 配合完整导航系统
roslaunch ai_node_master navigation.launch &
roslaunch ai_node_master trajectory_predictor.launch
```

## 四、RViz 可视化

在 RViz 中添加:
- Add → By topic → `/predicted_trajectories/Marker`

显示内容:
- 蓝色实线: 历史轨迹（过去 2 秒）
- 5 条彩色虚线: 5 模态预测轨迹（未来 3 秒）
- 颜色从红到橙，按概率排序

## 五、ROS 话题接口

| 话题 | 类型 | 方向 | 频率 | 说明 |
|------|------|------|------|------|
| `/odom` | nav_msgs/Odometry | 订阅 | 20Hz | 里程计输入 |
| `/predicted_trajectories` | visualization_msgs/MarkerArray | 发布 | 10Hz | 预测路径可视化 |

## 六、与规划器集成

预测轨迹可用于改进 DWA 局部规划:

```python
# 在 DWA 规划器中订阅预测轨迹
def predicted_path_callback(msg):
    # msg.markers[0] 是历史轨迹
    # msg.markers[1:] 是 K=5 预测轨迹（按概率排序）
    best_traj = msg.markers[1]  # 最高概率模态
    # 将预测终点作为 DWA 的动态目标点
    goal = (best_traj.points[-1].x, best_traj.points[-1].y)
```

## 七、性能指标

| 指标 | 值 | 备注 |
|------|-----|------|
| 模型大小 | 0.85 MB | 可存储在 NUC 上 |
| 推理延迟 | 0.49 ms | CPU 单次推理 |
| 等效频率 | ~2057 Hz | 远超 20Hz 需求 |
| 输入维度 | [1, 20, 11] | 20 帧历史 × 11 维特征 |
| 输出维度 | [5, 30, 2] | K=5 模态 × 30 帧预测 |
| ROS 频率 | 10 Hz | 降采样自 20Hz 里程计 |
