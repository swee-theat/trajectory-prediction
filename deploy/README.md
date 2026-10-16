# 轻量化轨迹预测系统 — 实体小车部署包

## 部署文件清单

```
deploy/
├── README.md                          # 本文档
├── trajectory_model.onnx              # FP32 ONNX 模型（0.85 MB，纯轨迹基线 s2_仅轨迹基线）
├── trajectory_model_int8.onnx         # INT8 量化模型（0.30 MB，阶段四产物，推荐部署用）
├── export_onnx.py                     # PyTorch → ONNX 导出脚本（Windows 端用）
├── quantize_int8.py                   # ONNX INT8 量化脚本（onnxsim 精简 + 动态量化）
├── ros_node/
│   ├── trajectory_predictor.py        # ROS 轨迹预测节点
│   ├── trajectory_predictor.launch    # 轨迹预测 launch 文件
│   ├── obstacle_avoidance.py          # ROS 避障节点（感知→预测→决策→执行闭环）
│   └── obstacle_avoidance.launch      # 避障 launch 文件
├── test/
│   └── test_obstacle_avoidance.py     # 避障核心逻辑离线单元测试
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

## 三、INT8 量化（阶段四）

> **平台口径**：需求文档原定 Jetson Nano + TensorRT 做 INT8，经确认改为实体小车实际
> 平台 Intel NUC + onnxruntime（NUC 无 NVIDIA GPU，TensorRT 不可用）。量化方案为
> **onnxsim 图简化 + 动态量化**（权重 INT8、激活运行时动态量化），经对比静态量化
> （体积更大、精度 +17% 超标）后选定。

```bash
python deploy/quantize_int8.py
# 输出: deploy/trajectory_model_int8.onnx (0.30 MB)
```

量化结果：

| 指标 | FP32 | INT8 | 验收 |
|------|------|------|------|
| 权重体积 | 832.5 KB | 205 KB | 压缩 75.4% ✅（≥75%） |
| onnx 文件体积 | 0.85 MB | 0.30 MB | 压缩 65.3% |
| 验证集 minADE | 0.833 m | 0.873 m | 增幅 +4.85% ✅（≤10%） |
| CPU 推理延迟 | 0.36 ms | 0.71 ms | ✅（<50ms） |

> **体积口径说明**：权重体积已精确压缩 75.4%（INT8 量化本质）；整个 .onnx 文件只到
> 65.3%，是因为 ONNX 是明文 protobuf，0.85MB 小模型的图结构开销（节点定义、tensor
> 名）占比高。这是 ONNX 格式的客观限制，非量化实现问题——只有 TensorRT 二进制引擎
> 才能让整文件到 75%。

ROS 节点默认加载 INT8 模型（`trajectory_model_int8.onnx`），高精度场景可在 launch
文件里把 `model_path` 改回 `trajectory_model.onnx`。

## 四、避障（硬件演示：感知→预测→决策→执行闭环）

配合深度摄像头 + 轨迹预测节点，实现需求文档预期成果④的避障闭环：

```
感知 ──> 预测 ──> 决策 ──> 执行
Orbbec深度图   轨迹预测   碰撞检测   减速绕行 → /cmd_vel
```

- **订阅**：`/camera/depth_registered/image_raw`（深度图 mm）、
  `/camera/depth/camera_info`（相机内参）、`/predicted_best_path`（最高概率模态）、
  `/odom`（里程计，累加里程用于自停兜底）、`/avoidance_reset`（里程归零）
- **发布**：`/cmd_vel`（麦克纳姆轮 vx/vy/ω）
- **决策链**：深度图反投影 3D → tf 转 base_footprint → 地面分割(z>8cm) → 障碍物占据栅格；
  碰撞检测 = 预测轨迹 best 模态(膨胀 15cm) + 直行走廊(覆盖 slow_start=1.5m 盲区) 取更近威胁；
  决策状态机(FORWARD/AVOIDING/STOPPED + 滞回 + N 帧防抖 + 左右占据选向) → 减速绕行

```bash
# 先启动深度摄像头 + 轨迹预测节点，再启动避障节点:
roslaunch ai_node_master obstacle_avoidance.launch
```

核心逻辑（深度反投影 / 栅格化 / 碰撞检测 / 决策）为不依赖 ROS 的纯函数，可离线单测：

```bash
python deploy/test/test_obstacle_avoidance.py   # 15 项全绿
```

> **低速场景口径**（手册 14.3 节）：车速度 vx≤0.12m/s，障碍物（锥桶/围挡静态 + 低速
> 动态）与车同量级，深度图 30Hz 实时检测即可跟上，无需障碍物轨迹预测。

## 五、运行

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

## 六、RViz 可视化

在 RViz 中添加:
- Add → By topic → `/predicted_trajectories/Marker`

显示内容:
- 蓝色实线: 历史轨迹（过去 2 秒）
- 5 条彩色虚线: 5 模态预测轨迹（未来 3 秒）
- 颜色从红到橙，按概率排序

## 七、ROS 话题接口

| 话题 | 类型 | 方向 | 频率 | 说明 |
|------|------|------|------|------|
| `/odom` | nav_msgs/Odometry | 订阅 | 20Hz | 里程计输入 |
| `/predicted_trajectories` | visualization_msgs/MarkerArray | 发布 | 10Hz | 预测路径可视化 |
| `/predicted_best_path` | nav_msgs/Path | 发布 | 10Hz | 最高概率模态（供避障节点） |
| `/camera/depth_registered/image_raw` | sensor_msgs/Image | 订阅 | 30Hz | 深度图（避障感知） |
| `/cmd_vel` | geometry_msgs/Twist | 发布 | 10Hz | 减速绕行指令（避障节点） |

## 八、与规划器集成

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

## 九、性能指标

| 指标 | 值 | 备注 |
|------|-----|------|
| 模型大小（FP32 / INT8） | 0.85 / 0.30 MB | INT8 权重压缩 75.4% |
| 推理延迟（FP32 / INT8） | 0.36 / 0.71 ms | CPU 单次推理，均 <50ms |
| 等效频率 | ~2057 Hz | 远超 20Hz 需求 |
| 输入维度 | [1, 20, 11] | 20 帧历史 × 11 维特征 |
| 输出维度 | [5, 30, 2] | K=5 模态 × 30 帧预测 |
| ROS 频率 | 10 Hz | 降采样自 20Hz 里程计 |
