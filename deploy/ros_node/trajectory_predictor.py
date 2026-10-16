#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 轨迹预测节点：订阅 /odom → 实时推理 → 发布预测路径到 RViz。

数据流:
    /odom (Odometry, 20Hz)
        → 降采样到 10Hz (dt=0.1s，与训练一致)
        → 维护 20 帧历史缓冲区 (2s)
        → 提取 11 维特征
        → ONNX Runtime CPU 推理
        → 5 条预测轨迹 (K=5, 30 帧, 3s)
        → 发布 visualization_msgs/MarkerArray → RViz 显示
        → 发布自定义 predicted_path 话题 (供规划器使用)

实体小车环境:
    - Intel NUC, Ubuntu 20.04, ROS Noetic
    - Python 3.8, onnxruntime, numpy
    - 麦克纳姆轮全向底盘, 速度 0.015~0.12 m/s

用法:
    # 放到小车 NUC 上后:
    python3 trajectory_predictor.py
    # 或
    rosrun ai_node_master trajectory_predictor.py
"""

import rospy
import numpy as np
import onnxruntime as ort
from collections import deque
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA, Header

# ── 配置（与训练参数保持一致）────────────────────────
OBS_LEN = 20          # 观测帧数 (2s @ 10Hz)
PRED_LEN = 30         # 预测帧数 (3s @ 10Hz)
INPUT_DIM = 11        # 特征维度
NUM_MODES = 5         # K 模态数
TARGET_FREQ = 10.0    # 采样+预测频率 (Hz)。训练数据是 10Hz (dt=0.1s)，odom 20Hz 必须降采样
MODEL_PATH = "deploy/trajectory_model_int8.onnx"  # 默认 INT8 量化模型（阶段四产物，体积 303KB）；高精度场景可切回 trajectory_model.onnx

# 模态颜色 (RGBA)
MODE_COLORS = [
    ColorRGBA(0.94, 0.27, 0.27, 0.8),  # 红
    ColorRGBA(0.08, 0.72, 0.65, 0.8),  # 青
    ColorRGBA(0.55, 0.24, 0.96, 0.8),  # 紫
    ColorRGBA(0.20, 0.60, 0.96, 0.8),  # 蓝
    ColorRGBA(0.96, 0.60, 0.15, 0.8),  # 橙
]


class TrajectoryPredictor:
    """订阅里程计、运行 ONNX 推理、发布预测轨迹的 ROS 节点。"""

    def __init__(self):
        rospy.init_node("trajectory_predictor", anonymous=False)

        # ONNX Runtime 会话（模型路径优先取 launch 的 model_path 参数，回退到默认值）
        self.model_path = rospy.get_param("~model_path", MODEL_PATH)
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name

        # 历史缓冲区：存储 [x, y, yaw] × 20 帧（10Hz 采样，dt=0.1s）
        self.history = deque(maxlen=OBS_LEN)
        # 用 Time(0) 初始化，保证第一帧 odom 一定被接受
        self._last_sample_time = rospy.Time(0)

        # 订阅里程计
        rospy.Subscriber("/odom", Odometry, self._odom_callback)

        # 发布预测路径（RViz 可视化）
        self.marker_pub = rospy.Publisher(
            "/predicted_trajectories", MarkerArray, queue_size=2
        )

        # 发布最高概率模态轨迹（供避障节点做碰撞检测，接口比 MarkerArray 明确）
        self.best_path_pub = rospy.Publisher(
            "/predicted_best_path", Path, queue_size=2
        )

        rospy.loginfo("轨迹预测节点已启动")
        rospy.loginfo(f"  ONNX 模型: {self.model_path}")
        rospy.loginfo(f"  K={NUM_MODES}, obs_len={OBS_LEN}, pred_len={PRED_LEN}")

    # ── 里程计回调 ──────────────────────────────────
    def _odom_callback(self, msg: Odometry):
        """提取 (x, y, yaw)，降采样到 10Hz 后存入缓冲区。

        关键：用 msg.header.stamp（odom 时间戳）而非 rospy.Time.now() 判断，
        保证降采样间隔精确对应 odom 采样时刻，历史缓冲区相邻帧 dt=0.1s，
        与训练数据（Argoverse 10Hz）严格一致。否则 20Hz 数据直接入库会让
        速度/曲率被高估 2 倍，20 帧只覆盖 1 秒而非模型期望的 2 秒。
        """
        # 降采样：odom(20Hz) → 10Hz，丢弃高频帧
        now = msg.header.stamp
        if (now - self._last_sample_time).to_sec() < (1.0 / TARGET_FREQ):
            return  # 距上次采样不足 0.1s，丢弃
        self._last_sample_time = now

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 四元数 → yaw
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny, cosy)

        self.history.append((x, y, yaw))

        # 采样频率 = 预测频率 = 10Hz，攒满 20 帧即可预测
        if len(self.history) < OBS_LEN:
            return
        self._predict_and_publish()

    # ── 特征计算 ────────────────────────────────────
    def _extract_features(self) -> np.ndarray:
        """从历史缓冲区提取 11 维特征，与训练预处理一致。

        特征: [x, y, vx, vy, t/T, 1, 0,0,0, heading, curvature]

        返回:
            features: [1, 20, 11] numpy 数组
        """
        data = np.array(self.history, dtype=np.float32)  # [20, 3]
        xs = data[:, 0]
        ys = data[:, 1]
        yaws = data[:, 2]

        # 以最后一帧为中心
        xs_centered = xs - xs[-1]
        ys_centered = ys - ys[-1]

        # 速度（dt=0.1s，与训练一致）
        vx = np.zeros(OBS_LEN, dtype=np.float32)
        vy = np.zeros(OBS_LEN, dtype=np.float32)
        vx[1:] = (xs_centered[1:] - xs_centered[:-1]) / 0.1
        vy[1:] = (ys_centered[1:] - ys_centered[:-1]) / 0.1
        vx[0] = vx[1]
        vy[0] = vy[1]

        # 航向角和曲率（用 arctan2(vy,vx) 而非里程计 yaw，与训练数据一致）
        heading = np.unwrap(np.arctan2(vy, vx))
        curvature = np.zeros(OBS_LEN, dtype=np.float32)
        curvature[1:] = (heading[1:] - heading[:-1]) / 0.1
        curvature[0] = curvature[1]
        curvature = np.clip(curvature, -3.0, 3.0)

        # 构建 [20, 11] 特征
        # 注意: feats[:, 6:9] (dist, rel_x, rel_y) 保持为 0。
        # 实体小车上未接入多智能体检测，单智能体模式下社交特征为 0 是正确行为，
        # 与训练数据中单智能体场景（无其他 agent）的预处理逻辑一致。
        feats = np.zeros((OBS_LEN, INPUT_DIM), dtype=np.float32)
        feats[:, 0] = xs_centered
        feats[:, 1] = ys_centered
        feats[:, 2] = vx
        feats[:, 3] = vy
        feats[:, 4] = np.arange(OBS_LEN, dtype=np.float32) / (OBS_LEN - 1)
        feats[:, 5] = 1.0
        feats[:, 9] = heading
        feats[:, 10] = curvature

        return feats[np.newaxis, :, :]  # [1, 20, 11]

    # ── 推理 + 发布 ─────────────────────────────────
    def _predict_and_publish(self):
        """运行 ONNX 推理并发布预测路径。"""
        features = self._extract_features()

        # ONNX 推理
        outputs = self.session.run(
            ["trajectories", "mode_probs", "uncertainties"],
            {self.input_name: features},
        )
        trajectories = outputs[0][0]   # [5, 30, 2]（相对位移）
        mode_probs = outputs[1][0]     # [5]

        # 还原绝对坐标：预测位移 + 最后一个观测点
        last_x = self.history[-1][0]
        last_y = self.history[-1][1]
        trajectories_abs = trajectories + np.array([last_x, last_y])

        # 发布最高概率模态轨迹（odom 系），供避障节点做碰撞检测
        self._publish_best_path(trajectories_abs, mode_probs)

        # 发布 MarkerArray 到 RViz
        self._publish_markers(trajectories_abs, mode_probs, last_x, last_y)

    def _publish_best_path(self, trajectories, mode_probs):
        """发布最高概率模态轨迹为 nav_msgs/Path（odom 系）。

        避障节点订阅此话题做碰撞检测，接口比 MarkerArray 明确（无需解析概率标签）。
        """
        best_idx = int(np.argmax(mode_probs))
        path = Path()
        path.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        for p in trajectories[best_idx]:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.best_path_pub.publish(path)

    def _publish_markers(self, trajectories, mode_probs, origin_x, origin_y):
        """创建 RViz MarkerArray 可视化。"""
        marker_array = MarkerArray()

        # 观测轨迹（历史）
        hist_marker = Marker()
        hist_marker.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        hist_marker.ns = "history"
        hist_marker.id = 0
        hist_marker.type = Marker.LINE_STRIP
        hist_marker.action = Marker.ADD
        hist_marker.scale.x = 0.03
        hist_marker.color = ColorRGBA(0.2, 0.4, 0.96, 0.9)
        hist_marker.points = [
            Point(x=p[0], y=p[1]) for p in self.history
        ]
        marker_array.markers.append(hist_marker)

        # 5 条预测轨迹
        sorted_idx = np.argsort(mode_probs)[::-1]  # 按概率降序
        for rank, k in enumerate(sorted_idx):
            traj = trajectories[k]
            prob = mode_probs[k]

            marker = Marker()
            marker.header = Header(stamp=rospy.Time.now(), frame_id="odom")
            marker.ns = "prediction"
            marker.id = k + 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color = MODE_COLORS[k]
            marker.points = [Point(x=p[0], y=p[1]) for p in traj]

            # 文本标签（显示概率）
            marker.text = f"K{k+1}: {prob*100:.0f}%"

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


# ── 入口 ────────────────────────────────────────────
if __name__ == "__main__":
    try:
        predictor = TrajectoryPredictor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
