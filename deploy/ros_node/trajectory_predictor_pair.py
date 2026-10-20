#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 两车交互轨迹预测节点：主车 /odom + 第二车 /odom → 交互推理 → 双车预测 + 碰撞预警。

数据流:
    主车 /odom (20Hz)  → 降采样 10Hz → ego 历史 20 帧
    第二车 /odom (20Hz) → 降采样 10Hz + offset 坐标变换 → 邻居历史 20 帧
    → ego 特征（居中到 ego 末帧）+ 邻居特征（居中到 ego 末帧，同坐标系）
    → 交互 ONNX 推理（history + neighbor_history + neighbor_mask + city_id）
    → ego 5 模态 + 邻居 5 模态 + 碰撞预警（best mode 逐帧距离 < collision_margin）

与单车版 trajectory_predictor.py 的关键差异:
    1. 输入增加 neighbor_history / neighbor_mask / city_id，输出增加邻居 5 模态
    2. neighbor_timeout fail-safe：第二车关机/WiFi 断/odom 停更 → 喂全 0 邻居
       → ego 本就不受邻居影响（ego_feat=encoded 不经过 interaction），邻居输出为
       垃圾 → 节点隐藏邻居轨迹并标注"邻居离线"（与避障节点 odom_timeout 同类设计）
    3. collision_margin 默认 0.6m（实车 0.11m/s 口径，非 demo 的 2m 口径）

特征口径与 demo/app.py 的 compute_features 逐维一致（这是 5.26m 邻居模型的前提）:
    [x, y, vx, vy, t/T, 1, 0, 0, 0, heading, curvature]
    heading = arctan2(vy, vx)（非里程计 yaw）、曲率 = Δheading/0.1 裁剪 ±3、社交特征 [6:9] 恒 0

实体小车环境:
    Intel NUC, Ubuntu 20.04, ROS Noetic, onnxruntime, 麦克纳姆轮底盘（速度 ≤0.12 m/s）

用法:
    rosrun ai_node_master trajectory_predictor_pair.py
    或 roslaunch ai_node_master trajectory_predictor_pair.launch
"""

import math
import numpy as np
import onnxruntime as ort
import rospy
from collections import deque
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA, Header

# ── 配置（与训练参数保持一致，不要动）──────────────────
OBS_LEN = 20          # 观测帧数 (2s @ 10Hz)
PRED_LEN = 30         # 预测帧数 (3s @ 10Hz)
INPUT_DIM = 11        # 特征维度
NUM_MODES = 5         # K 模态数
TARGET_FREQ = 10.0    # 采样+预测频率，训练数据 10Hz，odom 20Hz 必须降采样
DT = 0.1              # 相邻帧时间间隔（与训练一致）

# ego 模态颜色（与单车版一致）
MODE_COLORS = [
    ColorRGBA(0.94, 0.27, 0.27, 0.8),  # 红
    ColorRGBA(0.08, 0.72, 0.65, 0.8),  # 青
    ColorRGBA(0.55, 0.24, 0.96, 0.8),  # 紫
    ColorRGBA(0.20, 0.60, 0.96, 0.8),  # 蓝
    ColorRGBA(0.96, 0.60, 0.15, 0.8),  # 橙
]
# 邻居模态颜色（绿/棕系，与 ego 区分）
NEIGHBOR_MODE_COLORS = [
    ColorRGBA(0.30, 0.80, 0.35, 0.8),  # 绿
    ColorRGBA(0.60, 0.40, 0.15, 0.8),  # 棕
    ColorRGBA(0.90, 0.30, 0.70, 0.8),  # 粉
    ColorRGBA(0.00, 0.55, 0.55, 0.8),  # 深青
    ColorRGBA(0.50, 0.50, 0.90, 0.8),  # 蓝紫
]


class TrajectoryPredictorPair:
    """订阅主车与第二车里程计、运行交互 ONNX 推理、发布双车预测与碰撞预警。"""

    def __init__(self):
        rospy.init_node("trajectory_predictor_pair", anonymous=False)

        # ── 参数 ──────────────────────────────────────
        self.model_path = rospy.get_param(
            "~model_path", "deploy/trajectory_model_interaction.onnx"
        )
        self.neighbor_odom_topic = rospy.get_param("~neighbor_odom_topic", "/car2/odom")
        self.city_id = rospy.get_param("~city_id", 2)  # 与 demo 一致，固定 MIA 城
        # 碰撞阈值：实车 0.11m/s 口径，0.6m 约 5.5s 反应余量；非 demo 的 2m（2-3m/s 场景）。
        # 两车 0.11m/s 下相距 2m 内是常态，用 2m 会恒红预警。
        self.collision_margin = rospy.get_param("~collision_margin", 0.6)
        # 邻居失效回退阈值：超过 N 秒未收到邻居 odom → mask [[0,0]] 退单车 + 标注离线
        self.neighbor_timeout = rospy.get_param("~neighbor_timeout", 0.5)
        # 邻居坐标系偏移 [dx, dy, dyaw]：第二车 odom 原点在主车 odom 系下的位姿。
        # 演示流程：先摆位 → 再上电（两车 odom 各自清零）→ 再标 offset，否则每次重启重标。
        self.neighbor_offset = rospy.get_param("~neighbor_offset", [0.0, 0.0, 0.0])

        # ── ONNX 会话 ─────────────────────────────────
        self.session = ort.InferenceSession(self.model_path)
        self.output_names = [o.name for o in self.session.get_outputs()]

        # ── 历史缓冲区（@10Hz 采样）───────────────────
        self.ego_history = deque(maxlen=OBS_LEN)       # [x, y]
        self.neighbor_history = deque(maxlen=OBS_LEN)  # [x, y]（已变换到主车系）
        # 降采样时间戳：用各自 msg.header.stamp 判断（两车时钟无需同步，相位差可忽略）
        self._last_ego_sample = rospy.Time(0)
        self._last_neighbor_sample = rospy.Time(0)
        # 邻居收到时间（本地时钟，用于 timeout 判断，不依赖两车时钟同步）
        self._last_neighbor_receive = rospy.Time(0)
        self.neighbor_offline = True  # 初始视为离线，收到 odom 后置 False

        # ── 订阅 ──────────────────────────────────────
        rospy.Subscriber("/odom", Odometry, self._ego_odom_cb)
        rospy.Subscriber(self.neighbor_odom_topic, Odometry, self._neighbor_odom_cb)

        # ── 发布 ──────────────────────────────────────
        self.marker_pub = rospy.Publisher(
            "/predicted_trajectories", MarkerArray, queue_size=2
        )
        self.neighbor_marker_pub = rospy.Publisher(
            "/neighbor_predicted_trajectories", MarkerArray, queue_size=2
        )
        self.best_path_pub = rospy.Publisher(
            "/predicted_best_path", Path, queue_size=2
        )
        self.neighbor_best_pub = rospy.Publisher(
            "/neighbor_predicted_best_path", Path, queue_size=2
        )

        rospy.loginfo("两车交互轨迹预测节点已启动")
        rospy.loginfo(f"  ONNX 模型: {self.model_path}")
        rospy.loginfo(f"  邻居 odom 话题: {self.neighbor_odom_topic}")
        rospy.loginfo(f"  city_id={self.city_id}, 碰撞阈值={self.collision_margin}m, "
                      f"邻居超时={self.neighbor_timeout}s, offset={self.neighbor_offset}")

    # ── 邻居坐标变换 ─────────────────────────────────
    def _apply_offset(self, x, y):
        """把第二车 odom 坐标转换到主车 odom 系（平移 + 旋转）。

        邻居特征里的 heading 由位置差分 arctan2(vy,vx) 算出，故只需旋转位置，
        无需单独旋转朝向。
        """
        dx, dy, dyaw = self.neighbor_offset
        cos_y, sin_y = math.cos(dyaw), math.sin(dyaw)
        return (dx + cos_y * x - sin_y * y, dy + sin_y * x + cos_y * y)

    # ── 主车里程计回调 ───────────────────────────────
    def _ego_odom_cb(self, msg: Odometry):
        now = msg.header.stamp
        if (now - self._last_ego_sample).to_sec() < (1.0 / TARGET_FREQ):
            return  # 距上次采样不足 0.1s，丢弃
        self._last_ego_sample = now

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.ego_history.append((x, y))

        if len(self.ego_history) < OBS_LEN:
            return
        self._predict_and_publish()

    # ── 第二车里程计回调 ─────────────────────────────
    def _neighbor_odom_cb(self, msg: Odometry):
        now = msg.header.stamp
        if (now - self._last_neighbor_sample).to_sec() < (1.0 / TARGET_FREQ):
            return
        self._last_neighbor_sample = now

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        x, y = self._apply_offset(x, y)
        self.neighbor_history.append((x, y))
        self._last_neighbor_receive = rospy.Time.now()
        self.neighbor_offline = False

    # ── 邻居是否超时 ─────────────────────────────────
    def _neighbor_timeout_reached(self):
        if not self.neighbor_history:
            return True
        return (rospy.Time.now() - self._last_neighbor_receive).to_sec() > self.neighbor_timeout

    # ── 11 维特征（逐维对照 demo/app.py compute_features）──
    def _compute_features(self, xs_c: np.ndarray, ys_c: np.ndarray) -> np.ndarray:
        """以（某末帧）为原点居中后的坐标 → 11 维特征。

        与 demo/app.py compute_features 逐维一致：
        [x, y, vx, vy, t/T, 1, dist, rel_x, rel_y, heading, curvature]
        其中社交特征 dist/rel_x/rel_y（[6:9]）恒 0（实体小车无多智能体检测）。
        """
        T = len(xs_c)
        feats = np.zeros((T, INPUT_DIM), dtype=np.float32)

        vx = np.zeros(T, dtype=np.float32)
        vy = np.zeros(T, dtype=np.float32)
        vx[1:] = (xs_c[1:] - xs_c[:-1]) / DT
        vy[1:] = (ys_c[1:] - ys_c[:-1]) / DT
        vx[0] = vx[1]
        vy[0] = vy[1]

        heading = np.unwrap(np.arctan2(vy, vx))
        curvature = np.zeros(T, dtype=np.float32)
        curvature[1:] = (heading[1:] - heading[:-1]) / DT
        curvature[0] = curvature[1]
        curvature = np.clip(curvature, -3.0, 3.0)

        for i in range(T):
            feats[i] = [
                xs_c[i],                    # 0: x（以末帧为原点）
                ys_c[i],                    # 1: y
                vx[i],                      # 2: vx
                vy[i],                      # 3: vy
                i / (OBS_LEN - 1),          # 4: t/T（0 → 1.0）
                1.0,                        # 5: 常值偏置
                0.0,                        # 6: 最近邻距离（无检测，恒 0）
                0.0,                        # 7: 相对最近邻 x（恒 0）
                0.0,                        # 8: 相对最近邻 y（恒 0）
                heading[i],                 # 9: 航向角（rad）
                curvature[i],               # 10: 曲率（rad/s）
            ]
        return feats

    def _extract_ego_features(self) -> np.ndarray:
        """ego 特征：以 ego 末帧居中。"""
        data = np.array(self.ego_history, dtype=np.float32)  # [20, 2]
        xs_c = data[:, 0] - data[-1, 0]
        ys_c = data[:, 1] - data[-1, 1]
        return self._compute_features(xs_c, ys_c)  # [20, 11]

    def _extract_neighbor_features(self) -> np.ndarray:
        """邻居特征：以 ego 末帧居中（非邻居自己末帧，保证同坐标系）。"""
        data = np.array(self.neighbor_history, dtype=np.float32)  # [20, 2]
        ego_last = np.array(self.ego_history[-1], dtype=np.float32)  # [2]
        xs_c = data[:, 0] - ego_last[0]
        ys_c = data[:, 1] - ego_last[1]
        return self._compute_features(xs_c, ys_c)  # [20, 11]

    # ── 推理 + 发布 ─────────────────────────────────
    def _predict_and_publish(self):
        ego_feats = self._extract_ego_features()  # [20, 11]

        # 邻居在线判定：未超时 且 攒满 20 帧（与 demo 的 has_neighbor 口径一致）
        neighbor_online = (not self._neighbor_timeout_reached()) and (
            len(self.neighbor_history) >= OBS_LEN
        )

        neighbor_history = np.zeros((1, 2, OBS_LEN, INPUT_DIM), dtype=np.float32)
        if neighbor_online:
            neighbor_history[0, 0] = self._extract_neighbor_features()
        else:
            # 邻居失效回退：喂全 0。ego 预测不受影响（ego_feat=encoded 不经过
            # interaction），邻居输出基于全 0 输入是垃圾 → 下方跳过邻居发布 + 标注离线。
            # 注：neighbor_mask 对模型输出本就无影响（导出时被 dead-code 消除，ONNX
            # 无此输入），回退靠「喂全 0 + 节点隐藏邻居」实现，而非 mask。
            self.neighbor_offline = True

        city_id = np.array([self.city_id], dtype=np.int64)

        inputs = {
            "history": ego_feats[np.newaxis, :, :],
            "neighbor_history": neighbor_history,
            "city_id": city_id,
        }
        outs = dict(zip(self.output_names, self.session.run(None, inputs)))

        ego_traj = outs["trajectories"][0]        # [K, 30, 2] 相对 ego 末帧位移
        ego_probs = outs["mode_probs"][0]         # [K]

        ego_last = np.array(self.ego_history[-1], dtype=np.float32)  # [2]
        abs_ego = ego_traj + ego_last            # [K, 30, 2] 还原主车 odom 系

        # ── ego 发布 ──────────────────────────────────
        self._publish_ego_markers(abs_ego, ego_probs, ego_last)
        self._publish_best_path(self.best_path_pub, abs_ego, ego_probs)

        # ── 邻居发布 + 碰撞预警 ───────────────────────
        if neighbor_online and "neighbor_trajectories" in outs:
            nbr_traj = outs["neighbor_trajectories"][0]   # [K, 30, 2]
            nbr_probs = outs["neighbor_mode_probs"][0]    # [K]
            abs_nbr = nbr_traj + ego_last                # 邻居也相对 ego 末帧

            self._publish_neighbor_markers(abs_nbr, nbr_probs)
            self._publish_best_path(self.neighbor_best_pub, abs_nbr, nbr_probs)

            # 碰撞预警：best mode 两车逐帧距离 < margin
            ego_best = abs_ego[int(np.argmax(ego_probs))]
            nbr_best = abs_nbr[int(np.argmax(nbr_probs))]
            dmin_frame = np.linalg.norm(ego_best - nbr_best, axis=1)
            dmin = float(dmin_frame.min())
            risk_frames = [int(t) for t in range(PRED_LEN) if dmin_frame[t] < self.collision_margin]
            if risk_frames:
                rospy.logwarn_throttle(
                    1.0,
                    f"碰撞预警：未来最小距离 {dmin:.2f}m < {self.collision_margin}m，"
                    f"危险帧 {risk_frames[0]}~{risk_frames[-1]}",
                )
        else:
            # 邻居离线：发布"邻居离线"文字标记，并确保不残留邻居轨迹
            self._publish_neighbor_offline_marker(ego_last)
            rospy.logwarn_throttle(3.0, "邻居离线（超时/未攒满 20 帧），退化为单车预测")

    # ── ego 可视化 ───────────────────────────────────
    def _publish_ego_markers(self, abs_ego, ego_probs, ego_last):
        marker_array = MarkerArray()

        # 历史轨迹（蓝线）
        hist = Marker()
        hist.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        hist.ns = "history"
        hist.id = 0
        hist.type = Marker.LINE_STRIP
        hist.action = Marker.ADD
        hist.scale.x = 0.03
        hist.color = ColorRGBA(0.2, 0.4, 0.96, 0.9)
        hist.points = [Point(x=p[0], y=p[1]) for p in self.ego_history]
        marker_array.markers.append(hist)

        # 5 条预测轨迹
        sorted_idx = np.argsort(ego_probs)[::-1]
        for rank, k in enumerate(sorted_idx):
            traj = abs_ego[k]
            prob = ego_probs[k]
            marker = Marker()
            marker.header = Header(stamp=rospy.Time.now(), frame_id="odom")
            marker.ns = "prediction"
            marker.id = k + 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color = MODE_COLORS[k]
            marker.text = f"K{k+1}: {prob*100:.0f}%"
            marker.points = [Point(x=p[0], y=p[1]) for p in traj]
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    # ── 邻居可视化 ───────────────────────────────────
    def _publish_neighbor_markers(self, abs_nbr, nbr_probs):
        marker_array = MarkerArray()

        # 邻居历史（红线）
        hist = Marker()
        hist.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        hist.ns = "neighbor_history"
        hist.id = 0
        hist.type = Marker.LINE_STRIP
        hist.action = Marker.ADD
        hist.scale.x = 0.03
        hist.color = ColorRGBA(0.94, 0.27, 0.27, 0.9)
        hist.points = [Point(x=p[0], y=p[1]) for p in self.neighbor_history]
        marker_array.markers.append(hist)

        # 邻居 5 条预测轨迹
        sorted_idx = np.argsort(nbr_probs)[::-1]
        for rank, k in enumerate(sorted_idx):
            traj = abs_nbr[k]
            prob = nbr_probs[k]
            marker = Marker()
            marker.header = Header(stamp=rospy.Time.now(), frame_id="odom")
            marker.ns = "neighbor_prediction"
            marker.id = k + 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color = NEIGHBOR_MODE_COLORS[k]
            marker.text = f"邻K{k+1}: {prob*100:.0f}%"
            marker.points = [Point(x=p[0], y=p[1]) for p in traj]
            marker_array.markers.append(marker)

        self.neighbor_marker_pub.publish(marker_array)

    # ── 邻居离线标注 ─────────────────────────────────
    def _publish_neighbor_offline_marker(self, ego_last):
        marker_array = MarkerArray()
        text = Marker()
        text.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        text.ns = "neighbor_status"
        text.id = 0
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.scale.z = 0.15
        text.color = ColorRGBA(0.94, 0.27, 0.27, 1.0)
        text.text = "邻居离线"
        text.pose.position.x = float(ego_last[0])
        text.pose.position.y = float(ego_last[1]) + 0.3
        text.pose.orientation.w = 1.0
        marker_array.markers.append(text)
        self.neighbor_marker_pub.publish(marker_array)

    # ── 发布 best 模态路径 ───────────────────────────
    def _publish_best_path(self, publisher, abs_traj, probs):
        best_idx = int(np.argmax(probs))
        path = Path()
        path.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        for p in abs_traj[best_idx]:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        publisher.publish(path)


# ── 入口 ────────────────────────────────────────────
if __name__ == "__main__":
    try:
        predictor = TrajectoryPredictorPair()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
