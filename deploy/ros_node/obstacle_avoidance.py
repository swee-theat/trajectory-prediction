#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 避障节点：深度图感知 + 预测轨迹 → 碰撞检测 → 减速绕行 → cmd_vel。

实现需求文档预期成果④「感知→预测→决策→执行」闭环：
    感知: 订阅 Orbbec 深度图 /camera/depth_registered/image_raw（单位 mm）
    预测: 订阅轨迹预测节点输出的 /predicted_best_path（最高概率模态，odom 系）
    决策: 深度图 → 内参反投影 3D 点云 → tf 转到 base_footprint → 地面分割 →
          障碍物占据栅格；碰撞检测 = 预测轨迹(弯曲路径) + 直行走廊(覆盖
          1.5m 盲区) 取更近威胁；若有碰撞 → 减速(vx↓) + 绕行(横向 vy)
    执行: 发布 /cmd_vel（麦克纳姆轮三自由度 vx/vy/ω）

低速场景口径（手册 14.3 节）：
    - 车导航速度 vx ∈ [0.015, 0.12] m/s，vy ∈ [-0.1, 0.1]，ω ∈ [0.2, 0.5] rad/s
    - 障碍物（锥桶/围挡静态 + 低速动态）移动速度与车同量级（~0.1 m/s），
      深度图 30Hz 实时检测即可持续跟上，无需障碍物轨迹预测。

核心逻辑（depth_to_points / points_to_obstacle_grid / check_collision /
check_corridor / AvoidanceStateMachine）为不依赖 rospy 的纯逻辑，可在无 ROS
环境下离线单元测试（见 deploy/test/test_obstacle_avoidance.py）。

用法:
    rosrun ai_node_master obstacle_avoidance.py
    或 roslaunch ai_node_master obstacle_avoidance.launch
"""

import rospy
import numpy as np
import tf
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty


# ── 纯逻辑（不依赖 rospy，可离线单元测试）──────────────────────────
# 抽取至 avoidance_core.py：实车节点 / 离线单测 / Web demo 共用同一份逻辑
from avoidance_core import (
    decode_depth,
    depth_to_points,
    quaternion_rotation_matrix,
    transform_points,
    points_to_obstacle_grid,
    check_collision,
    check_corridor,
    merge_collision,
    side_occupancy,
    AvoidanceStateMachine,
)


# ── ROS 节点 ───────────────────────────────────────────────────────

class ObstacleAvoidanceNode:
    """订阅深度图 + 相机内参 + 预测轨迹，定时做碰撞检测并发布 cmd_vel。"""

    def __init__(self):
        rospy.init_node("obstacle_avoidance", anonymous=False)

        # 话题（均可被 launch 覆盖）
        self.depth_topic = rospy.get_param(
            "~depth_topic", "/camera/depth_registered/image_raw")
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/depth/camera_info")
        self.best_path_topic = rospy.get_param(
            "~best_path_topic", "/predicted_best_path")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")

        # 深度处理参数
        self.step = rospy.get_param("~depth_step", 4)
        self.max_range = rospy.get_param("~depth_max_range", 5.0)
        self.ground_thresh = rospy.get_param("~ground_thresh", 0.08)

        # 栅格参数（x_min = 车体中心到车头距离，排除车头自身深度点误判为障碍物）
        self.grid_res = rospy.get_param("~grid_res", 0.05)
        self.x_min = rospy.get_param("~x_min", 0.15)
        self.x_max = rospy.get_param("~x_max", 3.0)
        self.y_min = rospy.get_param("~y_min", -1.5)
        self.y_max = rospy.get_param("~y_max", 1.5)

        # 避障决策参数（低速场景，手册 14.3 节）
        # margin = 横向安全余量 = 车半宽(约0.10m) + 余量(0.05m)，同时作
        # check_collision 的轨迹膨胀半径与 check_corridor 的走廊半宽（两处同一口径）
        self.margin = rospy.get_param("~margin", 0.15)
        self.max_vx = rospy.get_param("~max_vx", 0.11)
        self.slow_start = rospy.get_param("~slow_start", 1.5)
        self.safe_stop = rospy.get_param("~safe_stop", 0.4)
        self.lat_vy = rospy.get_param("~lat_vy", 0.05)
        self.decide_hz = rospy.get_param("~decide_hz", 10.0)
        self.hyst = rospy.get_param("~hyst", 0.2)
        self.side_thresh = rospy.get_param("~side_thresh", 0.05)
        self.clear_frames = rospy.get_param("~clear_frames", 5)

        # 里程自停（M3 演示安全兜底）：累计行驶里程超限锁死，reset 话题归零
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.reset_topic = rospy.get_param("~reset_topic", "/avoidance_reset")
        self.max_distance = rospy.get_param("~max_distance", 5.0)
        self.total_distance = 0.0
        self.last_odom_xy = None
        self.distance_locked = False

        # 决策状态机（3 状态 + 滞回 + N 帧清空防抖 + 左右选向锁定）
        self.sm = AvoidanceStateMachine(
            self.max_vx, self.slow_start, self.safe_stop, self.lat_vy,
            hyst=self.hyst, side_thresh=self.side_thresh,
            clear_frames=self.clear_frames)

        # 最新数据缓存（深度 30Hz / 预测 10Hz / 决策 10Hz 解耦）
        self.latest_depth = None
        self.latest_depth_frame = None
        self.latest_K = None
        self.latest_path = None

        self.tf_listener = tf.TransformListener()

        rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo,
                         self._caminfo_cb, queue_size=1)
        rospy.Subscriber(self.best_path_topic, Path, self._path_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber(self.reset_topic, Empty, self._reset_cb, queue_size=1)

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=2)

        # M4 关停钩子：节点关闭时发布停车，防止底盘保持最后指令继续跑
        rospy.on_shutdown(self._shutdown_hook)

        # 决策循环（独立于传感器回调，用最新缓存数据）
        rospy.Timer(rospy.Duration(1.0 / self.decide_hz), self._decide_tick)

        rospy.loginfo("避障节点已启动")
        rospy.loginfo(f"  深度图: {self.depth_topic} | 相机内参: {self.camera_info_topic}")
        rospy.loginfo(f"  预测轨迹: {self.best_path_topic} | 输出: {self.cmd_vel_topic}")
        rospy.loginfo(f"  决策: {self.decide_hz}Hz | 减速区间 [{self.safe_stop}, {self.slow_start}]m")

    # ── 回调：只缓存最新数据 ──────────────────────────────────────
    def _depth_cb(self, msg):
        self.latest_depth = decode_depth(msg)
        self.latest_depth_frame = msg.header.frame_id

    def _caminfo_cb(self, msg):
        # CameraInfo.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.latest_K = (msg.K[0], msg.K[4], msg.K[2], msg.K[5])

    def _path_cb(self, msg):
        self.latest_path = msg

    def _odom_cb(self, msg):
        """里程计：累加行驶距离（相邻位置差）。"""
        xy = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        if self.last_odom_xy is not None:
            self.total_distance += float(np.linalg.norm(xy - self.last_odom_xy))
        self.last_odom_xy = xy

    def _reset_cb(self, msg):
        """M3 里程自停归零：收到 reset 消息清里程、解锁恢复行驶。"""
        self.total_distance = 0.0
        self.distance_locked = False
        rospy.loginfo("里程已归零，恢复行驶")

    def _shutdown_hook(self):
        """M4 关停钩子：节点关闭时发布停车。"""
        try:
            self._publish_cmd(0.0, 0.0, 0.0)
            rospy.loginfo("避障节点关闭，已发布停车")
        except Exception:
            pass  # 关闭时发布失败不阻塞退出

    # ── 决策循环 ──────────────────────────────────────────────────
    def _decide_tick(self, _event):
        # 数据未就绪则停车（安全）
        if self.latest_depth is None or self.latest_K is None \
                or self.latest_path is None or self.latest_depth_frame is None:
            self._publish_cmd(0.0, 0.0, 0.0)
            return

        # M3 里程自停兜底：累计里程超限则锁死停车，reset 归零才解锁
        if self.total_distance >= self.max_distance:
            self.distance_locked = True
        if self.distance_locked:
            self._publish_cmd(0.0, 0.0, 0.0)
            return

        try:
            # 1. 深度图 → 相机系点云
            pts_cam = depth_to_points(
                self.latest_depth, self.latest_K, self.step, self.max_range)

            # 2. 相机系 → base_footprint（tf）
            trans, rot = self.tf_listener.lookupTransform(
                self.base_frame, self.latest_depth_frame, rospy.Time(0))
            pts_bf = transform_points(pts_cam, trans, rot)

            # 3. 障碍物占据栅格
            grid, x_min, y_min = points_to_obstacle_grid(
                pts_bf, self.ground_thresh, self.grid_res,
                (self.x_min, self.x_max), (self.y_min, self.y_max))

            # 4. 预测轨迹 odom → base_footprint（tf）
            traj_bf = self._path_to_base(self.latest_path)

            # 5. 碰撞检测：预测轨迹（弯曲路径）+ 直行走廊（覆盖 slow_start 盲区）
            d_col_traj, obs_y_traj = check_collision(
                traj_bf, grid, self.grid_res, x_min, y_min, self.margin)
            d_col_corr, obs_y_corr = check_corridor(
                grid, self.grid_res, x_min, y_min,
                self.slow_start, self.margin)
            d_col, obs_y = merge_collision(
                d_col_traj, obs_y_traj, d_col_corr, obs_y_corr)

            # 6. 左右半区占据统计（绕行选向）
            left_count, right_count = side_occupancy(
                grid, self.grid_res, x_min, y_min, self.slow_start)

            # 7. 决策状态机（带滞回 + N 帧防抖 + 左右选向锁定）+ 发布
            vx, vy, w = self.sm.step(d_col, obs_y, left_count, right_count)
            self._publish_cmd(vx, vy, w)

        except Exception as e:
            # 任何异常（tf / 深度解码 / 数值运算）都停车降级，不让节点崩溃
            rospy.logwarn_throttle(2.0, f"决策异常，停车: {e}")
            self._publish_cmd(0.0, 0.0, 0.0)

    def _path_to_base(self, path_msg):
        """nav_msgs/Path（odom 系）→ base_footprint 系 (N,2) 轨迹点。"""
        if len(path_msg.poses) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        src_frame = path_msg.header.frame_id or "odom"
        trans, rot = self.tf_listener.lookupTransform(
            self.base_frame, src_frame, rospy.Time(0))
        pts = np.array([[p.pose.position.x, p.pose.position.y, 0.0]
                        for p in path_msg.poses], dtype=np.float32)
        pts_bf = transform_points(pts, trans, rot)
        return pts_bf[:, :2]

    def _publish_cmd(self, vx, vy, w):
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = w
        self.cmd_pub.publish(twist)


if __name__ == "__main__":
    try:
        node = ObstacleAvoidanceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
