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

def decode_depth(img_msg):
    """sensor_msgs/Image → 深度图（米，float32）。

    支持 Orbbec astra_camera 常见的 16UC1（mm）与 32FC1（m）两种编码。
    """
    if img_msg.encoding == "16UC1":
        arr = np.frombuffer(img_msg.data, dtype=np.uint16).reshape(
            img_msg.height, img_msg.width
        )
        return arr.astype(np.float32) / 1000.0  # mm → m
    if img_msg.encoding == "32FC1":
        return np.frombuffer(img_msg.data, dtype=np.float32).reshape(
            img_msg.height, img_msg.width
        )
    raise ValueError(f"不支持的深度编码: {img_msg.encoding}")


def depth_to_points(depth_m, K, step=4, max_range=5.0):
    """深度图（米）→ 相机系 3D 点云 (N,3)，numpy 向量化。

    参数:
        depth_m: [H, W] 深度图（米）
        K: 内参 (fx, fy, cx, cy)
        step: 像素下采样步长（控制点数，640×480 下 step=4 → ~1.9 万点）
        max_range: 有效深度上限（米），超出视为无效（噪声/无穷远）
    """
    H, W = depth_m.shape
    fx, fy, cx, cy = K
    v = np.arange(0, H, step)
    u = np.arange(0, W, step)
    vv, uu = np.meshgrid(v, u, indexing="ij")

    d = depth_m[vv, uu]
    valid = (d > 0.01) & (d < max_range) & np.isfinite(d)
    d = d[valid]
    uu = uu[valid]
    vv = vv[valid]

    # 针孔模型反投影（相机系：x 右、y 下、z 前）
    X = (uu - cx) * d / fx
    Y = (vv - cy) * d / fy
    Z = d
    return np.stack([X, Y, Z], axis=1).astype(np.float32)


def quaternion_rotation_matrix(q):
    """四元数 [x, y, z, w] → 3×3 旋转矩阵（ROS geometry_msgs 顺序）。"""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def transform_points(points, trans, rot_xyzw):
    """点云坐标变换：points(N,3) → R·p + t。"""
    R = quaternion_rotation_matrix(rot_xyzw)
    t = np.array(trans, dtype=np.float32)
    return points @ R.T + t


def points_to_obstacle_grid(points_bf, ground_thresh, grid_res, x_range, y_range):
    """base_footprint 系点云 → 障碍物 2D 占据栅格。

    地面分割：z > ground_thresh 的点视为障碍物（锥桶竖立、高于地面）。
    栅格 grid[i, j] 对应坐标 (x_min + j*res, y_min + i*res)。

    返回:
        (grid [ny,nx] bool, x_min, y_min)
    """
    obs = points_bf[points_bf[:, 2] > ground_thresh]
    x_min, x_max = x_range
    y_min, y_max = y_range
    nx = int((x_max - x_min) / grid_res)
    ny = int((y_max - y_min) / grid_res)
    grid = np.zeros((ny, nx), dtype=bool)
    if len(obs) == 0:
        return grid, x_min, y_min

    xi = ((obs[:, 0] - x_min) / grid_res).astype(np.int32)
    yi = ((obs[:, 1] - y_min) / grid_res).astype(np.int32)
    mask = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    grid[yi[mask], xi[mask]] = True
    return grid, x_min, y_min


def check_collision(traj_xy, obstacle_grid, grid_res, x_min, y_min, margin):
    """预测轨迹(N,2) → 最近碰撞点的累计弧长距离（米）。

    逐帧（沿轨迹顺序）检查轨迹点是否落入障碍物栅格（带 margin 膨胀，
    margin = 车半宽+余量，与 check_corridor 的走廊半宽同一口径），
    返回第一个碰撞点的累计弧长；无碰撞返回 (None, 0.0)。

    返回:
        (d_col, obs_y)：d_col 为碰撞点累计弧长（无碰撞则 None），
                        obs_y 为碰撞点横向坐标（决定绕行方向）。
    """
    if len(traj_xy) == 0 or not obstacle_grid.any():
        return None, 0.0

    ny, nx = obstacle_grid.shape
    margin_cells = int(np.ceil(margin / grid_res))
    cum = 0.0
    prev = traj_xy[0]
    for p in traj_xy[1:]:
        cum += float(np.linalg.norm(p - prev))
        prev = p
        xi = int((p[0] - x_min) / grid_res)
        yi = int((p[1] - y_min) / grid_res)
        x0 = max(0, xi - margin_cells)
        x1 = min(nx, xi + margin_cells + 1)
        y0 = max(0, yi - margin_cells)
        y1 = min(ny, yi + margin_cells + 1)
        if obstacle_grid[y0:y1, x0:x1].any():
            return cum, float(p[1])
    return None, 0.0


def check_corridor(obstacle_grid, grid_res, x_min, y_min,
                   corridor_len, corridor_half_width):
    """车体正前方「名义直行走廊」碰撞检测（不依赖预测轨迹长度）。

    走廊: base_footprint 系 x∈[0, corridor_len]、|y| < corridor_half_width 的矩形。
    corridor_half_width = 车半宽+余量（与 check_collision 的 margin 同一口径）。
    直接查栅格内最近占据格，覆盖到 slow_start 距离。

    为什么需要: 预测轨迹受模型输出限制（低速 0.11m/s × 3s ≈ 0.33m），
    check_collision 沿轨迹点查够不到 1.5m 处的障碍物，减速逻辑永不触发。
    走廊补上这段盲区，模型轨迹只负责提供弯曲路径（展示价值保留）。

    返回:
        (d_col, obs_y)：d_col 为走廊内最近占据格到车头(x=0)的距离，
                        obs_y 为该最近一排占据格的横向中心 y；走廊空返回 (None, 0.0)。
    """
    if not obstacle_grid.any():
        return None, 0.0
    rows, cols = np.where(obstacle_grid)   # 所有占据格
    xc = x_min + (cols + 0.5) * grid_res   # 格中心 x
    yc = y_min + (rows + 0.5) * grid_res   # 格中心 y
    in_x = (xc >= 0.0) & (xc <= corridor_len)
    in_y = np.abs(yc) < corridor_half_width
    mask = in_x & in_y
    if not mask.any():
        return None, 0.0
    sel_x = xc[mask]
    sel_y = yc[mask]
    d_col = float(sel_x.min())
    # 最近一排（最小 x 附近的同列格）横向中心，作绕行方向依据
    near = np.abs(sel_x - d_col) < grid_res
    obs_y = float(sel_y[near].mean())
    return d_col, obs_y


def merge_collision(d_traj, y_traj, d_corr, y_corr):
    """合并预测轨迹碰撞与直行走廊碰撞，取更近的威胁作为 d_col。

    轨迹只到 ~0.33m，走廊覆盖到 slow_start(1.5m)；取两者更小 d_col，
    保证障碍物一进入减速区间即触发避障。返回 (d_col, obs_y)。
    """
    if d_corr is not None and (d_traj is None or d_corr <= d_traj):
        return d_corr, y_corr
    return d_traj, y_traj


def side_occupancy(obstacle_grid, grid_res, x_min, y_min, corridor_len):
    """统计车前方左右两半区的障碍物占据格数（用于绕行选向）。

    左半区: x∈[0, corridor_len]、y > 0
    右半区: x∈[0, corridor_len]、y < 0
    返回 (left_count, right_count)。占据少的一侧空隙更大，优先往该侧绕——
    比只看碰撞点横向位置 obs_y 更鲁棒（正对障碍物中心时也能选出更空的一侧）。
    """
    if not obstacle_grid.any():
        return 0, 0
    rows, cols = np.where(obstacle_grid)
    xc = x_min + (cols + 0.5) * grid_res
    yc = y_min + (rows + 0.5) * grid_res
    in_x = (xc >= 0.0) & (xc <= corridor_len)
    left = in_x & (yc > 0.0)
    right = in_x & (yc < 0.0)
    return int(left.sum()), int(right.sum())


class AvoidanceStateMachine:
    """避障决策状态机（3 状态 + 滞回 + N 帧清空防抖 + 左右选向锁定）。

    状态:
        FORWARD  直行（无碰撞）
        AVOIDING 避障（减速 vx↓ + 横向 vy 绕行）
        STOPPED  停车（碰撞距离过近）

    设计要点（解决无状态决策的抖动/跳变问题）:
        - 滞回: 退出避障/停车的阈值 > 进入阈值（hyst 间距），避免在边界来回震荡。
        - N 帧清空防抖: 退出避障/停车需连续 clear_frames 帧满足退出条件，避免
          深度图单帧漏检导致 d_col=None 的瞬态抖动。
        - 左右选向: 进入避障时用左右半区占据统计选空隙大的一侧（side_occupancy），
          正对障碍物中心也能选出更空的一侧，不再盲目默认右绕。
        - 方向锁定: 进入避障后绕行方向锁定不变，避免中途跳变。

    输入每帧碰撞检测结果 d_col（None=无碰撞）、障碍物横向位置 obs_y，
    与左右半区占据计数 left_count/right_count；输出 (vx, vy, w)。纯逻辑类。
    """

    FORWARD = "forward"
    AVOIDING = "avoiding"
    STOPPED = "stopped"

    def __init__(self, max_vx, slow_start, safe_stop, lat_vy,
                 hyst=0.2, side_thresh=0.05, clear_frames=5):
        self.max_vx = max_vx
        self.slow_start = slow_start
        self.safe_stop = safe_stop
        self.lat_vy = lat_vy
        self.hyst = hyst
        self.side_thresh = side_thresh
        self.clear_frames = clear_frames
        self.state = self.FORWARD
        self.avoid_sign = 0.0  # +1 向左绕 / -1 向右绕（AVOIDING 时锁定）
        self.clear_count = 0   # 连续满足退出条件的帧数（防抖动）

    def step(self, d_col, obs_y, left_count=0, right_count=0):
        """推进状态机一帧，返回 (vx, vy, w)。

        left_count/right_count 为左右半区障碍物占据格数（side_occupancy 结果），
        进入避障时据此选空隙大的一侧绕行；默认 0 时退回 obs_y 判断（向后兼容）。
        """
        # ── 状态转移（带滞回 + N 帧连续清空防抖） ──
        if self.state == self.FORWARD:
            if d_col is not None and d_col <= self.safe_stop:
                self.state = self.STOPPED
                self.avoid_sign = self._pick_sign(obs_y, left_count, right_count)
            elif d_col is not None and d_col <= self.slow_start:
                self.state = self.AVOIDING
                self.avoid_sign = self._pick_sign(obs_y, left_count, right_count)
        elif self.state == self.AVOIDING:
            if d_col is None or d_col > self.slow_start + self.hyst:
                self.clear_count += 1
                if self.clear_count >= self.clear_frames:
                    self.state = self.FORWARD
                    self.avoid_sign = 0.0
                    self.clear_count = 0
            elif d_col <= self.safe_stop:
                self.state = self.STOPPED
                self.clear_count = 0
            else:
                self.clear_count = 0
        elif self.state == self.STOPPED:
            if d_col is None or d_col > self.safe_stop + self.hyst:
                self.clear_count += 1
                if self.clear_count >= self.clear_frames:
                    self.state = self.FORWARD
                    self.avoid_sign = 0.0
                    self.clear_count = 0
            else:
                self.clear_count = 0

        # ── 输出（按转移后的状态） ──
        if self.state == self.FORWARD:
            return self.max_vx, 0.0, 0.0
        if self.state == self.STOPPED:
            # 停车但保持横向侧移（麦克纳姆轮 vx=0 仍可纯侧移），
            # 边停边挪出走廊，避免静态障碍物下的永久死锁。
            return 0.0, self.avoid_sign * self.lat_vy, 0.0

        # AVOIDING（转移后仍在此状态，则 d_col 必在 (safe_stop, slow_start+hyst]）
        if d_col is None:  # 防御：理论上转移已处理，双保险
            return self.max_vx, 0.0, 0.0
        ratio = min(1.0, (d_col - self.safe_stop) / max(1e-3, self.slow_start - self.safe_stop))
        vx = max(0.0, self.max_vx * ratio)
        vy = self.avoid_sign * self.lat_vy
        return vx, vy, 0.0

    def _pick_sign(self, obs_y, left_count=0, right_count=0):
        """绕行方向：优先左右半区占据统计选空隙大的一侧，退回 obs_y 判断兜底。

        1. 左右占据不等 → 选占据少（空隙大）的一侧绕：
           left<right（左侧空）→ 向左绕(+1)，反之向右绕(-1)。
        2. 左右占据相等（含都 0）→ 按障碍物横向位置 obs_y：
           左侧(obs_y>thresh)→向右绕(-1)，右侧→向左绕(+1)。
        3. 仍无法区分（正对中心）→ 默认向右绕，保证确定性。
        """
        if left_count != right_count:
            return 1.0 if left_count < right_count else -1.0
        if obs_y > self.side_thresh:
            return -1.0
        if obs_y < -self.side_thresh:
            return 1.0
        return -1.0  # 正对中心：默认向右绕


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
