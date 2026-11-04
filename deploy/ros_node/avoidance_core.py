#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""避障决策纯逻辑（不依赖 ROS，可被实车节点 / 离线单测 / Web demo 复用）。

从 obstacle_avoidance.py 抽出的无 rospy 依赖部分，只依赖 numpy：
    - 深度图解码与点云反投影（decode_depth / depth_to_points）
    - 坐标变换（quaternion_rotation_matrix / transform_points）
    - 障碍物占据栅格（points_to_obstacle_grid）
    - 碰撞检测（check_collision 预测轨迹 / check_corridor 直行走廊 / merge_collision）
    - 左右半区占据统计（side_occupancy）
    - 避障决策状态机（AvoidanceStateMachine：3 状态 + 滞回 + N 帧防抖 + 左右选向锁定）

实车节点 obstacle_avoidance.py 与本模块共用同一份逻辑；Web demo 亦 import 本模块，
用「模拟障碍物」替代真实深度相机，展示「预测 → 碰撞检测 → 减速绕行」闭环。
"""

import numpy as np


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
