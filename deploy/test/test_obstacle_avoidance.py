#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""obstacle_avoidance 核心逻辑离线单元测试（无需 ROS 环境）。

通过 mock rospy / tf / 消息模块，使 obstacle_avoidance.py 可被 import，
只测试其中的纯逻辑：decode_depth / depth_to_points / transform_points /
points_to_obstacle_grid / check_collision / check_corridor / AvoidanceStateMachine。

用法:
    python deploy/test/test_obstacle_avoidance.py
"""

import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "ros_node"))
sys.stdout.reconfigure(encoding="utf-8")

# ── mock ROS 依赖，使 obstacle_avoidance 可 import（只 import，不实例化节点）──
def _mock_module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m

sys.modules["rospy"] = _mock_module("rospy")
sys.modules["tf"] = _mock_module(
    "tf", TransformListener=object, LookupException=Exception,
    ConnectivityException=Exception, ExtrapolationException=Exception)
sys.modules["sensor_msgs"] = _mock_module("sensor_msgs")
sys.modules["sensor_msgs.msg"] = _mock_module(
    "sensor_msgs.msg", Image=object, CameraInfo=object)
sys.modules["nav_msgs"] = _mock_module("nav_msgs")
sys.modules["nav_msgs.msg"] = _mock_module(
    "nav_msgs.msg", Path=object, Odometry=object)
sys.modules["geometry_msgs"] = _mock_module("geometry_msgs")
sys.modules["geometry_msgs.msg"] = _mock_module("geometry_msgs.msg", Twist=object)
sys.modules["std_msgs"] = _mock_module("std_msgs")
sys.modules["std_msgs.msg"] = _mock_module("std_msgs.msg", Empty=object)

import numpy as np
import obstacle_avoidance as oa


# ── 测试用例 ──────────────────────────────────────────────────────

class _FakeImage:
    """模拟 sensor_msgs/Image，仅含解码所需字段。"""
    def __init__(self, data, encoding, height, width):
        self.data = data
        self.encoding = encoding
        self.height = height
        self.width = width


def test_decode_depth_16uc1():
    """16UC1(mm) → 米。"""
    img = _FakeImage(
        np.array([1000, 2000, 0], dtype=np.uint16).tobytes(),
        "16UC1", 1, 3)
    out = oa.decode_depth(img)
    assert out.shape == (1, 3)
    assert np.allclose(out, [[1.0, 2.0, 0.0]]), f"解码错误: {out}"
    print("  ✓ test_decode_depth_16uc1")


def test_depth_to_points():
    """针孔反投影：K=(2,2,1,1)，2×2 深度图全 2m。"""
    depth = np.full((2, 2), 2.0, dtype=np.float32)
    pts = oa.depth_to_points(depth, K=(2.0, 2.0, 1.0, 1.0), step=1, max_range=5.0)
    # 像素 (0,0): X=(0-1)*2/2=-1, Y=-1, Z=2
    # 像素 (1,1): X=0, Y=0, Z=2
    assert pts.shape == (4, 3)
    assert np.allclose(pts[0], [-1, -1, 2]), f"反投影错误: {pts[0]}"
    assert np.allclose(pts[3], [0, 0, 2]), f"反投影错误: {pts[3]}"
    print("  ✓ test_depth_to_points")


def test_transform_points():
    """平移 + 绕 z 轴旋转 90°。"""
    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    # 平移 [1,2,3]
    t = oa.transform_points(pts, trans=[1, 2, 3], rot_xyzw=[0, 0, 0, 1])
    assert np.allclose(t, [[2, 2, 3]]), f"平移错误: {t}"
    # 绕 z 轴 90°（四元数 [0,0,sin45,cos45]）：(1,0,0) → (0,1,0)
    s = np.sin(np.pi / 4)
    r = oa.transform_points(pts, trans=[0, 0, 0], rot_xyzw=[0, 0, s, s])
    assert np.allclose(r, [[0, 1, 0]], atol=1e-5), f"旋转错误: {r}"
    print("  ✓ test_transform_points")


def test_points_to_obstacle_grid():
    """地面分割：z>阈值记为障碍物，栅格化位置正确。"""
    pts = np.array([
        [1.0, 0.5, 0.20],  # 障碍物（高于地面）
        [1.0, 0.5, 0.00],  # 地面
        [2.0, -1.0, 0.15],  # 障碍物（横向 -1m）
    ], dtype=np.float32)
    grid, x_min, y_min = oa.points_to_obstacle_grid(
        pts, ground_thresh=0.08, grid_res=0.1,
        x_range=(0.0, 3.0), y_range=(-1.5, 1.5))
    assert grid.shape == (30, 30)  # 3m/0.1 × 3m/0.1
    # [1.0, 0.5] → xi=10, yi=(0.5+1.5)/0.1=20
    assert grid[20, 10], "障碍物 [1.0,0.5] 未栅格化"
    # [2.0, -1.0] → xi=20, yi=(-1.0+1.5)/0.1=5
    assert grid[5, 20], "障碍物 [2.0,-1.0] 未栅格化"
    print("  ✓ test_points_to_obstacle_grid")


def test_check_collision_hit():
    """直行轨迹 + 前方 x=1m 障碍物 → 碰撞弧长约 1m。"""
    grid = np.zeros((30, 30), dtype=bool)
    grid[:, 10] = True  # x=1.0m 整列占据
    traj = np.array([[0, 0], [0.5, 0], [1.0, 0], [1.5, 0]], dtype=np.float32)
    d_col, _ = oa.check_collision(
        traj, grid, grid_res=0.1, x_min=0.0, y_min=-1.5, margin=0.1)
    assert d_col is not None, "应检测到碰撞"
    assert abs(d_col - 1.0) < 0.1, f"碰撞距离应为 ~1m，实际 {d_col}"
    print("  ✓ test_check_collision_hit")


def test_check_collision_miss():
    """轨迹远离障碍物 → 无碰撞。"""
    grid = np.zeros((30, 30), dtype=bool)
    grid[13:18, 10] = True  # x=1.0m 处 y∈[-0.2,0.2] 的局部障碍物
    traj = np.array([[0, 1.2], [1.0, 1.2], [2.0, 1.2]], dtype=np.float32)  # y=1.2 远离
    d_col, _ = oa.check_collision(
        traj, grid, grid_res=0.1, x_min=0.0, y_min=-1.5, margin=0.1)
    assert d_col is None, "不应检测到碰撞"
    print("  ✓ test_check_collision_miss")


def test_check_corridor_hit():
    """直行走廊：前方 x=1.225m 障碍物 → d_col≈1.225，覆盖 slow_start 盲区。"""
    grid = np.zeros((60, 60), dtype=bool)  # 0.05m 分辨率，x∈[0,3]，y∈[-1.5,1.5]
    grid[30, 24] = True  # 格中心 x=(24+0.5)*0.05=1.225, y=-1.5+30.5*0.05=0.025
    d_col, obs_y = oa.check_corridor(
        grid, grid_res=0.05, x_min=0.0, y_min=-1.5,
        corridor_len=1.5, corridor_half_width=0.15)
    assert d_col is not None, "应检测到走廊障碍物"
    assert abs(d_col - 1.225) < 0.05, f"最近距离应为 ~1.225m，实际 {d_col}"
    assert abs(obs_y) < 0.05, f"障碍物横向应在中心，实际 {obs_y}"
    print("  ✓ test_check_corridor_hit")


def test_check_corridor_miss():
    """障碍物在走廊外（横向超半宽 / 前方超长度）→ 无碰撞。"""
    grid = np.zeros((60, 60), dtype=bool)
    grid[40, 24] = True  # y=0.525 > 半宽 0.15，横向在走廊外
    d_col, _ = oa.check_corridor(
        grid, grid_res=0.05, x_min=0.0, y_min=-1.5,
        corridor_len=1.5, corridor_half_width=0.15)
    assert d_col is None, "横向走廊外不应检测到"
    grid2 = np.zeros((60, 60), dtype=bool)
    grid2[30, 40] = True  # x=2.025 > 走廊长 1.5
    d_col, _ = oa.check_corridor(
        grid2, grid_res=0.05, x_min=0.0, y_min=-1.5,
        corridor_len=1.5, corridor_half_width=0.15)
    assert d_col is None, "前方超走廊长度不应检测到"
    print("  ✓ test_check_corridor_miss")


def test_corridor_covers_slow_start():
    """物理前提：轨迹只到 0.33m 够不到 1.2m 桶，走廊补上 → 触发避障。"""
    grid = np.zeros((60, 60), dtype=bool)
    grid[30, 24] = True  # 桶在 x=1.225m, y≈0
    # 低速预测轨迹（0.11m/s × 3s ≈ 0.33m）
    traj = np.array([[0, 0], [0.11, 0], [0.22, 0], [0.33, 0]], dtype=np.float32)
    d_traj, y_traj = oa.check_collision(traj, grid, 0.05, 0.0, -1.5, 0.15)
    assert d_traj is None, "0.33m 轨迹应够不到 1.2m 桶（物理盲区）"
    d_corr, y_corr = oa.check_corridor(grid, 0.05, 0.0, -1.5, 1.5, 0.15)
    assert d_corr is not None and abs(d_corr - 1.225) < 0.05, "走廊应检出 1.2m 桶"
    # 合并取更近威胁 → 1.225m → 状态机进入避障（而非全速 FORWARD）
    d_col, obs_y = oa.merge_collision(d_traj, y_traj, d_corr, y_corr)
    sm = _make_sm()
    vx, vy, _ = sm.step(d_col, obs_y)
    assert sm.state == oa.AvoidanceStateMachine.AVOIDING, "桶在 1.2m 应进入避障"
    assert vx < 0.11, f"应减速，实际 vx={vx}"
    print("  ✓ test_corridor_covers_slow_start")


def test_side_occupancy():
    """左右半区占据统计：x∈[0, corridor_len] 内按 y 正负分左右，超走廊不计。"""
    grid = np.zeros((60, 60), dtype=bool)  # 0.05m，x∈[0,3]，y∈[-1.5,1.5]
    grid[40, 20] = True  # y=0.525>0 左半区，x=1.025 ∈[0,1.5]
    grid[20, 20] = True  # y=-0.475<0 右半区，x=1.025
    grid[30, 40] = True  # y=0.025>0 左半区，但 x=2.025>1.5 超走廊，不计
    left, right = oa.side_occupancy(grid, 0.05, 0.0, -1.5, 1.5)
    assert (left, right) == (1, 1), f"左右占据应为 (1,1)，实际 ({left},{right})"
    print("  ✓ test_side_occupancy")


def _make_sm():
    return oa.AvoidanceStateMachine(
        max_vx=0.11, slow_start=1.5, safe_stop=0.4, lat_vy=0.05,
        hyst=0.2, side_thresh=0.05)


def test_state_machine_basic():
    """状态机基本状态流：直行 → 避障(减速+绕行) → 停车 → 恢复直行。"""
    sm = _make_sm()
    assert sm.state == oa.AvoidanceStateMachine.FORWARD

    # 无碰撞 → 直行
    vx, vy, w = sm.step(None, 0.0)
    assert (vx, vy, w) == (0.11, 0.0, 0.0), f"应直行: {(vx, vy, w)}"
    assert sm.state == oa.AvoidanceStateMachine.FORWARD

    # 碰撞 1.0m、障碍物在左(obs_y=0.5) → 进入避障，向右绕(-1)
    vx, vy, w = sm.step(1.0, 0.5)
    assert sm.state == oa.AvoidanceStateMachine.AVOIDING, "应进入避障状态"
    assert sm.avoid_sign == -1.0, f"障碍物在左应向右绕: sign={sm.avoid_sign}"
    assert 0 < vx < 0.11, f"应减速: vx={vx}"
    assert vy == -0.05, f"应向右绕 vy=-0.05，实际 {vy}"
    assert w == 0.0

    # 距离过近 → 停车（但保持横向侧移脱困，避免死锁）
    vx, vy, w = sm.step(0.3, 0.5)
    assert sm.state == oa.AvoidanceStateMachine.STOPPED, "应停车"
    assert vx == 0.0, "应停止前进"
    assert vy == -0.05, f"停车应保持向右横移脱困，实际 vy={vy}"
    assert w == 0.0

    # 障碍物移开 → 连续 clear_frames 帧清空后才恢复直行（防抖）
    for _ in range(5):
        vx, vy, w = sm.step(None, 0.0)
    assert sm.state == oa.AvoidanceStateMachine.FORWARD, "连续清空后应恢复直行"
    assert (vx, vy, w) == (0.11, 0.0, 0.0)
    print("  ✓ test_state_machine_basic")


def test_state_machine_side_lock():
    """绕行方向锁定：避障过程中障碍物横向位置变化不翻转方向。"""
    sm = _make_sm()
    # 障碍物在左(obs_y=0.5)进入避障 → 向右绕
    _, vy1, _ = sm.step(1.0, 0.5)
    assert vy1 == -0.05, f"初始应向右绕: vy={vy1}"
    # 避障过程中 obs_y 跳到右侧(-0.5)，方向仍锁定向右，不跳变
    _, vy2, _ = sm.step(1.0, -0.5)
    assert vy2 == -0.05, f"方向应锁定向右，实际 vy={vy2}"
    assert sm.avoid_sign == -1.0

    # 连续 clear_frames 帧清空后恢复直行，再重新进入：障碍物在右 → 向左绕
    for _ in range(5):
        sm.step(None, 0.0)
    assert sm.state == oa.AvoidanceStateMachine.FORWARD
    _, vy3, _ = sm.step(1.0, -0.5)
    assert vy3 == 0.05, f"障碍物在右应向左绕: vy={vy3}"
    print("  ✓ test_state_machine_side_lock")


def test_state_machine_side_selection():
    """左右占据选向：优先选空隙大（占据少）的一侧，优先于 obs_y 判断。"""
    sm = _make_sm()
    # 左占据多(3)、右占据少(1) → 选右侧绕(-1)，即使 obs_y 偏左(0.5)
    vx, vy, _ = sm.step(1.0, 0.5, left_count=3, right_count=1)
    assert sm.state == oa.AvoidanceStateMachine.AVOIDING
    assert sm.avoid_sign == -1.0, f"左侧占据多应向右绕，sign={sm.avoid_sign}"
    assert vy == -0.05, f"应向右绕 vy=-0.05，实际 {vy}"

    sm2 = _make_sm()
    # 右占据多(4)、左占据少(1) → 选左侧绕(+1)，即使 obs_y 偏右(-0.5)
    _, vy2, _ = sm2.step(1.0, -0.5, left_count=1, right_count=4)
    assert sm2.avoid_sign == 1.0, f"右侧占据多应向左绕，sign={sm2.avoid_sign}"
    assert vy2 == 0.05, f"应向左绕 vy=0.05，实际 {vy2}"
    print("  ✓ test_state_machine_side_selection")


def test_state_machine_hysteresis():
    """滞回：退出避障/停车需越过阈值+hyst，边界不震荡。"""
    sm = _make_sm()
    # 进入避障（1.5m 内）
    sm.step(1.4, 0.3)
    assert sm.state == oa.AvoidanceStateMachine.AVOIDING
    # 1.6m 仍 < slow_start+hyst=1.7 → 保持避障，不因单帧噪声退出
    sm.step(1.6, 0.3)
    assert sm.state == oa.AvoidanceStateMachine.AVOIDING, "滞回内应保持避障"
    # 1.8m > 1.7 → 连续 clear_frames 帧越过滞回才退出避障（防抖）
    for _ in range(5):
        sm.step(1.8, 0.3)
    assert sm.state == oa.AvoidanceStateMachine.FORWARD, "连续越过滞回应退出避障"

    # 停车后：0.5m 仍 < safe_stop+hyst=0.6 → 保持停车
    sm.step(0.3, 0.0)
    assert sm.state == oa.AvoidanceStateMachine.STOPPED
    sm.step(0.5, 0.0)
    assert sm.state == oa.AvoidanceStateMachine.STOPPED, "滞回内应保持停车"
    # 0.7m > 0.6 → 连续越过滞回才恢复
    for _ in range(5):
        sm.step(0.7, 0.0)
    assert sm.state == oa.AvoidanceStateMachine.FORWARD
    print("  ✓ test_state_machine_hysteresis")


def test_state_machine_clear_debounce():
    """N 帧连续清空防抖：单帧漏检不退出，重新检出清零，连续 clear_frames 帧才退出。"""
    sm = _make_sm()  # clear_frames 默认 5
    sm.step(1.0, 0.5)  # 进入 AVOIDING
    assert sm.state == oa.AvoidanceStateMachine.AVOIDING

    # 单帧漏检（d_col=None）→ 不退出，仍 AVOIDING
    sm.step(None, 0.0)
    assert sm.state == oa.AvoidanceStateMachine.AVOIDING, "单帧漏检不应退出避障"

    # 中断：重新检测到障碍物 → clear_count 归零
    sm.step(1.0, 0.5)
    assert sm.clear_count == 0, "重新检测到障碍物应清零防抖计数"

    # 连续 clear_frames 帧 None → 退出 FORWARD
    for _ in range(5):
        sm.step(None, 0.0)
    assert sm.state == oa.AvoidanceStateMachine.FORWARD, "连续 5 帧清空应退出避障"
    assert sm.avoid_sign == 0.0, "退出后绕行方向应归零"
    print("  ✓ test_state_machine_clear_debounce")


def main():
    print("障碍物避障核心逻辑单元测试:")
    tests = [
        test_decode_depth_16uc1,
        test_depth_to_points,
        test_transform_points,
        test_points_to_obstacle_grid,
        test_check_collision_hit,
        test_check_collision_miss,
        test_check_corridor_hit,
        test_check_corridor_miss,
        test_corridor_covers_slow_start,
        test_side_occupancy,
        test_state_machine_basic,
        test_state_machine_side_lock,
        test_state_machine_side_selection,
        test_state_machine_hysteresis,
        test_state_machine_clear_debounce,
    ]
    for t in tests:
        t()
    print(f"\n全部 {len(tests)} 项测试通过 ✅")


if __name__ == "__main__":
    main()
