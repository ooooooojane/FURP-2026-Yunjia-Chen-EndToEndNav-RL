#!/usr/bin/env python3
"""ROS1 推理节点: 激光 → CNNTD3 → 速度指令(真机部署用)

与训练严格一致的修复(F1-F7, 2026-08-17):
  F1 激光归一化: inf→7.0 再 ÷7(世界 range_max=7, prepare_state 同口径)
  F2 状态末两位: 上一步真实动作 (lin*2, (ang+1)/2), 不写死 [0,0]
  F3 速度映射:   [(a0+1)/4, a1] → 线速度 0-0.5 m/s, 角速度 ±1 rad/s
  F4 初始化竞态: 先加载模型/建发布器, 再创建订阅
  F5 看门狗:     scan/odom 超时(0.5s)或节点关闭 → 零速度
  F6 LiDAR:      <180 填充, >180 分箱取 min; 360° 雷达取前向 180 束(偏移可调)
  F7 目标停止:   距离 < goal_threshold → 持续零速度

真机安全与测量扩展(S1-S6, 2026-08-17 依 GPT 真机方案):
  S1 模型/输入异常兜底: 推理异常或状态含 NaN/Inf → 立即停车, 不保留最后速度
  S2 safety shield: 前方最近障碍 < safety_dist → 强制停车(参数可关)
  S3 控制频率对齐: 按 ~control_hz 节流(仿真步长 0.3s → 默认 3.3Hz), 与训练节奏一致
  S4 阶段限速:     ~lin_scale / ~ang_scale 缩放输出(阶段B: lin_scale=0.16 → ≤0.08 m/s)
  S5 延迟记录:     ~record_latency_to=CSV 时记录 callback 与 sensor-to-command 延迟
                   (model-only 延迟由 latency_bench.py 另测), 退出时写 CSV + 摘要
  S6 静默检查:     ~dry_run=true 时不发布任何速度, 打印状态诊断(真机门禁用)

话题: 订阅 /scan, /odom; 发布 /cmd_vel。
用法(机器人板载 ROS1):
  python rl_nav_node_ros1.py _goal_x:=5.0 _goal_y:=5.0
  # 阶段 B 低速: _lin_scale:=0.16 _ang_scale:=0.5 _control_hz:=3.3
  # 门禁静默检查: _dry_run:=true _record_latency_to:=/home/wheeltec/latency.csv
"""
import csv
import os
import statistics
import time

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

STATE_DIM = 185
LASER_DIM = 180
RANGE_MAX = 7.0            # F1: 与仿真世界 range_max 一致
GOAL_THRESHOLD = 0.3       # F7: 与仿真 goal_threshold 一致
WATCHDOG_TIMEOUT = 0.5     # F5: 数据超时即停车(GPT 建议 0.3-0.5s)


class RLNavNode:
    def __init__(self):
        self.goal_x = rospy.get_param("~goal_x", 9.0)
        self.goal_y = rospy.get_param("~goal_y", 9.0)
        self.lin_scale = rospy.get_param("~lin_scale", 1.0)        # S4
        self.ang_scale = rospy.get_param("~ang_scale", 1.0)        # S4
        self.w_step = rospy.get_param("~w_step", 0.0)              # S7: 角速度每步变化上限(诊断, 0=关)
        self.control_hz = rospy.get_param("~control_hz", 3.3)      # S3
        self.safety_dist = rospy.get_param("~safety_dist", 0.20)   # S2
        self.dry_run = rospy.get_param("~dry_run", False)          # S6
        self.record_to = rospy.get_param("~record_latency_to", "") # S5
        self.front_center_frac = rospy.get_param("~front_center_frac", 0.75)  # F6: 车头在扫描中的位置(实测标定)

        model_path = rospy.get_param("~model_path", "cnntd3_actor.pt")
        # 支持两种部署形态: .pt(TorchScript) 与 .onnx(ONNX Runtime, INT8/FP32 压缩版)
        self.model, self.ort_sess, self.ort_iname = None, None, None
        if model_path.endswith(".onnx"):
            import onnxruntime as ort
            self.ort_sess = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
            self.ort_iname = self.ort_sess.get_inputs()[0].name
            rospy.loginfo("加载 ONNX 模型(ORT): %s", model_path)
        else:
            self.model = torch.jit.load(model_path)   # F4: 先加载
            self.model.eval()

        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)  # F4: 再建发布器

        self.robot_x, self.robot_y, self.robot_yaw = 0.0, 0.0, 0.0
        self.last_lin, self.last_ang = 0.0, 0.0    # F2: 上一步真实指令
        self.last_scan_time = rospy.Time.now()     # F5
        self.last_odom_time = rospy.Time.now()
        self._last_infer_time = 0.0                # S3: 节流用
        self._latency_samples = []                 # S5: (callback_ms, sensor2cmd_ms)
        self._scan_meta_logged = False

        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)   # F4: 最后建订阅
        # 里程计话题可选: 默认/odom(nav_msgs/Odometry); EKF融合输出为
        # PoseWithCovarianceStamped → 用 _odom_topic:=/robot_pose_ekf/odom_combined _odom_pose:=true
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        if rospy.get_param("~odom_pose", False):
            from geometry_msgs.msg import PoseWithCovarianceStamped
            rospy.Subscriber(odom_topic, PoseWithCovarianceStamped, self.pose_callback)
        else:
            rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        rospy.loginfo("里程计话题: %s (pose=%s)", odom_topic,
                      rospy.get_param("~odom_pose", False))
        rospy.on_shutdown(self.stop)
        rospy.Timer(rospy.Duration(0.25), self.watchdog)  # F5
        rospy.loginfo("RLNavNode ready | model=%s goal=(%.1f,%.1f) lin_scale=%.2f "
                      "ang_scale=%.2f control_hz=%.1f dry_run=%s",
                      model_path, self.goal_x, self.goal_y, self.lin_scale,
                      self.ang_scale, self.control_hz, self.dry_run)

    def stop(self):
        """F5/F7/S1/S2: 发布零速度(安全兜底唯一出口)"""
        if not self.dry_run:
            twist = Twist()
            self.pub.publish(twist)
        self.last_lin, self.last_ang = 0.0, 0.0

    def watchdog(self, event):
        now = rospy.Time.now()
        if (now - self.last_scan_time).to_sec() > WATCHDOG_TIMEOUT or \
           (now - self.last_odom_time).to_sec() > WATCHDOG_TIMEOUT:
            self.stop()
            rospy.logwarn_throttle(1.0, "Watchdog: sensor timeout, stopping")

    def odom_callback(self, msg):
        self.last_odom_time = rospy.Time.now()
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def pose_callback(self, msg):   # EKF 融合输出(PoseWithCovarianceStamped)
        self.last_odom_time = rospy.Time.now()
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def lidar_callback(self, msg):
        self.last_scan_time = rospy.Time.now()
        t_cb_enter = time.perf_counter()

        if not self._scan_meta_logged:
            rospy.loginfo("scan: n=%d angle_min=%.3f angle_max=%.3f (360°=%s)",
                          len(msg.ranges), msg.angle_min, msg.angle_max,
                          abs((msg.angle_max - msg.angle_min) - 2 * np.pi) < 0.01)
            self._scan_meta_logged = True

        # S3: 控制频率节流(与仿真 0.3s 步长对齐)
        now_mono = time.monotonic()
        if now_mono - self._last_infer_time < 1.0 / self.control_hz:
            return
        self._last_infer_time = now_mono

        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        distance = float(np.hypot(dx, dy))

        if distance < GOAL_THRESHOLD:      # F7: 到达 → 停车
            self.stop()
            rospy.loginfo_throttle(2.0, "Goal reached, stopped")
            self._record_latency(t_cb_enter, msg)
            return

        try:
            ranges = self._extract_front_ranges(msg)   # F6
            ranges = np.where(np.isfinite(ranges), ranges, RANGE_MAX)  # F1
            ranges = np.clip(ranges, 0.0, RANGE_MAX) / RANGE_MAX
            ranges = self._resample_to_180(ranges)

            heading_error = np.arctan2(dy, dx) - self.robot_yaw
            cos, sin = float(np.cos(heading_error)), float(np.sin(heading_error))
            lin_enc = self.last_lin * 2.0              # F2
            ang_enc = (self.last_ang + 1.0) / 2.0

            state = np.concatenate([ranges, [distance / 10.0, cos, sin, lin_enc, ang_enc]]).astype(np.float32)
            if not np.all(np.isfinite(state)):         # S1: NaN/Inf 输入 → 停车
                rospy.logwarn("Non-finite state, stopping")
                self.stop()
                return

            if self.dry_run:                           # S6: 打印诊断, 继续跑推理
                self._dry_run_diag(state, distance, msg)

            if self.ort_sess is not None:          # ONNX Runtime 推理(压缩版)
                action = self.ort_sess.run(
                    None, {self.ort_iname: state.reshape(1, -1)})[0].flatten()
            else:                                  # TorchScript 推理
                with torch.no_grad():
                    action = self.model(
                        torch.tensor(state).unsqueeze(0)).numpy().flatten()

            # S2: safety shield(基于原始前向距离, 未归一化)
            raw_front = ranges * RANGE_MAX
            if float(np.min(raw_front)) < self.safety_dist:
                rospy.logwarn_throttle(1.0, "Safety shield: front min=%.2fm < %.2fm, stop",
                                       float(np.min(raw_front)), self.safety_dist)
                self.stop()
                return

            lin_cmd = float((action[0] + 1.0) / 4.0) * self.lin_scale      # F3 + S4
            ang_cmd = float(np.clip(action[1], -1.0, 1.0)) * self.ang_scale
            if self.w_step > 0.0:   # S7: 角速度变化率限制(诊断用, 默认关)
                ang_cmd = float(np.clip(ang_cmd,
                                        self.last_ang - self.w_step,
                                        self.last_ang + self.w_step))

            if self.dry_run:                           # S6: 只诊断不发布
                return
            twist = Twist()
            twist.linear.x = lin_cmd
            twist.angular.z = ang_cmd
            self.pub.publish(twist)
            self.last_lin, self.last_ang = lin_cmd, ang_cmd   # F2: 记录实际指令
        except Exception as e:                       # S1: 任何异常 → 停车
            rospy.logerr_throttle(2.0, "Inference error: %s — stopping", e)
            self.stop()
        finally:
            self._record_latency(t_cb_enter, msg)

    def _dry_run_diag(self, state, distance, scan_msg):
        """S6: 静默检查 — 打印状态健康度(GPT 门禁: 185维/无NaN/0-1/前方障碍/cos-sin)

        raw_min_idx: 全圈扫描中最短束的编号(校准用)。
        校准方法: 在机器人正前方放障碍物, 读 raw_min_idx, 则
        front_center_frac = raw_min_idx / n(束总数), 下次启动时传入该值。
        """
        scan = state[:LASER_DIM]
        n = len(scan_msg.ranges)
        full = np.array(scan_msg.ranges, dtype=np.float32)
        finite = full[np.isfinite(full) & (full > 0)]
        raw_min_idx = int(np.argmin(full)) if len(finite) else -1
        checks = {
            "state_dim": len(state),
            "nan_count": int(np.isnan(state).sum()),
            "scan_min": float(np.min(scan)),
            "scan_max": float(np.max(scan)),
            "front_beam_min": float(np.min(scan)),
            "raw_min_idx": raw_min_idx,
            "n_beams": n,
            "frac_建议": round(raw_min_idx / n, 3) if n and raw_min_idx >= 0 else None,
            "dist_enc": float(state[LASER_DIM]),
            "cos_sin": [float(state[LASER_DIM + 1]), float(state[LASER_DIM + 2])],
            "last_act_enc": [float(state[LASER_DIM + 3]), float(state[LASER_DIM + 4])],
            "dist_to_goal": round(distance, 3),
        }
        rospy.loginfo_throttle(1.0, "dry-run: %s", checks)

    def _extract_front_ranges(self, msg):
        """F6: 提取前向 180° 窗口(共 n/2 束), 再降采样到 180 束。

        front_center_frac: 机器人正前方在 360° 扫描中的位置比例(需标定)。
        本次实测: 雷达 0° 指向机器人左侧 → 车头在扫描的 75% 处(0.75)。
        若换成 0.25 后手在正前方有反应, 则用 0.25。
        """
        n = len(msg.ranges)
        if n == LASER_DIM:
            return np.array(msg.ranges, dtype=np.float32)
        span = msg.angle_max - msg.angle_min
        if abs(span - 2 * np.pi) < 0.01:  # 360° 雷达
            half = n // 2
            center = int(n * self.front_center_frac)
            start = (center - half // 2) % n
            idx = (np.arange(half) + start) % n
            return np.array(msg.ranges)[idx].astype(np.float32)
        idx = np.linspace(0, n - 1, LASER_DIM).astype(int)
        return np.array(msg.ranges)[idx].astype(np.float32)

    def _resample_to_180(self, ranges):
        """F6: 分箱取 min(与训练降采样一致); <180 补满(视为 7m 无障碍)"""
        n = len(ranges)
        if n == LASER_DIM:
            return ranges
        if n > LASER_DIM:
            bin_size = n // LASER_DIM
            return np.array([ranges[i * bin_size:(i + 1) * bin_size].min()
                             for i in range(LASER_DIM)], dtype=np.float32)
        return np.concatenate([ranges, np.full(LASER_DIM - n, 1.0, dtype=np.float32)])

    def _record_latency(self, t_cb_enter, scan_msg):
        """S5: 记录 callback 与 sensor-to-command 延迟(仅记录模式)"""
        if not self.record_to:
            return
        cb_ms = (time.perf_counter() - t_cb_enter) * 1000.0
        try:
            s2c_ms = (rospy.Time.now() - scan_msg.header.stamp).to_sec() * 1000.0
        except Exception:
            s2c_ms = float("nan")
        self._latency_samples.append((cb_ms, s2c_ms))

    def dump_latency(self):
        """S5: 退出时写 CSV(每样本) + 控制台摘要(p50/p95/p99)"""
        if not self.record_to or not self._latency_samples:
            return
        with open(self.record_to, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["callback_ms", "sensor_to_cmd_ms"])
            w.writerows(self._latency_samples)
        cbs = [s[0] for s in self._latency_samples]
        s2c = [s[1] for s in self._latency_samples if np.isfinite(s[1])]

        def stats(xs):
            s = sorted(xs)
            return (round(statistics.mean(s), 3),
                    round(s[len(s) // 2], 3),
                    round(s[int(len(s) * 0.95)], 3),
                    round(s[int(len(s) * 0.99)], 3))
        rospy.loginfo("latency CSV -> %s | n=%d", self.record_to, len(self._latency_samples))
        if cbs:
            m, p50, p95, p99 = stats(cbs)
            rospy.loginfo("callback     ms: mean=%.3f p50=%.3f p95=%.3f p99=%.3f", m, p50, p95, p99)
        if s2c:
            m, p50, p95, p99 = stats(s2c)
            rospy.loginfo("sensor2cmd   ms: mean=%.3f p50=%.3f p95=%.3f p99=%.3f", m, p50, p95, p99)


def main():
    rospy.init_node("rl_nav_node")
    node = RLNavNode()
    try:
        rospy.spin()
    finally:
        node.stop()
        node.dump_latency()


if __name__ == "__main__":
    main()
