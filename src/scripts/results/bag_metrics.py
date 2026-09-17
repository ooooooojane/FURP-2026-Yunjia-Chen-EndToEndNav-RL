#!/usr/bin/env python3
"""单个bag指标提取: 用法 python3 bag_metrics.py <bag路径>"""
import math
import sys

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.ROS1_NOETIC)
f = sys.argv[1]
name = f.split("trial_")[-1].split(".")[0]

with Reader(f) as r:
    odom, cmd, goal_ts = [], [], None
    for conn, ts, raw in r.messages():
        try:
            if conn.topic == "/odom":
                m = typestore.deserialize_ros1(raw, conn.msgtype)
                odom.append((ts, m.pose.pose.position.x, m.pose.pose.position.y))
            elif conn.topic == "/cmd_vel":
                m = typestore.deserialize_ros1(raw, conn.msgtype)
                cmd.append((ts, m.angular.z))
            elif conn.topic == "/rosout" and goal_ts is None:
                m = typestore.deserialize_ros1(raw, conn.msgtype)
                if "Goal reached" in str(m):
                    goal_ts = ts
        except Exception:
            pass

if not odom:
    print(f"{name}: 空")
    sys.exit(0)
if goal_ts:
    end = min(odom, key=lambda o: abs(o[0] - goal_ts))
    d = math.hypot(end[1] - 3, end[2])
else:
    end = odom[-1]
    d = math.hypot(end[1] - 3, end[2])
ys = [o[2] for o in odom]
maxy = max(abs(min(ys)), abs(max(ys)))
ws = [c[1] for c in cmd]
rev = sum(1 for i in range(1, len(ws))
          if (ws[i] > 0.05) != (ws[i - 1] > 0.05) and abs(ws[i]) > 0.05 and abs(ws[i - 1]) > 0.05)
dur = (odom[-1][0] - odom[0][0]) / 1e9
print(f"{name}: 终({end[1]:.2f},{end[2]:.2f})距{d:.2f}m y摆{maxy:.2f}m 反转{rev}次 时{dur:.0f}s 到达={goal_ts is not None}")
