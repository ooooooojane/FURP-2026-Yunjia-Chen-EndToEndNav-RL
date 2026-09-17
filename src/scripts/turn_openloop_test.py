#!/usr/bin/env python3
"""开环定角速度测试(GPT建议): 固定w发固定时长, 对比odom累计转角 vs 物理转角。

每档: w=±0.3, ±0.6, ±1.0, 各3次重复。
物理转角: 用手机俯拍/地面参照物, 指令结束后观察是否惯性续转。
用法: python3 turn_openloop_test.py <w> <秒数> [重复次数]
例:   python3 turn_openloop_test.py 0.3 20.9 3   # 0.3rad/s×20.9s = odom应转360°
"""
import math
import sys
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

w = float(sys.argv[1])
duration = float(sys.argv[2])
n_rep = int(sys.argv[3]) if len(sys.argv) > 3 else 1

rospy.init_node("turn_openloop", anonymous=True)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

total_yaw = 0.0
prev = None
samples = []


def cb(msg):
    global total_yaw, prev
    q = msg.pose.pose.orientation
    y = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    if prev is not None:
        d = y - prev
        if d > math.pi:
            d -= 2 * math.pi
        if d < -math.pi:
            d += 2 * math.pi
        total_yaw += d
    prev = y


rospy.Subscriber("/odom", Odometry, cb)
time.sleep(1.0)  # 订阅就绪

for rep in range(n_rep):
    total_yaw = 0.0
    prev = None
    print(f"\n=== 档位 w={w:+.2f} rad/s, 时长{duration}s, 第{rep+1}次 ===", flush=True)
    t0 = time.time()
    while time.time() - t0 < duration:
        tw = Twist()
        tw.angular.z = w
        pub.publish(tw)
        time.sleep(0.05)  # 20Hz
    # 停
    pub.publish(Twist())
    # 继续观察1.5s(检测惯性续转)
    obs_start = total_yaw
    time.sleep(1.5)
    post = total_yaw - obs_start
    deg = math.degrees(total_yaw)
    print(f"odom累计转角: {deg:+.1f}° (期望{math.degrees(w*duration):+.1f}°)", flush=True)
    print(f"停止后1.5s继续转了: {math.degrees(post):+.1f}° (0=无惯性, 非0=有惯性)", flush=True)
    time.sleep(1.0)

print("\n[*] 完成。请用手机俯拍/参照物记录每次物理转角, 与odom对比。")
