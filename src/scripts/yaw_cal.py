#!/usr/bin/env python3
"""yaw标尺校准: 机器人原地转360°, 测odom累计转角, 输出odom_z_scale建议值。"""
import math

import rospy
from nav_msgs.msg import Odometry

total = 0.0
prev = None


def cb(msg):
    global total, prev
    q = msg.pose.pose.orientation
    y = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    if prev is not None:
        d = y - prev
        if d > math.pi:
            d -= 2 * math.pi
        if d < -math.pi:
            d += 2 * math.pi
        total += d
    prev = y


rospy.init_node("yaw_cal", anonymous=True)
rospy.Subscriber("/odom", Odometry, cb)


def on_shutdown():
    deg = math.degrees(total)
    print(f"odom累计转角: {deg:.1f}度 (物理应为360度)", flush=True)
    if abs(deg) > 1:
        print(f"建议 odom_z_scale = 360/{abs(deg):.1f} = {360/abs(deg):.3f}", flush=True)
    else:
        print("转角约0, 检查是否转满一圈", flush=True)


rospy.on_shutdown(on_shutdown)
print("准备好了! 现在把机器人原地转一整圈360度(10秒内), 转完按Ctrl+C", flush=True)
rospy.spin()
