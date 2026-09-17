"""ROS2 node: LIDAR scan → CNNTD3 model → velocity command.
   Hardware: Turtlebot3 Burger + Orange Pi + LDS-02 LIDAR."""
import rclpy
import torch
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class RLNavNode(Node):
    def __init__(self):
        super().__init__('rl_nav_node')

        # ── LIDAR: Turtlebot3 LDS-02 (360 points, 0.12-3.5m) ──
        self.sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        # ── Odometry for goal tracking ──
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # ── Velocity command ──
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Load model ──
        self.model = torch.jit.load('cnntd3_actor.pt')
        self.model.eval()
        self.get_logger().info('CNNTD3 model loaded (Turtlebot3 Burger)')

        # ── Goal & odometry state ──
        self.goal_x = 9.0
        self.goal_y = 9.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = np.arctan2(siny, cosy)

    def lidar_callback(self, msg):
        # 1. Raw LIDAR → 180-dim normalized vector
        # LDS-02: 360 points, range 0.12–3.5m
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=0.12)
        ranges = np.clip(ranges, 0.12, 3.5) / 3.5  # normalize to [0,1]

        # Downsample 360 → 180 (take min over each 2-point bin)
        n = len(ranges)
        if n > 180:
            bin_size = n // 180
            ranges = np.array([ranges[i*bin_size:(i+1)*bin_size].min()
                               for i in range(180)], dtype=np.float32)

        # 2. Compute goal vector from odometry
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        distance = np.sqrt(dx*dx + dy*dy)
        goal_angle = np.arctan2(dy, dx)
        heading_error = goal_angle - self.robot_yaw
        cos = np.cos(heading_error)
        sin = np.sin(heading_error)

        # 3. Assemble 185-dim state
        state = np.concatenate([
            ranges,                           # 180
            [distance/10.0, cos, sin,         # 3
             0.0, 0.0]                        # 2 (last action, dummy)
        ]).astype(np.float32)
        state_tensor = torch.tensor(state).unsqueeze(0)

        # 4. Inference
        with torch.no_grad():
            action = self.model(state_tensor).numpy().flatten()

        # 5. Action → velocity: Turtlebot3 max 0.22 m/s
        lin_vel_cmd = float((action[0] + 1) / 4) * 0.22
        ang_vel_cmd = float(action[1]) * 2.84  # max 2.84 rad/s

        twist = Twist()
        twist.linear.x = lin_vel_cmd
        twist.angular.z = ang_vel_cmd
        self.pub.publish(twist)


def main():
    rclpy.init()
    node = RLNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
