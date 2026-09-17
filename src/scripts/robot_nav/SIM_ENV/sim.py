import irsim
import numpy as np
import random

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class SIM(SIM_ENV):
    """Simulation environment for robot navigation with configurable reward function.

    Reward modes:
      - "default": goal +100, collision -100, small shaping for speed & obstacle clearance
      - "dense":   default + distance-based progress reward (-distance/10 per step)
      - "sparse":  only goal +100 / collision -100, zero intermediate reward
      - "harsh":   default but collision penalty doubled to -200
    """

    def __init__(self, world_file="robot_world.yaml", disable_plotting=False,
                 reward_mode="default", seed=None):
        display = False if disable_plotting else True
        if seed is not None:
            # 固定障碍布局的关键: 创建环境前播种全部三个随机源。
            # (实测: 只播种 irsim 内部RNG不够——障碍生成还消费 Python random/numpy,
            #  不同进程布局不同; 三源同播后跨进程布局完全一致。)
            random.seed(seed)
            np.random.seed(seed)
            from irsim.util.random import set_seed
            set_seed(seed)
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display,
            seed=seed
        )
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.reward_mode = reward_mode
        self.prev_distance = None

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        self.env.render()

        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]

        robot_state = self.env.get_robot_state()
        goal_vector = [
            self.robot_goal[0].item() - robot_state[0].item(),
            self.robot_goal[1].item() - robot_state[1].item(),
        ]
        distance = np.linalg.norm(goal_vector)
        goal = self.env.robot.arrive
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self.cossin(pose_vector, goal_vector)
        collision = self.env.robot.collision
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(goal, collision, action, latest_scan, distance)

        return latest_scan, distance, cos, sin, collision, goal, action, reward

    def reset(self, robot_state=None, robot_goal=None, random_obstacles=True,
              random_obstacle_ids=None):
        if robot_state is None:
            robot_state = [[random.uniform(1, 9)], [random.uniform(1, 9)], [0]]

        self.env.robot.set_state(state=np.array(robot_state), init=True)

        if random_obstacles:
            if random_obstacle_ids is None:
                random_obstacle_ids = [i + 1 for i in range(7)]
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14], range_high=[10, 10, 3.14],
                ids=random_obstacle_ids, non_overlapping=True,
            )

        if robot_goal is None:
            self.env.robot.set_random_goal(
                obstacle_list=self.env.obstacle_list, init=True,
                range_limits=[[1, 1, -3.141592653589793], [9, 9, 3.141592653589793]],
            )
        else:
            self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal

        action = [0.0, 0.0]
        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        self.prev_distance = distance
        return latest_scan, distance, cos, sin, False, False, action, reward

    def get_reward(self, goal, collision, action, laser_scan, distance):
        if goal:
            return 100.0

        if collision:
            return -200.0 if self.reward_mode == "harsh" else -100.0

        # Intermediate reward
        r_obstacle = lambda x: 1.35 - x if x < 1.35 else 0.0
        base = action[0] - abs(action[1]) / 2 - r_obstacle(min(laser_scan)) / 2

        if self.reward_mode == "sparse":
            return 0.0

        if self.reward_mode == "dense":
            # Shaping: reward progress toward goal, penalize moving away
            progress = 0.0
            if self.prev_distance is not None:
                progress = self.prev_distance - distance  # positive = getting closer
            self.prev_distance = distance
            return base + progress * 5.0

        # default
        return base
