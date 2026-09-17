import sys
sys.path.insert(0, "/home/furp/DRL-robot-navigation-IR-SIM")
from robot_nav.SIM_ENV.sim import SIM
sim = SIM(world_file="worlds/robot_world.yaml", disable_plotting=True)
print("脚本方式加载成功! step_time =", sim.env.step_time, "| 障碍物 =", len(sim.env.obstacle_list))
scan = sim.env.get_lidar_scan()
print("激光:", len(scan["ranges"]), "束 | range_max =", scan["range_max"])
