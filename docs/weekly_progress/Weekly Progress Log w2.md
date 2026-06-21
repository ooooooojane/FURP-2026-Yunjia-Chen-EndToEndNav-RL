### Week 2 — 2026-06-21

**Attended this week's meeting:** Yes

**Progress this week**
- Made a bootable Ubuntu 22.04 USB on macOS and installed the system onto a Samsung T7 SSD with manual partitioning (EFI, swap, root).
- Set up essential apps on Ubuntu: Chrome, WeChat, and VSCode.
- Configured Habitat: set up conda environment, installed habitat-sim and habitat-lab, and passed the random-walk smoke test.
- Set up Isaac Lab: created a Python 3.10 environment, installed Isaac Sim and Isaac Lab, verified with create_empty.py; attempted training test (Isaac-Ant-v0) but hit hardware limits.
- Set up Nav2 on ROS 2 Humble: launched Gazebo simulation with TurtleBot3, spawned the robot, and verified that Nav2 nodes (map_server, planner_server, controller_server) are running.

**Challenges & blockers** 
*Resolved*: Ubuntu boot issues on external drive — fixed /etc/fstab by switching from device names to UUID for EFI mounting.
*Resolved*: Missing gym in Habitat — installed gym==0.21.0 after downgrading setuptools.
*Resolved*: Isaac Lab dependency resolution (resolution-too-deep) — fixed by using a fresh Python 3.10 conda environment and proper pip extra index URL.
*Resolved*: Gazebo spawn_entity issues — loaded the ROS factory plugin and manually spawned the robot model.
*Resolved*: Isaac Lab smoke test fails due to insufficient GPU memory (NVIDIA T400, 2GB VRAM) — PhysX cannot allocate GPU memory for physics simulation; training cannot proceed on current hardware.
*Ongoing*: Nav2 map display — map_server is running but the /map topic contains all -1 values (unknown cells). The default map file is missing, and custom map creation leads to map_server configuration failure (Transitioning failed). Currently debugging the map loading issue.

**Next steps**
- Complete the Nav2 smoke test by either: (a) debugging the map_server configuration failure, (b) using an alternative simulation like tb3_loopback_simulation.launch.py, or (c) creating a minimal working map file with correct YAML formatting.
- For Isaac Lab: due to hardware limitations, either (a) run lightweight environment loading tests only (without training), (b) use cloud GPU resources, or (c) focus primarily on the Habitat path for the project.
- Begin Week 3 objectives: download the PointNav or VLN dataset for Habitat and run baseline models.
- Document all working configurations and smoke test results for the Week 2 checkpoint submission.

**Hours spent (optional):** idk i spent my whole week in the computer lab.
