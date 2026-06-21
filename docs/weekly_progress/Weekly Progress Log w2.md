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
- Download the PointNav or VLN dataset for Habitat and run baseline models.
- Document all working configurations and smoke test results for the Week 2 checkpoint submission.

**Hours spent (optional):** idk i spent my whole week in the computer lab.


<img width="1920" height="1080" alt="截图 2026-06-19 20-46-10" src="https://github.com/user-attachments/assets/df36d42b-98bf-4ac8-a16a-dbc7157d0672" />
<img width="1920" height="1080" alt="截图 2026-06-19 21-58-13" src="https://github.com/user-attachments/assets/6748fcf5-2698-4554-ac63-22723296b55e" />
<img width="1920" height="1080" alt="截图 2026-06-16 21-27-52" src="https://github.com/user-attachments/assets/5c4086cb-c7d3-49fc-a86a-fcbe3b4b18ee" />
<img width="1920" height="1080" alt="截图 2026-06-16 19-40-57" src="https://github.com/user-attachments/assets/ad964e9d-b364-4ebb-9031-f7d366f71627" />

