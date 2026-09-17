"""Generalization test: run any algorithm+reward on 3 unseen maps."""
import sys, torch, numpy as np
from pathlib import Path
from robot_nav.SIM_ENV.sim import SIM

ALGO = sys.argv[1] if len(sys.argv) > 1 else "SAC"       # SAC or PPO
REWARD = sys.argv[2] if len(sys.argv) > 2 else "default"  # default, dense, harsh, sparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Build model ──────────────────────────────────────────
algo_upper = ALGO.upper()
model_name = algo_upper if REWARD == "default" else f"{algo_upper}_{REWARD}"

if algo_upper == "SAC":
    from robot_nav.models.SAC.SAC import SAC
    model = SAC(state_dim=185, action_dim=2, max_action=1, device=device,
                save_every=0, load_model=True, model_name=model_name,
                save_directory=Path("robot_nav/models/SAC/checkpoint"),
                load_directory=Path("robot_nav/models/SAC/checkpoint"))
elif algo_upper == "PPO":
    from robot_nav.models.PPO.PPO import PPO
    model = PPO(state_dim=185, action_dim=2, max_action=1, device=device,
                save_every=0, load_model=True, model_name=model_name,
                save_directory=Path("robot_nav/models/PPO/checkpoint"),
                load_directory=Path("robot_nav/models/PPO/checkpoint"))
else:
    raise ValueError(f"Unknown algo: {ALGO}")

print(f"Model: {model_name} on {device}\n")

# ── Test on unseen maps ──────────────────────────────────
maps = ["circle_world.yaml", "cross_world.yaml", "eval_world.yaml"]
EPISODES = 10

for world in maps:
    sim = SIM(world_file=f"worlds/{world}", disable_plotting=True)
    goals, cols, rewards = 0, 0, 0.0
    for ep in range(EPISODES):
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
        done, steps = False, 0
        while not done and steps < 500:
            state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
            action = model.get_action(np.array(state), add_noise=False)
            a_in = [(action[0] + 1) / 4, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1])
            rewards += reward; steps += 1
            done = collision or goal
        if goal: goals += 1
        if collision: cols += 1

    print(f"  {world:<28s}  Goal: {goals}/{EPISODES}  "
          f"Collision: {cols}/{EPISODES}  Avg Reward: {rewards/EPISODES:.1f}")

print(f"\n{model_name} generalization done.")
