"""Generalization test: run CNNTD3 on unseen maps."""
import torch, numpy as np
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
from robot_nav.SIM_ENV.sim import SIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTD3(state_dim=185, action_dim=2, max_action=1, device=device,
               save_every=0, load_model=True, model_name="CNNTD3")
print(f"Model loaded on {device}\n")

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

    print(f"{world:<25s}  Goal: {goals}/{EPISODES}  "
          f"Collision: {cols}/{EPISODES}  Avg Reward: {rewards/EPISODES:.1f}")

print("\nDone.")
