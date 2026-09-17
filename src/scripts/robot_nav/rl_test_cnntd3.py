"""Test trained CNNTD3 model with visualization (for recording navigation video)."""
import torch
import numpy as np
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
from robot_nav.SIM_ENV.sim import SIM

action_dim = 2
max_action = 1
state_dim = 185
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNTD3(
    state_dim=state_dim, action_dim=action_dim,
    max_action=max_action, device=device,
    save_every=0, load_model=True, model_name="CNNTD3",
)
print(f"Model loaded on {device}")

# plotting ON for video recording
sim = SIM(world_file="worlds/robot_world.yaml", disable_plotting=False)

for ep in range(5):
    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
    done = False
    steps = 0
    ep_reward = 0

    while not done and steps < 500:
        state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
        action = model.get_action(np.array(state), add_noise=False)
        a_in = [(action[0] + 1) / 4, action[1]]
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
        ep_reward += reward
        steps += 1
        done = collision or goal

    print(f"Episode {ep+1}: steps={steps}, reward={ep_reward:.1f}, "
          f"collision={collision}, goal={goal}")

print("Done. Close the simulation window to exit.")
