"""Failure analysis: run 100 episodes, log every collision scenario detail."""
import torch, numpy as np, json
from collections import Counter
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
from robot_nav.SIM_ENV.sim import SIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTD3(state_dim=185, action_dim=2, max_action=1, device=device,
               save_every=0, load_model=True, model_name="CNNTD3")

sim = SIM(world_file="worlds/robot_world.yaml", disable_plotting=True)

N = 500
results = []  # {success, collision, goal, steps, final_distance, min_obstacle_dist, ...}

for ep in range(N):
    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
    done, steps = False, 0
    min_obstacle = float('inf')
    start_distance = distance
    trajectory_distances = []

    while not done and steps < 500:
        state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
        action = model.get_action(np.array(state), add_noise=False)
        a_in = [(action[0] + 1) / 4, action[1]]
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1])
        steps += 1
        min_obstacle = min(min_obstacle, min(latest_scan))
        trajectory_distances.append(distance)
        done = collision or goal

    # Categorize failure
    fail_reason = "success"
    if collision:
        # Check context: near goal? narrow passage? early collision?
        near_goal = distance < 2.0
        tight_squeeze = min_obstacle < 0.5  # obstacle within 0.5m
        early = steps < 30
        stagnant = max(trajectory_distances) - min(trajectory_distances) < 1.0  # didn't move much

        if near_goal:
            fail_reason = "near_goal_collision"
        elif tight_squeeze:
            fail_reason = "tight_space"
        elif early:
            fail_reason = "early_collision"
        elif stagnant:
            fail_reason = "stuck"
        else:
            fail_reason = "mid_route_collision"
    elif not goal:
        fail_reason = "timeout_no_goal"

    results.append({
        "episode": ep + 1,
        "success": goal,
        "collision": collision,
        "steps": steps,
        "final_distance": round(distance, 2),
        "min_obstacle": round(min_obstacle, 2),
        "fail_reason": fail_reason,
    })

# Print summary
successes = sum(1 for r in results if r["success"])
collisions = sum(1 for r in results if r["collision"])
timeouts = sum(1 for r in results if not r["success"] and not r["collision"])

print(f"{'='*55}")
print(f"  Failure Analysis — CNNTD3 Baseline (100 episodes)")
print(f"{'='*55}")
print(f"  Success:        {successes}/{N} ({successes*100/N:.0f}%)")
print(f"  Collisions:     {collisions}/{N}")
print(f"  Timeouts:       {timeouts}/{N}")
print()

# Breakdown
reasons = Counter(r["fail_reason"] for r in results)
print(f"  Failure breakdown:")
for reason, count in reasons.most_common():
    label = {
        "success": "Success",
        "timeout_no_goal": "Timeout (no goal reached)",
        "near_goal_collision": "Collision near goal (<2m)",
        "tight_space": "Collision in tight space (<0.5m clearance)",
        "early_collision": "Collision within first 30 steps",
        "stuck": "Stuck (barely moved)",
        "mid_route_collision": "Mid-route collision",
    }.get(reason, reason)
    print(f"  {label:<50s} {count:>3}")

# Average stats for failures
failed = [r for r in results if not r["success"]]
if failed:
    print(f"\n  Avg final distance on failure: {np.mean([r['final_distance'] for r in failed]):.1f}m")
    print(f"  Avg steps on failure:         {np.mean([r['steps'] for r in failed]):.0f}")

print(f"\n{'='*55}")
print("  Raw data saved to: training_logs/failure_analysis.json")
print(f"{'='*55}")

with open("training_logs/failure_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
