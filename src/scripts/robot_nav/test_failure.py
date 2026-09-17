"""Failure analysis: run N episodes, categorize every failure."""
import sys, torch, numpy as np, json
from collections import Counter
from pathlib import Path

ALGO = sys.argv[1] if len(sys.argv) > 1 else "SAC"
REWARD = sys.argv[2] if len(sys.argv) > 2 else "default"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
algo_upper = ALGO.upper()
model_name = algo_upper if REWARD == "default" else f"{algo_upper}_{REWARD}"

print(f"Failure Analysis: {model_name} ({N} episodes)")

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

from robot_nav.SIM_ENV.sim import SIM
sim = SIM(world_file="worlds/robot_world.yaml", disable_plotting=True)

results = []

for ep in range(N):
    if (ep + 1) % 100 == 0:
        print(f"  {ep+1}/{N}...")
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

    # Categorize
    fail_reason = "success"
    if collision:
        near_goal = distance < 2.0
        tight_squeeze = min_obstacle < 0.5
        early = steps < 30
        stagnant = len(trajectory_distances) > 1 and max(trajectory_distances) - min(trajectory_distances) < 1.0
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

# ── Print summary ──────────────────────────────────────
successes = sum(1 for r in results if r["success"])
collisions = sum(1 for r in results if r["collision"])
timeouts = sum(1 for r in results if not r["success"] and not r["collision"])

print(f"\n{'='*55}")
print(f"  Failure Analysis — {model_name} ({N} episodes)")
print(f"{'='*55}")
print(f"  Success:        {successes}/{N} ({successes*100/N:.0f}%)")
print(f"  Collisions:     {collisions}/{N}")
print(f"  Timeouts:       {timeouts}/{N}")

reasons = Counter(r["fail_reason"] for r in results)
print(f"\n  Failure breakdown:")
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

failed = [r for r in results if not r["success"]]
if failed:
    print(f"\n  Avg final distance on failure: {np.mean([r['final_distance'] for r in failed]):.1f}m")
    print(f"  Avg steps on failure:         {np.mean([r['steps'] for r in failed]):.0f}")

out_path = f"training_logs/failure_{model_name}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: {out_path}")
