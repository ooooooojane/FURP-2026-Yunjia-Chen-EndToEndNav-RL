#!/usr/bin/env python3
"""单进程多方法评估: 同一环境实例, 同一批场景, 所有方法逐场景评估。

设计动机: 跨进程场景复现依赖 irsim 内部RNG 的完全可控, 已多次失败。
本方案: 一个进程内创建环境(布局固定) → 进程内生成场景 → 每个场景对每个方法
重播种后 reset+跑局。配对由构造保证: 同一sim实例、同一场景、同一噪声序列。

用法(neupan_env, 有全部依赖):
  python eval_multi_policy.py --n-scenes 500 --seed 42 --world-seed 42 \
      --policies "cnntd3,sac,ppo,neupan" \
      --out-prefix results/mp42
  # 压缩矩阵:
  python eval_multi_policy.py --n-scenes 200 --seed 42 --world-seed 42 \
      --policies "jit:baseline:exported/CNNTD3_actor.pt,jit:pruned10:exported/compressed/CNNTD3_pruned10.pt,..." \
      --out-prefix results/mp_matrix
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from robot_nav.SIM_ENV.sim import SIM  # noqa: E402


def make_policy(spec, seed):
    """spec: 'cnntd3' | 'sac' | 'ppo' | 'neupan' | 'jit:name:path' | 'ort:name:path'"""
    if spec == "cnntd3":
        from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
        return ("cnntd3", CNNTD3(state_dim=185, action_dim=2, max_action=1, device="cpu",
                                 save_every=0, load_model=True, model_name="CNNTD3",
                                 load_directory=str(ROOT / "robot_nav/models/CNNTD3/checkpoint")))
    if spec == "sac":
        from robot_nav.models.SAC.SAC import SAC
        return ("sac", SAC(state_dim=185, action_dim=2, device="cpu", max_action=1,
                           save_every=0, load_model=True, model_name="SAC",
                           load_directory=str(ROOT / "robot_nav/models/SAC/checkpoint")))
    if spec == "ppo":
        from robot_nav.models.PPO.PPO import PPO
        return ("ppo", PPO(state_dim=185, action_dim=2, max_action=1,
                           save_every=0, load_model=True, model_name="PPO",
                           load_directory=str(ROOT / "robot_nav/models/PPO/checkpoint")))
    if spec == "neupan":
        from neupan import neupan
        planner = neupan.init_from_yaml("/home/furp/planner_diff_small_best.yaml")
        return ("neupan", planner)
    parts = spec.split(":", 2)
    kind, name, path = parts[0], parts[1], str(ROOT / parts[2])
    if kind == "jit":
        import torch
        m = torch.jit.load(path).eval()
        from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
        helper = CNNTD3(state_dim=185, action_dim=2, max_action=1, device="cpu",
                        save_every=0, load_model=False)
        return (name, (m, helper))
    if kind == "ort":
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
        helper = CNNTD3(state_dim=185, action_dim=2, max_action=1, device="cpu",
                        save_every=0, load_model=False)
        return (name, (sess, helper))
    raise ValueError(spec)


def get_action(policy, name, state_tuple, sim):
    """统一动作接口(与 eval_compare.py 一致)"""
    latest_scan, distance, cos, sin, collision, goal, a, _ = state_tuple
    if name == "cnntd3" or name == "sac" or name == "ppo":
        state, _ = policy.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
        action = policy.get_action(np.array(state), add_noise=False)
        return [(action[0] + 1) / 4, action[1]]
    if name == "neupan":
        state = np.array(sim.env.get_robot_state())[:3]
        scan = dict(sim.env.get_lidar_scan())
        scan["range_max"] = 15.0
        points = policy.scan_to_point(state, scan)
        action, _ = policy(state, points, None)
        v, w = float(action[0]), float(action[1])
        return [min(max(v, 0.0), 1.0), min(max(w, -1.0), 1.0)]
    # jit/ort: (model, helper)
    model, helper = policy
    state, _ = helper.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
    if hasattr(model, "run"):  # ort session
        act = model.run(None, {model.get_inputs()[0].name: np.array(state, dtype=np.float32).reshape(1, -1)})[0]
        act = act.flatten()
    else:                      # torchscript
        import torch
        act = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).detach().numpy().flatten()
    return [(act[0] + 1) / 4, act[1]]


def reset_clear(sim, robot_state=None, robot_goal=None, max_tries=30):
    for _ in range(max_tries):
        st = sim.reset(robot_state=robot_state, robot_goal=robot_goal,
                       random_obstacles=(robot_state is None))
        if min(st[0]) < 0.5:
            continue
        s1 = sim.step(lin_velocity=0.0, ang_velocity=1.0)
        s2 = sim.step(lin_velocity=0.3, ang_velocity=0.0)
        if not (s1[4] or s2[4]):
            return s2
    raise RuntimeError("reset_clear: 无法找到合法起点")


def run_episode(policy, name, sim, state_tuple):
    done, steps, min_ob = False, 0, float("inf")
    while not done and steps < 500:
        latest_scan, distance, cos, sin, collision, goal, a, _ = state_tuple
        min_ob = min(min_ob, min(latest_scan))
        action = get_action(policy, name, state_tuple, sim)
        state_tuple = sim.step(lin_velocity=action[0], ang_velocity=action[1])
        steps += 1
        collision = bool(state_tuple[4])
        goal = bool(state_tuple[5])
        done = collision or goal
    distance = state_tuple[1]
    if collision:
        near = distance < 2.0
        tight = min_ob < 0.5
        early = steps < 30
        reason = ("near_goal_collision" if near else
                  "tight_space" if tight else
                  "early_collision" if early else "mid_route_collision")
    elif goal:
        reason = "success"
    else:
        reason = "timeout_no_goal"
    return {"success": bool(goal), "collision": bool(collision), "steps": steps,
            "min_obstacle": round(min_ob, 2), "fail_reason": reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--world-seed", type=int, default=42)
    ap.add_argument("--policies", required=True, help="逗号分隔, 如 cnntd3,sac,ppo,neupan")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    try:
        from irsim.util.random import set_seed as irsim_set_seed
    except ImportError:
        irsim_set_seed = None

    policies = [make_policy(spec, args.seed) for spec in
                [s.strip() for s in args.policies.split(",") if s.strip()]]
    print(f"[*] 策略: {[n for n, _ in policies]}")

    sim = SIM(world_file=str(ROOT / "robot_nav/worlds/robot_world.yaml"),
              disable_plotting=True, reward_mode="default", seed=args.world_seed)

    # 1) 进程内生成场景(每attempt播种, 探针验证), 记录每个场景的播种值
    scenes, attempt = [], 0
    while len(scenes) < args.n_scenes:
        sk = args.seed * 10000 + attempt
        attempt += 1
        random.seed(sk)
        np.random.seed(sk)
        if irsim_set_seed is not None:
            irsim_set_seed(sk)
        rs = [[random.uniform(1, 9)], [random.uniform(1, 9)], [0]]
        # 记录三随机源状态快照(起点采样后、reset前): 评估时精确恢复,
        # 消除"生成多消耗2次Python random"导致的边界场景翻转(seed7 400+场景崩溃根因)
        try:
            from irsim.util.random import rng as irsim_rng
            ir_state = irsim_rng.bit_generator.state
        except Exception:
            ir_state = None
        np_st = np.random.get_state()
        py_state = random.getstate()
        try:
            st = reset_clear(sim, robot_state=rs, robot_goal=None)
        except RuntimeError:
            continue
        pose = [rs[0][0], rs[1][0], 0.0]
        goal = np.array(sim.env.robot.goal)[:3].reshape(-1).tolist()
        scenes.append({"pose": pose, "goal": goal, "seed_k": sk,
                       "py_state": py_state,
                       "np_state": [np_st[0], np_st[1].tolist(), np_st[2],
                                    np_st[3], np_st[4]],
                       "ir_state": ir_state})
    print(f"[*] 场景: {len(scenes)} (尝试 {attempt})")

    # 2) 每个场景 × 每个策略: 重播种后 reset+跑局(同实例→同场景→配对由构造保证)
    results = {name: [] for name, _ in policies}
    t0 = time.time()
    for k, sc in enumerate(scenes):
        for name, pol in policies:
            # 精确恢复生成时的随机状态(比重新播种更严格, 消除消耗路径差异)
            def _to_tuple(x):
                return tuple(_to_tuple(i) for i in x) if isinstance(x, list) else x
            random.setstate(_to_tuple(sc["py_state"]))
            np_st = sc["np_state"]
            np.random.set_state((np_st[0], np.array(np_st[1]), np_st[2], np_st[3], np_st[4]))
            if sc.get("ir_state") is not None:
                try:
                    from irsim.util.random import rng as irsim_rng
                    irsim_rng.bit_generator.state = sc["ir_state"]
                except Exception:
                    pass
            # 评估不重复探针: 场景合法性已在生成阶段用探针验证过;
            # 直接 reset 开跑 → 消除"探针结果跨进程翻转"这一脆环节, 配对由构造保证
            st = sim.reset(robot_state=sc["pose"], robot_goal=None,
                           random_obstacles=False)
            if min(st[0]) < 0.3:   # 兜底: 生成验证过, 此处理论不触发
                print(f"[WARN] 场景{sc.get('seed_k')}重置后前方<0.3m, 跳过该策略", flush=True)
                continue
            if name == "neupan":  # 每局重新生成参考路径
                start = np.array(sim.env.get_robot_state())[:3]
                goal = np.array(sim.robot_goal)[:3]
                pol.update_initial_path_from_goal(start, goal)
            r = run_episode(pol, name, sim, st)
            r["episode"] = k + 1
            results[name].append(r)
        if (k + 1) % 50 == 0:
            print(f"  [{k+1}/{len(scenes)}] {(time.time()-t0)/60:.1f}min", flush=True)

    # 3) 汇总保存
    for name, rs in results.items():
        n = len(rs)
        succ = sum(1 for r in rs if r["success"])
        coll = sum(1 for r in rs if r["collision"])
        reasons = Counter(r["fail_reason"] for r in rs)
        summary = {"policy": name, "n": n, "success": succ / n, "collision": coll / n,
                   "timeout": (n - succ - coll) / n,
                   "avg_steps": sum(r["steps"] for r in rs) / n,
                   "reasons": dict(reasons)}
        print(f"  {name}: {succ}/{n} = {succ*100/n:.1f}%  {dict(reasons)}")
        with open(f"{args.out_prefix}_{name}.json", "w") as f:
            json.dump({"summary": summary, "episodes": rs}, f, ensure_ascii=False, indent=1)
    print("[*] 完成")


if __name__ == "__main__":
    main()
