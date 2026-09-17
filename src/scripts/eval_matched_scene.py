#!/usr/bin/env python3
"""Matched physical-to-simulation diagnostic (v5: shield modes A-D, non-terminating).

v5 changes (2026-08-21, per GPT review):
  - Shield semantics unified with the real node: a shield step zeroes/slows the
    command for THAT step only; the episode continues (the real node judges
    again next scan). Episodes no longer terminate on the first shield trigger.
  - Four pre-registered shield modes (no post-hoc tuning):
      none         A: no shield (diagnostic baseline)
      hard         B: front_min < 0.20 -> v=0, w=0 (original rule, non-terminating)
      soft_linear  C: front_min < 0.20 -> v=0, w kept (policy may turn away)
      staged       D: >=0.40 normal; 0.25-0.40 v scaled down;
                     0.20-0.25 v=0 with w kept; <0.20 emergency stop
  - Clearance tiers: wide (0.475m), medium (0.225m), low (current, -0.025m overlap).
  - Screening mode: --no-perturb runs the nominal pose only (small fixed-seed
    screening per GPT plan); effective modes are then expanded to perturbations.

Run from Code/scripts:
  python eval_matched_scene.py --torchscript exported/CNNTD3_actor.pt \
      --int8 exported/compressed/CNNTD3_int8_ir9.onnx \
      --world-dir matched_worlds --shield-modes none hard soft_linear staged \
      --no-perturb --output results/matched_scene_results_v5.json
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robot_nav.SIM_ENV.sim import SIM  # noqa: E402


LOCAL_TO_WORLD = np.array([0.5, 4.0, 0.0], dtype=float)  # v2: 走廊型世界平移
GOAL_LOCAL = np.array([3.0, 0.0, 0.0], dtype=float)
MAX_STEPS = 200


class TorchScriptPolicy:
    def __init__(self, path):
        import torch
        self.torch = torch
        self.model = torch.jit.load(str(path), map_location='cpu')
        self.model.eval()

    def raw_action(self, state):
        with self.torch.no_grad():
            out = self.model(self.torch.tensor(state, dtype=self.torch.float32).reshape(1, -1))
        return out.detach().cpu().numpy().reshape(-1)


class OrtPolicy:
    def __init__(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def raw_action(self, state):
        out = self.session.run(None, {self.input_name: state.astype(np.float32).reshape(1, -1)})[0]
        return out.reshape(-1)


def prepare_state(state_tuple):
    scan, distance, cos_h, sin_h, collision, goal, last_action, _ = state_tuple
    scan = np.asarray(scan, dtype=np.float32).copy()
    scan[~np.isfinite(scan)] = 7.0
    scan = np.clip(scan, 0.0, 7.0) / 7.0
    state = np.concatenate([
        scan,
        np.asarray([
            distance / 10.0,
            cos_h,
            sin_h,
            last_action[0] * 2.0,
            (last_action[1] + 1.0) / 2.0,
        ], dtype=np.float32),
    ])
    if state.shape != (185,):
        raise ValueError(f'Expected 185-D state, got {state.shape}')
    return state


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        from irsim.util.random import set_seed
        set_seed(seed)
    except ImportError:
        pass


def count_reversal(previous, current, threshold=0.10):
    return (abs(previous) >= threshold and abs(current) >= threshold
            and np.sign(previous) != np.sign(current))


def apply_shield(mode, front_min, scan_m, v_cmd, w_cmd):
    """Shield policy per mode. All modes are NON-TERMINATING (one step only).

    Returns (v, w, triggered).
    E (dir_away): direction-aware — the argmin beam decides which side the
    obstacle is on (beam 0 = right-front, beam 179 = left-front, verified in
    IR-SIM); trigger slows v to 0.15 and steers w AWAY at 0.5 rad/s.
    """
    if mode == 'none':
        return v_cmd, w_cmd, False
    if mode == 'hard':                       # B: 原规则, 非终止
        if front_min < 0.20:
            return 0.0, 0.0, True
        return v_cmd, w_cmd, False
    if mode == 'soft_linear':                # C: 只停线速度, 保留角速度
        if front_min < 0.20:
            return 0.0, w_cmd, True
        return v_cmd, w_cmd, False
    if mode == 'staged':                     # D: 分级减速
        if front_min >= 0.40:
            return v_cmd, w_cmd, False
        if front_min >= 0.25:
            scale = float(np.clip((front_min - 0.25) / 0.15, 0.0, 1.0))
            return v_cmd * scale, w_cmd, True
        if front_min >= 0.20:
            return 0.0, w_cmd, True         # 停线速度, 保留角速度(可转向)
        return 0.0, 0.0, True               # 紧急停(非终止)
    if mode == 'dir_away':                   # E: 方向感知, 减速+朝远离方向转
        # v5e 修正: 触发距离必须大于动力学反转距离(τ_ang=1.0s 下从 w=+0.93
        # 反转到 -1.0 需约1.5s≈0.3m前进量) → 0.30m 触发 + 全力反向(w=±1.0) + 蠕行
        if front_min < 0.30:
            finite = scan_m[np.isfinite(scan_m)]
            idx = int(np.argmin(finite)) if finite.size else 90
            away_w = 1.0 if idx < 90 else -1.0   # 障碍在右(束<90)→左转; 在左→右转
            return 0.10, away_w, True
        return v_cmd, w_cmd, False
    raise ValueError(f'unknown shield mode: {mode}')


def run_one(policy, world, seed, dy, dyaw_deg, lin_scale, ang_scale,
            shield_mode, max_steps, ang_tau=1.0, lin_tau=0.5):
    set_all_seeds(seed)
    sim = SIM(world_file=str(world), disable_plotting=True,
              reward_mode='default', seed=seed)

    start_local = np.array([0.0, dy, math.radians(dyaw_deg)])
    start_world = (start_local + LOCAL_TO_WORLD).reshape(3, 1)
    goal_world = (GOAL_LOCAL + LOCAL_TO_WORLD).reshape(3, 1)
    st = sim.reset(robot_state=start_world, robot_goal=goal_world,
                   random_obstacles=False)

    trajectory, commands = [], []
    collision_seen = False
    reversals = 0
    shield_triggers = 0
    previous_w = 0.0
    final = st
    # 底盘动力学一阶滞后(真机bag拟合: 角速度 τ≈1.0s, 线速度 τ≈0.5s)
    v_phys, w_phys = 0.0, 0.0
    DT = 0.3

    for step in range(max_steps):
        state = prepare_state(st)
        raw = policy.raw_action(state)
        v_cmd = float((np.clip(raw[0], -1.0, 1.0) + 1.0) / 4.0) * lin_scale
        w_cmd = float(np.clip(raw[1], -1.0, 1.0)) * ang_scale

        # 盾: 前向180°最小原始距离(0距离束保留, 与真机节点一致)
        scan_m = np.asarray(st[0], dtype=float)
        finite = scan_m[np.isfinite(scan_m)]
        front_min = float(np.min(finite)) if finite.size else 7.0
        v_cmd, w_cmd, trig = apply_shield(shield_mode, front_min, scan_m, v_cmd, w_cmd)
        shield_triggers += 1 if trig else 0

        # 一阶滞后: 底盘实际执行速度
        if lin_tau > 0:
            v_phys += (v_cmd - v_phys) * (DT / lin_tau)
        else:
            v_phys = v_cmd
        if ang_tau > 0:
            w_phys += (w_cmd - w_phys) * (DT / ang_tau)
        else:
            w_phys = w_cmd

        if count_reversal(previous_w, w_cmd):
            reversals += 1
        previous_w = w_cmd
        commands.append([step, v_cmd, w_cmd, front_min])

        final = sim.step(lin_velocity=v_phys, ang_velocity=w_phys)
        # 状态的上一步动作 = 发布指令(与真机节点 F2 一致)
        final = list(final)
        final[6] = [v_cmd, w_cmd]
        final = tuple(final)
        st = final  # 每步更新状态
        pose = np.asarray(sim.env.get_robot_state(), dtype=float).reshape(-1)[:3]
        local_pose = pose - LOCAL_TO_WORLD
        trajectory.append([step + 1, *local_pose.tolist()])
        collision_seen = collision_seen or bool(final[4])

        if bool(final[4]) or bool(final[5]):   # 非终止盾: 只按碰撞/到达/超时退出
            break

    pose = np.asarray(sim.env.get_robot_state(), dtype=float).reshape(-1)[:3]
    local_pose = pose - LOCAL_TO_WORLD
    goal_distance = float(np.linalg.norm(local_pose[:2] - GOAL_LOCAL[:2]))
    success = bool(final[5]) and not collision_seen
    reason = ('success' if success else 'collision' if collision_seen else 'timeout')
    return {
        'seed': seed,
        'initial_dy_m': dy,
        'initial_dyaw_deg': dyaw_deg,
        'success': success,
        'reason': reason,
        'collision': collision_seen,
        'goal_flag': bool(final[5]),
        'shield_triggers': shield_triggers,
        'steps': len(commands),
        'goal_distance_m': goal_distance,
        'final_local_pose': local_pose.tolist(),
        'angular_reversals': reversals,
        'trajectory': trajectory,
        'commands': commands,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world-dir', type=Path, default=HERE / 'matched_worlds')
    ap.add_argument('--torchscript', type=Path, required=True)
    ap.add_argument('--int8', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=HERE / 'results' / 'matched_scene_results.json')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 7])
    ap.add_argument('--lin-scale', type=float, default=0.8)
    ap.add_argument('--ang-scale', type=float, default=1.0)
    ap.add_argument('--shield-modes', type=str, nargs='+',
                    default=['hard'],
                    choices=['none', 'hard', 'soft_linear', 'staged', 'dir_away'])
    ap.add_argument('--no-perturb', action='store_true',
                    help='screening mode: nominal pose only')
    ap.add_argument('--max-steps', type=int, default=MAX_STEPS)
    ap.add_argument('--local-to-world', type=float, nargs=3,
                    default=[0.5, 4.0, 0.0],
                    help='world translation of the local origin')
    ap.add_argument('--ang-tau', type=float, default=1.0,
                    help='first-order lag time constant (s) for angular channel, 0=ideal')
    ap.add_argument('--lin-tau', type=float, default=0.5,
                    help='first-order lag time constant (s) for linear channel, 0=ideal')
    ap.add_argument('--goal-x', type=float, default=3.0,
                    help='goal local x (default 3.0)')
    ap.add_argument('--goal-y', type=float, default=0.0,
                    help='goal local y (default 0.0)')
    args = ap.parse_args()
    global LOCAL_TO_WORLD, GOAL_LOCAL
    LOCAL_TO_WORLD = np.array(args.local_to_world, dtype=float)
    GOAL_LOCAL = np.array([args.goal_x, args.goal_y, 0.0], dtype=float)

    scenes = {
        'straight': args.world_dir / 'matched_straight_rect.yaml',
        'box_low_clearance': args.world_dir / 'matched_box_left_rect.yaml',
        'box_med_clearance': args.world_dir / 'matched_box_left_med_rect.yaml',
        'box_wide_clearance': args.world_dir / 'matched_box_left_wide_rect.yaml',
        'box_wide_right': args.world_dir / 'matched_box_right_wide_rect.yaml',
        'box_double_slalom': args.world_dir / 'matched_double_obstacle.yaml',
    }
    policies = {
        'torchscript_fp32': TorchScriptPolicy(args.torchscript),
        'onnx_int8': OrtPolicy(args.int8),
    }
    perturbations = [(0.0, 0.0)] if args.no_perturb else [
        (0.0, 0.0), (-0.05, 0.0), (0.05, 0.0), (0.0, -3.0), (0.0, 3.0)]

    records = []
    for scene_name, world in scenes.items():
        if not world.exists():
            raise FileNotFoundError(world)
        for policy_name, policy in policies.items():
            for seed in args.seeds:
                for dy, dyaw in perturbations:
                    for shield_mode in args.shield_modes:
                        result = run_one(policy, world, seed, dy, dyaw,
                                         args.lin_scale, args.ang_scale,
                                         shield_mode, args.max_steps,
                                         args.ang_tau, args.lin_tau)
                        result.update({'scene': scene_name, 'policy': policy_name,
                                       'shield_mode': shield_mode})
                        records.append(result)

    summary = {}
    for scene_name in scenes:
        summary[scene_name] = {}
        for policy_name in policies:
            summary[scene_name][policy_name] = {}
            for shield_mode in args.shield_modes:
                group = [r for r in records
                         if r['scene'] == scene_name and r['policy'] == policy_name
                         and r['shield_mode'] == shield_mode]
                summary[scene_name][policy_name][shield_mode] = {
                    'n': len(group),
                    'success': sum(r['success'] for r in group),
                    'collision': sum(r['collision'] for r in group),
                    'shield_any': sum(r['shield_triggers'] > 0 for r in group),
                    'shield_total': sum(r['shield_triggers'] for r in group),
                    'timeout': sum(r['reason'] == 'timeout' for r in group),
                    'median_reversals': float(np.median([r['angular_reversals'] for r in group])),
                }

    payload = {
        'status': 'generated by eval_matched_scene.py v5 (shield modes A-D, non-terminating)',
        'geometry_m': {
            'robot_length': 0.4436,
            'robot_width': 0.3474,
            'box_side': 0.3024,
            'clearances_m': {
                'box_low_clearance': 0.1488 - 0.1737,   # -0.0249 重叠
                'box_med_clearance': 0.55 - 0.1512 - 0.1737,
                'box_wide_clearance': 0.8 - 0.1512 - 0.1737,
            },
            'start_local': [0.0, 0.0, 0.0],
            'goal_local': [3.0, 0.0, 0.0],
        },
        'protocol': vars(args) | {'world_dir': str(args.world_dir),
                                  'torchscript': str(args.torchscript),
                                  'int8': str(args.int8),
                                  'output': str(args.output)},
        'summary': summary,
        'records': records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
