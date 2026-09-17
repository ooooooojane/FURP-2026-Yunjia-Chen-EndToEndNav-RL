#!/usr/bin/env python3
"""训练完成后的一键验证: 防止"训练时是好的、保存下来是坏的"重演。

用法(主机 furp_env / Mac drl env, 从项目根运行):
  python verify_model.py --algo cnntd3 [--episodes 10]

两步验证:
  1. 探针: 对 3 个随机状态输出动作 —— 动作必须随输入变化
     (如果恒定输出 (±1,±1) 之类的饱和值 = 权重退化, 直接报警)
  2. 快评: 10 局成功率(带起点过滤) —— 应 > 0.6 才算健康
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from robot_nav.SIM_ENV.sim import SIM  # noqa: E402
from eval_compare import DRLPolicy, reset_clear  # noqa: E402


def probe(model, sim, n=5):
    print("== 探针: 动作应随状态变化 ==")
    ok = True
    actions = []
    for i in range(n):
        st = sim.reset()
        a = model.get_action(st, sim)
        actions.append(a)
        print("  状态%d -> 动作 (%+.3f, %+.3f)" % (i, a[0], a[1]))
    acts = np.array(actions)
    spread = acts.max(axis=0) - acts.min(axis=0)
    # 退化判据: 两个通道都几乎不动(注意: 单一通道饱和可能是合法策略, 不算退化)
    if spread[0] < 0.05 and spread[1] < 0.05:
        print("  [警告] 两个动作通道都几乎不变! 权重疑似退化")
        ok = False
    else:
        print("  [OK] 动作随输入变化, 权重正常")
    return ok


def quick_eval(policy, sim, n_episodes=20, seed=42, world="robot_world.yaml"):
    print("== 快评: %d 局(seed=%d, 世界=%s, 带起点过滤) ==" % (n_episodes, seed, world))
    random = __import__("random")
    random.seed(seed)
    np.random.seed(seed)
    succ = coll = 0
    for ep in range(n_episodes):
        st = reset_clear(sim)          # 起点过滤: 消除"生成即碰撞"的假失败
        if hasattr(policy, "reset_episode"):
            policy.reset_episode(sim)
        done, steps = False, 0
        while not done and steps < 500:
            a = policy.get_action(st, sim)
            st = sim.step(lin_velocity=a[0], ang_velocity=a[1])
            steps += 1
            done = bool(st[4]) or bool(st[5])   # 碰撞=索引4, goal=索引5(索引3是sin!)
        succ += bool(st[5])
        coll += bool(st[4])
    print("  成功 %d/%d, 碰撞 %d/%d" % (succ, n_episodes, coll, n_episodes))
    return succ / n_episodes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["cnntd3", "sac", "ppo"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--world", default="robot_world.yaml",
                    help="验证世界(默认 robot_world=训练分布; eval_world 是跨分布测试)")
    args = ap.parse_args()

    # 注意: 世界文件必须用绝对路径(irsim 按 sys.path[0] 解析相对路径,
    # 脚本方式运行才找得到 robot_nav/worlds/, 这里直接用绝对路径最稳)
    sim = SIM(world_file=str(ROOT / "robot_nav/worlds" / args.world),
              disable_plotting=True, reward_mode="default")
    policy = DRLPolicy(args.algo)

    probe_ok = probe(policy, sim)
    rate = quick_eval(policy, sim, n_episodes=args.episodes, world=args.world)

    verdict = "[OK] 验证通过" if (probe_ok and rate >= 0.6) else "[FAIL] 验证失败"
    print("== 结论: %s (goal率 %.0f%%) ==" % (verdict, rate * 100))
