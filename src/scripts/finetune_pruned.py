#!/usr/bin/env python3
"""剪枝-微调流程(任务 #4 升级): 结构化剪枝 actor → TD3 微调恢复成功率

用法(主机 furp_env, 项目根运行):
  python finetune_pruned.py --prune-ratio 0.5 --epochs 8

流程(遵循文献 prune-then-finetune 范式 [Han16, Li17]):
  1. 加载新权重 CNNTD3(actor + 完整 critic)
  2. structured_prune: 按 L2 范数删 MLP 整条神经元(actor 结构 400→200)
  3. PrunedCNNTD3: 换入剪枝 actor, critic/目标网络/训练逻辑全部复用
  4. 用与 rl_train.py 相同的循环微调 N 轮(经验回放 + TD3 更新)
  5. 保存检查点 + 导出 TorchScript → 用 eval_compare.py 做 100 局评估
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from robot_nav.SIM_ENV.sim import SIM  # noqa: E402
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3  # noqa: E402
from robot_nav.utils import ReplayBuffer  # noqa: E402

# 检查点在 GPU 机器训练,CPU 加载需 map_location
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": "cpu"})

STATE_DIM, ACTION_DIM = 185, 2


def structured_prune(actor, ratio):
    """结构化剪枝: 按 L2 范数删 MLP 整条神经元(真实减少计算量)。

    只剪 layer_1/layer_2/layer_3(参数大头), 卷积编码器与嵌入层保持。
    与 quantize_bench.py 中的实现一致, 保证压缩实验口径统一。
    """
    h1 = max(round(400 * (1 - ratio)), 50)
    h2 = max(round(300 * (1 - ratio)), 50)

    w1, b1 = actor.layer_1.weight.detach(), actor.layer_1.bias.detach()
    w2, b2 = actor.layer_2.weight.detach(), actor.layer_2.bias.detach()
    w3, b3 = actor.layer_3.weight.detach(), actor.layer_3.bias.detach()

    keep1 = torch.argsort(w1.norm(dim=1), descending=True)[:h1].sort().values
    keep2 = torch.argsort(w2.norm(dim=1), descending=True)[:h2].sort().values

    new_l1 = torch.nn.Linear(36, h1)
    new_l1.weight.data = w1[keep1]
    new_l1.bias.data = b1[keep1]
    new_l2 = torch.nn.Linear(h1, h2)
    new_l2.weight.data = w2[keep2][:, keep1]
    new_l2.bias.data = b2[keep2]
    new_l3 = torch.nn.Linear(h2, ACTION_DIM)
    new_l3.weight.data = w3[:, keep2]
    new_l3.bias.data = b3

    new_actor = torch.nn.Module()
    new_actor.cnn1 = actor.cnn1
    new_actor.cnn2 = actor.cnn2
    new_actor.cnn3 = actor.cnn3
    new_actor.goal_embed = actor.goal_embed
    new_actor.action_embed = actor.action_embed
    new_actor.layer_1 = new_l1
    new_actor.layer_2 = new_l2
    new_actor.layer_3 = new_l3

    def forward(x):
        # 与 CNNTD3 Actor.forward 相同的切片: 激光 s[:, :-5], 目标 s[:, -5:-2], 动作 s[:, -2:]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        laser = x[:, :-5].unsqueeze(1)
        feat = torch.nn.functional.leaky_relu(new_actor.cnn1(laser))
        feat = torch.nn.functional.leaky_relu(new_actor.cnn2(feat))
        feat = torch.nn.functional.leaky_relu(new_actor.cnn3(feat))
        feat = feat.flatten(1)
        goal = torch.nn.functional.leaky_relu(new_actor.goal_embed(x[:, -5:-2]))
        act = torch.nn.functional.leaky_relu(new_actor.action_embed(x[:, -2:]))
        h = torch.cat([feat, goal, act], dim=1)
        h = torch.nn.functional.leaky_relu(new_actor.layer_1(h))
        h = torch.nn.functional.leaky_relu(new_actor.layer_2(h))
        return torch.tanh(new_actor.layer_3(h))

    new_actor.forward = forward
    new_actor.eval()
    return new_actor


class PrunedCNNTD3(CNNTD3):
    """CNNTD3 子类: actor 换成剪枝版, critic/目标网络/训练逻辑复用。

    部署形态是 actor(0.12ms 的"大脑"); critic 只是训练时的"教练",
    保持完整不剪 —— 论文表述: "deployment uses the compressed actor only"。
    """

    def __init__(self, pruned_actor, lr=3e-4, **kwargs):
        super().__init__(**kwargs)   # 加载原始权重(actor 先建完整版再替换)
        self.actor = pruned_actor.to(self.device)
        self.actor_target = copy.deepcopy(pruned_actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)


def evaluate(model, sim, eval_episodes=10):
    print("Evaluating scenarios ...")
    goals = 0
    col = 0
    for _ in range(eval_episodes):
        s = sim.reset()
        done, count = False, 0
        while not done and count < 501:
            st, _ = model.prepare_state(s[0], s[1], s[2], s[3], s[4], s[5], s[6])
            a = model.get_action(np.array(st), False)
            s = sim.step(lin_velocity=(a[0] + 1) / 4, ang_velocity=a[1])
            count += 1
            done = bool(s[4]) or bool(s[5])
        goals += bool(s[5])
        col += bool(s[4])
    print("  Goal rate: %.1f%%  Collision: %.1f%%" % (goals * 10, col * 10))
    return goals / eval_episodes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune-ratio", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--episodes-per-epoch", type=int, default=70)
    ap.add_argument("--train-every-n", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    print("== 1/5 加载原始 CNNTD3(新权重)==")
    model = CNNTD3(state_dim=STATE_DIM, action_dim=ACTION_DIM, max_action=1,
                   device="cuda", save_every=0, load_model=True, model_name="CNNTD3",
                   load_directory=str(ROOT / "robot_nav/models/CNNTD3/checkpoint"))

    print("== 2/5 结构化剪枝 actor(%.0f%%)==" % (args.prune_ratio * 100))
    pruned = structured_prune(model.actor, args.prune_ratio)

    print("== 3/5 构建微调模型(剪枝 actor + 完整 critic)==")
    ft = PrunedCNNTD3(pruned, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                      max_action=1, device="cuda", save_every=0,
                      load_model=True, model_name="CNNTD3",
                      load_directory=str(ROOT / "robot_nav/models/CNNTD3/checkpoint"))

    print("== 4/5 微调 %d 轮 ==" % args.epochs)
    sim = SIM(world_file=str(ROOT / "robot_nav/worlds/robot_world.yaml"),
              disable_plotting=True, reward_mode="default")
    buffer = ReplayBuffer(buffer_size=50000, random_seed=666)

    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(0.0, 0.0)
    epoch, episode, steps = 0, 0, 0
    while epoch < args.epochs:
        state, _ = ft.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
        action = ft.get_action(np.array(state), True)
        a_in = [(action[0] + 1) / 4, action[1]]
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1])
        next_state, terminal = ft.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
        buffer.add(state, action, reward, terminal, next_state)
        if terminal or steps == args.max_steps:
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            episode += 1
            if episode % args.train_every_n == 0:
                ft.train(replay_buffer=buffer, iterations=args.iterations,
                         batch_size=args.batch_size)
            steps = 0
        else:
            steps += 1
        if (episode + 1) % args.episodes_per_epoch == 0:
            episode = 0
            epoch += 1
            rate = evaluate(ft, sim, eval_episodes=10)
            print("Epoch %d done, goal rate %.2f" % (epoch, rate), flush=True)

    print("== 5/5 保存 + 导出 ==")
    tag = "CNNTD3_pruned%d_ft" % int(args.prune_ratio * 100)
    ft.save(filename=tag, directory=str(ROOT / "robot_nav/models/CNNTD3/checkpoint"))
    ft.actor.eval()
    x = torch.randn(1, STATE_DIM)
    traced = torch.jit.trace(ft.actor.cpu(), x)
    out = ROOT / "exported" / "compressed" / (tag + ".pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(out))
    print("已保存:", out)
    print("评估: python eval_compare.py --policy jit --model exported/compressed/%s.pt "
          "--episodes 100 --world robot_world.yaml --seed 42" % tag)
