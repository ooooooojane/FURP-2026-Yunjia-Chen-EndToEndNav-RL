#!/usr/bin/env python3
"""把全部模型变体导出为 TorchScript(.pt),统一延迟测量口径。

用法: cd ~/Desktop/t7_extract && python export_torchscript.py
输出: ~/Desktop/t7_extract/exported/*.pt  (平台无关,Mac/主机通用)
"""
import sys
from pathlib import Path

import torch

# 检查点是在 GPU 机器上训练的,Mac 无 CUDA——强制 map_location="cpu" 加载
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "map_location": "cpu"})

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
from robot_nav.models.PPO.PPO import PPO
from robot_nav.models.SAC.SAC import SAC

STATE_DIM, ACTION_DIM, MAX_ACTION = 185, 2, 1
CKPT = "robot_nav/models"  # 相对 ROOT
OUT = ROOT / "exported"
OUT.mkdir(exist_ok=True)


def export(cls, model_name, attr, tag):
    load_dir = ROOT / CKPT / cls.__name__ / "checkpoint"
    # 跳过没有对应检查点的变体
    candidates = list(load_dir.glob(f"{model_name}_*"))
    if not candidates:
        print(f"[skip] {model_name}: 无检查点")
        return
    try:
        model = cls(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, max_action=MAX_ACTION,
            device="cpu", save_every=0, load_model=True,
            model_name=model_name, load_directory=load_dir,
        )
    except TypeError:
        # SAC: device 在 max_action 前面,且用关键字更稳
        model = cls(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, device="cpu",
            max_action=MAX_ACTION, save_every=0, load_model=True,
            model_name=model_name, load_directory=load_dir,
        )
    net = getattr(model, attr)
    if attr == "policy":  # PPO: policy 是 ActorCritic,导出其中的 actor
        net = net.actor
    net.eval()
    x = torch.randn(1, STATE_DIM)

    if cls is SAC:
        # SAC actor 输出 SquashedNormal 分布,trace 包装层取确定性均值
        import torch.nn as nn
        wrapper = nn.Module()
        wrapper.actor = net
        wrapper.forward = lambda x: wrapper.actor(x).mean  # noqa: E731
        net = wrapper
    traced = torch.jit.trace(net, x)
    fn = OUT / f"{tag}.pt"
    traced.save(str(fn))
    print(f"[OK] {tag}.pt  <- {model_name} ({fn.stat().st_size//1024}KB)")


for variant in ["", "_dense", "_harsh", "_sparse"]:
    export(CNNTD3, f"CNNTD3{variant}", "actor", f"CNNTD3{variant}_actor")
for variant in ["", "_dense", "_harsh"]:
    export(SAC, f"SAC{variant}", "actor", f"SAC{variant}_actor")
for variant in ["", "_dense", "_harsh", "_sparse"]:
    export(PPO, f"PPO{variant}", "policy", f"PPO{variant}_actor")

print("\n完成。输出目录:", OUT)
