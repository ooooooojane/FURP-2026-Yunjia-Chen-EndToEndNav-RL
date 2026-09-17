# TD3 vs SAC vs PPO — 完整实验数据

> 所有模型均训练 60 epochs，评估设置：eval_episodes=10，max_steps=500
> 泛化测试：在训练时未见过的 circle/cross/eval 三张地图上各跑 10 episodes
> 失败分析：在训练地图 (robot_world) 上跑 500 episodes，分类记录失败原因

---

## 1. 训练性能 (Last 20 Epochs Average)

| Reward | TD3 Goal | TD3 Reward | TD3 Collision | SAC Goal | SAC Reward | SAC Collision | PPO Goal | PPO Reward |
|--------|----------|------------|---------------|----------|------------|---------------|----------|------------|
| Default | **0.94** | **88.8** | 0.06 | 0.925 | 82.9 | 0.05 | 0.22 | −24.8 |
| Dense | 0.84 | 27.4 | — | **0.93** | **53.8** | — | 0.63 | 41.7 |
| Harsh | 0.825 | 43.8 | — | **0.855** | **50.9** | — | 0.37 | −72.1 |
| Sparse | ❌ 放弃 (3ep) | — | — | — | — | — | 0.53 | 17.5 |

| δ (Default→Harsh) | TD3 | SAC | PPO |
|---------------------|------|------|------|
| Goal Rate 降幅 | −11.5pp | −7.0pp | N/A (Default 太低无法比较) |

---

## 2. 泛化测试 (Zero-Shot on Unseen Maps)

| Model | Reward | Circle | Cross | Eval | **Avg** |
|-------|--------|--------|-------|------|---------|
| **TD3** | Default | 10/10 | 10/10 | 10/10 | **100%** |
| | Dense | 10/10 | 10/10 | 9/10 | **96.7%** |
| | Harsh | 10/10 | 10/10 | 9/10 | **96.7%** |
| **SAC** | Default | 10/10 | 10/10 | 6/10 | **86.7%** |
| | Dense | 10/10 | 9/10 | 8/10 | **90.0%** |
| | Harsh | 9/10 | 10/10 | 5/10 | **80.0%** |
| **PPO** | Default | 1/10 | 1/10 | 1/10 | **10.0%** |
| | Dense | 5/10 | 8/10 | 6/10 | **63.3%** |
| | Harsh | 4/10 | 4/10 | 2/10 | **33.3%** |
| | Sparse | 9/10 | 7/10 | 2/10 | **60.0%** |

| 泛化保持率 | TD3 | SAC | PPO |
|-----------|------|------|------|
| Default | 100% | 86.7% | 10% |
| Dense | 96.7% | 90% | 63.3% |
| Harsh | 96.7% | 80% | 33.3% |

---

## 3. 失败分析 (500 Episodes on Training Map)

### 3.1 TD3 Default

| | Count | Rate |
|---|---|---|
| **Success** | 466 | **93.2%** |
| Collision (total) | 30 | 6.0% |
| ├ tight_space (<0.5m) | 18 | 3.6% |
| ├ near_goal (<2m) | 11 | 2.2% |
| └ early (<30 steps) | 1 | 0.2% |
| Timeout | 4 | 0.8% |

### 3.2 SAC

| | Default | Dense | Harsh |
|---|---|---|---|
| **Success** | 457 (91.4%) | 447 (89.4%) | 445 (89.0%) |
| Collision (total) | 36 | 53 | 48 |
| ├ tight_space | 21 | 35 | 26 |
| ├ near_goal | 15 | 18 | 21 |
| └ early | 0 | 0 | 1 |
| Timeout | 8 | 0 | 7 |

### 3.3 PPO

| | Default | Dense | Harsh |
|---|---|---|---|
| **Success** | 114 (22.8%) | 315 (63.0%) | 177 (35.4%) |
| Collision (total) | 301 | 157 | 265 |
| ├ tight_space | **243 (48.6%)** | 112 | **209 (41.8%)** |
| └ near_goal | 58 | 45 | 56 |
| Timeout | 85 | 28 | 58 |

---

## 4. 关键发现

| 发现 | 数据 |
|------|------|
| **TD3 泛化最强** | Default 100% 泛化保持，Dense/Harsh 97% |
| **SAC 对 Reward 最鲁棒** | Default→Harsh 仅降 7pp，Dense 甚至超 Default |
| **PPO 在本任务中全面落后** | 最佳 Dense 仅 63%，Default 基本是随机 |
| **窄缝 (tight_space) 是所有模型的头号杀手** | TD3 60%, SAC 58%, PPO 71% 的碰撞来自窄缝 |
| **Harsh 未解决窄缝问题** | SAC Harsh 的 tight_space 碰撞 (26) 反而比 Default (21) 多 |
| **Dense 0 Timeout** | SAC Dense 从不超时——激进策略，要么到要么撞 |
| **Eval 地图是所有模型的天花板** | TD3 90%, SAC 50-80%, PPO 10-20% |
