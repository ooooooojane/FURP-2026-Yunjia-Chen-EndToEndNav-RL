# Source code

Everything behind *Benchmarking Edge Deployment of DRL Navigation Policies:
Latency, Compression, and Real-Robot Validation*.

## Layout

```
src/
├── scripts/          working directory — self-contained, cd in and run
├── paper_data/       source data for every number in the report (see its README)
└── patches/          what we changed in the upstream project, as a diff
```

## Where the training/simulation code lives

The training and simulation stack is **not vendored here**. It is the
open-source project
[`reiniscimurs/DRL-robot-navigation-IR-SIM`](https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM)
— which supplies the CNNTD3, SAC and PPO implementations benchmarked in the
report. Our modifications to it are in [`patches/`](patches/) as a single diff
against upstream commit `31e1a4d` (4 files, +201/−150).

The copy under `scripts/robot_nav/` is the patched source tree, included so the
scripts below can be read in context. **To reproduce a training run, apply the
patch to a fresh upstream clone** rather than using this copy directly — see
`patches/README.md` for the exact commands.

## `scripts/` — what each file does

All scripts assume the working directory is `scripts/`.

### Core — the report's results come from these

| Script | What it does | Feeds |
|---|---|---|
| `eval_multi_policy.py` | Evaluates CNNTD3 / SAC / PPO / NeuPAN in a single process on **identical** scenarios; writes `mp42_*.json`, `mp7_*.json`, `mp_matrix_*.json` | Table 3, Table 4 success |
| `latency_repeat_batches.py` | 5 independent batches × 1000 forward passes per (model, device), with per-platform synchronisation | **Table 2** (authoritative) |
| `latency_bench_unified.py` | PyTorch FP32 vs ORT FP32 vs ORT INT8 in one protocol on one batch | Table 4 latency |
| `latency_bench_robot.py` | On-board (Jetson) model-only latency using real navigation states as input | Table 5 model-only |
| `quantize_bench.py` | Structured L2 pruning (10/30/50%) + ONNX static INT8 with real calibration | Table 4 size |
| `finetune_pruned.py` | Prunes an actor, then TD3 fine-tunes it to recover success rate | Table 4 "pruned + FT" row |
| `export_torchscript.py` | Traces trained variants to TorchScript; unwraps SAC's distribution to `.mean` | prerequisite for all latency measurement |
| `reexport_onnx_ir9.py` | Rewrites ONNX to `ir_version=9` so the robot's onnxruntime 1.16.x can load it | required for on-robot INT8 |
| `rl_nav_node_ros1.py` | **The deployed node.** `/scan` → 185-dim state → CNNTD3 → `/cmd_vel`, 0.20 m safety layer, odometry-defined arrival | Table 5, real robot |
| `eval_matched_scene.py` | Matched physical↔simulation diagnostic with shield modes A–D | matched-geometry discussion |
| `run_all_v5.sh` | Full sweep driver: Table 3 dual-seed × 4 methods × 500 scenarios + Table 4 matrix × 200 scenarios | reproduction entry point |
| `run_overnight_all.sh` | Full re-run under the deterministic scenario-manifest protocol | produced the final v3 data |
| `run_table4_matrix.sh` | Table 4 success matrix, 7 tiers, parallel | Table 4 success |

### Supporting

| Script | What it does |
|---|---|
| `make_figures/make_paper_figures.py` | Generates the report's IEEE-ready latency and compression figures |
| `verify_model.py` | Post-training weight sanity check (probe + quick eval) |
| `results/bag_metrics.py` | Extracts per-bag metrics (odom / cmd_vel / goal timing) from a ROS1 bag |
| `yaw_cal.py` | Spins the robot 360° and measures odom error, suggests `odom_z_scale` |
| `turn_openloop_test.py` | Open-loop fixed-ω step; odom-integrated vs physical yaw |
| `latency_bench_all.py` | Single-device batch latency re-test (superseded by `latency_repeat_batches.py`, kept for the earlier protocol) |

### The two ROS nodes — read this before getting confused

There are two, and only one of them was deployed:

- **`rl_nav_node_ros1.py`** — the node actually run on the WHEELTEC S100
  (ROS1 Noetic, Jetson Orin). All real-robot results in the report come from
  this one.
- **`ros2_ws/src/rl_nav_node/`** — an earlier ROS2 skeleton (Week 5), written
  against a Turtlebot3 Burger + Orange Pi + LDS-02 LiDAR and never deployed.
  Kept as a record of the bring-up work; it is **not** the node behind Table 5.

### Deliberately not included

- **`eval_compare.py`** — no copy on disk matches any hash in
  `PROVENANCE_MANIFEST.json`, and its outputs were withdrawn (they produced the
  void 75.2 / 89.6 / 25.0 / 3.3 success rates). Publishing it would hand a
  reviewer a file that cannot be reconciled with the registry.
- **Raw ROS `.bag` recordings** (456 MB) — kept off-repo. `bag_metrics.py`
  documents the expected format; per-trial outcomes are in `paper_data/`.
- **Trained checkpoints** (`.pth`, 51 MB; TorchScript `.pt`, 14 MB) — excluded
  to keep the repository light. Training is reproducible from the patch and the
  logged hyperparameters; the exact training logs are in `scripts/training_logs/`.
  **One exception:** `ros2_ws/src/rl_nav_node/cnntd3_actor.pt` (552 KB) *is*
  included, because `inference_node.py` loads it by relative path and the ROS2
  package is unusable without it.
- **Upstream files we never used** — `rnn_*`, `rvo_*`, `marl_*` experiment entry
  points and the `DDPG` / `HCM` / `MARL` / `RCPG` model implementations are
  inherited from upstream and irrelevant to this work. They remain in
  `robot_nav/` only because `rl_train.py`'s model factory imports across the
  full algorithm set.

## `scripts/training_logs/`

Training evidence for the reward-sensitivity study (Weeks 4–5): per-run console
logs, `epoch_table.csv` / `.md` (epoch-by-epoch comparison), reward-sensitivity
plots, and failure-mode breakdowns (`failure_*.json`).

## Running things

The work spanned three environments, so there is no single `requirements.txt`
covering all of them:

| Environment | Used for |
|---|---|
| Ubuntu host + RTX 4070 (`torch 2.13.0+cu130`) | training, simulation, NeuPAN, GPU latency |
| macOS / Apple M5 (CPU + MPS) | CPU latency, quantization and pruning, reporting |
| WHEELTEC S100 on-board (Jetson Orin, ROS1 Noetic) | real-robot deployment and on-board latency |

Dependencies follow the upstream project (`pyproject.toml` / `poetry.lock` in
the upstream repo), plus `onnx`, `onnxruntime` and `torch` for the compression
work. The on-robot environment needed `onnxruntime 1.16.x` on Python 3.8
(aarch64) — hence `reexport_onnx_ir9.py`.
