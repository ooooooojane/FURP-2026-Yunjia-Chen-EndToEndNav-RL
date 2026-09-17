#!/bin/bash
# 通宵全量重跑: 场景manifest协议(确定性, 可配对)下的所有成功率数据。
# 背景: 旧协议场景生成不确定(同seed两次运行结果不同), 且DRL seed42用了旧终止逻辑。
# 新协议: make(world_seed=42)固定障碍布局 + 每局固定(起点,终点)场景manifest + 终止修复。
# 产出: DRL×3×2seed×500局 + NeuPAN×2×500局 + 压缩矩阵7档×200局, 全部同场景可配对。
set -e
cd ~/DRL-robot-navigation-IR-SIM
PY=~/miniforge3/envs/furp_env/bin/python3
NP=~/miniforge3/envs/neupan_env/bin/python3
M42=results/manifest_500_seed42.json
M7=results/manifest_500_seed7.json
M42_200=results/manifest_200_seed42.json

echo "=== [0/4] 生成场景manifest ==="
$PY eval_compare.py --policy jit --model exported/CNNTD3_actor.pt --episodes 1 --seed 42 --gen-manifest $M42 --n-scenes 500 2>&1 | tail -1
$PY eval_compare.py --policy jit --model exported/CNNTD3_actor.pt --episodes 1 --seed 7 --gen-manifest $M7 --n-scenes 500 2>&1 | tail -1
$PY eval_compare.py --policy jit --model exported/CNNTD3_actor.pt --episodes 1 --seed 42 --gen-manifest $M42_200 --n-scenes 200 2>&1 | tail -1

echo "=== [1/4] DRL 三模型 × seed42 500局 ==="
$PY eval_compare.py --policy cnntd3 --episodes 500 --seed 42 --tag m42 --manifest $M42 > ~/t3_cnntd3_m42.log 2>&1 &
$PY eval_compare.py --policy sac    --episodes 500 --seed 42 --tag m42 --manifest $M42 > ~/t3_sac_m42.log 2>&1 &
$PY eval_compare.py --policy ppo    --episodes 500 --seed 42 --tag m42 --manifest $M42 > ~/t3_ppo_m42.log 2>&1 &
wait
echo "=== [2/4] DRL 三模型 × seed7 500局 ==="
$PY eval_compare.py --policy cnntd3 --episodes 500 --seed 7 --tag m7 --manifest $M7 > ~/t3_cnntd3_m7.log 2>&1 &
$PY eval_compare.py --policy sac    --episodes 500 --seed 7 --tag m7 --manifest $M7 > ~/t3_sac_m7.log 2>&1 &
$PY eval_compare.py --policy ppo    --episodes 500 --seed 7 --tag m7 --manifest $M7 > ~/t3_ppo_m7.log 2>&1 &
wait
echo "=== [3/4] NeuPAN × 2 seed 各500局 ==="
$NP eval_compare.py --policy neupan --episodes 500 --seed 42 --tag m42 --manifest $M42 \
    --neupan-repo ~/github/neupan --planner-yaml ~/planner_diff_small_best.yaml > ~/t3_neupan_m42.log 2>&1 &
$NP eval_compare.py --policy neupan --episodes 500 --seed 7 --tag m7 --manifest $M7 \
    --neupan-repo ~/github/neupan --planner-yaml ~/planner_diff_small_best.yaml > ~/t3_neupan_m7.log 2>&1 &
wait
echo "=== [4/4] 压缩矩阵 7档 × 200局(manifest seed42) ==="
for v in "jit baseline exported/CNNTD3_actor.pt" \
         "jit pruned10 exported/compressed/CNNTD3_pruned10.pt" \
         "jit pruned30 exported/compressed/CNNTD3_pruned30.pt"; do
  set -- $v
  $PY eval_compare.py --policy $1 --model $3 --episodes 200 --seed 42 --tag $2 --manifest $M42_200 > ~/t4_$2.log 2>&1 &
done
wait
for v in "jit pruned50 exported/compressed/CNNTD3_pruned50.pt" \
         "jit ft50 exported/compressed/CNNTD3_pruned50_ft.pt" \
         "ort fp32 exported/compressed/CNNTD3.onnx"; do
  set -- $v
  $PY eval_compare.py --policy $1 --model $3 --episodes 200 --seed 42 --tag $2 --manifest $M42_200 > ~/t4_$2.log 2>&1 &
done
wait
$PY eval_compare.py --policy ort --model exported/compressed/CNNTD3_int8_realcalib.onnx \
    --episodes 200 --seed 42 --tag int8 --manifest $M42_200 > ~/t4_int8.log 2>&1
echo "=== 全部完成 $(date +%H:%M) ==="
