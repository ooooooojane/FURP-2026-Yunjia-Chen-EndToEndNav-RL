#!/bin/bash
# v5 全量: 单进程多方法评估(配对由构造保证, 无跨进程一致性问题)。
# A/B: 表3 双seed × [cnntd3,sac,ppo,neupan] 各500场景 (并行)
# C:   表4 压缩矩阵 7档 × 200场景
set -e
cd ~/DRL-robot-navigation-IR-SIM
PY=~/miniforge3/envs/neupan_env/bin/python3

echo "=== [A] seed42: cnntd3,sac,ppo,neupan × 500场景 ==="
$PY eval_multi_policy.py --n-scenes 500 --seed 42 --policies "cnntd3,sac,ppo,neupan" \
    --out-prefix results/mp42 > ~/mp42.log 2>&1 &
echo "=== [B] seed7: cnntd3,sac,ppo,neupan × 500场景 ==="
$PY eval_multi_policy.py --n-scenes 500 --seed 7 --policies "cnntd3,sac,ppo,neupan" \
    --out-prefix results/mp7 > ~/mp7.log 2>&1 &
wait
echo "=== [C] 压缩矩阵 7档 × 200场景 ==="
$PY eval_multi_policy.py --n-scenes 200 --seed 42 \
    --policies "jit:baseline:exported/CNNTD3_actor.pt,jit:pruned10:exported/compressed/CNNTD3_pruned10.pt,jit:pruned30:exported/compressed/CNNTD3_pruned30.pt,jit:pruned50:exported/compressed/CNNTD3_pruned50.pt,jit:ft50:exported/compressed/CNNTD3_pruned50_ft.pt,ort:fp32:exported/compressed/CNNTD3.onnx,ort:int8:exported/compressed/CNNTD3_int8_realcalib.onnx" \
    --out-prefix results/mp_matrix > ~/mp_matrix.log 2>&1
echo "=== v5 全部完成 $(date +%H:%M) ==="
