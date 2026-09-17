#!/bin/bash
# 表 4 成功率重跑矩阵 v2:7 档 × 100 局 seed42,每档独立输出文件(--tag, 并行安全)。
# 用法(主机):bash run_table4_matrix.sh
# 输出: results/<policy>_100ep_seed42_<tag>.json (每档唯一)
set -e
cd ~/DRL-robot-navigation-IR-SIM
PY=~/miniforge3/envs/furp_env/bin/python3

run_variant() { # $1=policy(jit|ort) $2=tag $3=model
  local pol=$1 tag=$2 model=$3
  $PY eval_compare.py --policy $pol --model "$model" --episodes 100 --seed 42 --tag $tag \
    > eval_t4v2_$tag.log 2>&1
  echo "[*] $tag done: $(grep 'Success:' eval_t4v2_$tag.log | head -1)"
}
export -f run_variant
export PY

# 并行 3 路(输出文件已唯一, 无覆盖风险)
run_variant jit baseline    exported/CNNTD3_actor.pt &
run_variant jit pruned10    exported/compressed/CNNTD3_pruned10.pt &
run_variant jit pruned30    exported/compressed/CNNTD3_pruned30.pt &
wait
run_variant jit pruned50    exported/compressed/CNNTD3_pruned50.pt &
run_variant jit ft50        exported/compressed/CNNTD3_pruned50_ft.pt &
run_variant ort fp32        exported/compressed/CNNTD3.onnx &
wait
run_variant ort int8        exported/compressed/CNNTD3_int8_realcalib.onnx &
wait
echo "=== v2 全部完成 ==="
