#!/usr/bin/env python3
"""将 ONNX 的 ir_version 降为 9(兼容机器人 onnxruntime 1.16.x)。

模型为小型 CNN+MLP, IR9 完全兼容; 直接改字段即可, 无需重新导出/量化。
产出: exported/compressed/CNNTD3_ir9.onnx + CNNTD3_int8_ir9.onnx
"""
from pathlib import Path

import onnx

OUT = Path(__file__).resolve().parent / "exported" / "compressed"

for src, dst in [("CNNTD3.onnx", "CNNTD3_ir9.onnx"),
                 ("CNNTD3_int8_realcalib.onnx", "CNNTD3_int8_ir9.onnx")]:
    m = onnx.load(str(OUT / src))
    print(f"{src}: ir_version={m.ir_version} opset={m.opset_import[0].version}")
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, str(OUT / dst))
    print(f"  -> {dst}: ir_version={m.ir_version} 尺寸={ (OUT/dst).stat().st_size//1024 }KB")
print("[*] 完成")
