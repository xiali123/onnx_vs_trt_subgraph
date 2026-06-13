import os
import glob
import numpy as np
import onnx
from onnx_probe import GraphModel, AllTensorSelector, DumpBuilder, ORTRunner

model_path = os.path.join(os.path.dirname(__file__), "large_model.onnx")
model = GraphModel(model_path)

dump_dir = os.path.join(os.path.dirname(__file__), "dump_models_v1")

# Step 1: 按节点数分割成多个独立子图
builder = DumpBuilder(model, AllTensorSelector())
builder.build(save_dir=dump_dir, nodes_per_subgraph=100)

# Step 2: 级联推理 — 前一个子图的输出作为后一个子图的输入
carry = {"input": np.random.randn(1, 16, 28, 28).astype(np.float32)}

for dump_file in sorted(glob.glob(os.path.join(dump_dir, "dump_*.onnx"))):
    sub = onnx.load(dump_file)
    input_names = [inp.name for inp in sub.graph.input]

    inputs = {}
    for name in input_names:
        if name not in carry:
            raise KeyError(f"missing boundary input '{name}' for {os.path.basename(dump_file)}")
        inputs[name] = carry[name]

    runner = ORTRunner(dump_file)
    outputs = runner.dump(inputs, dump_dir)
    carry.update(outputs)
    print(f"{os.path.basename(dump_file)}: {len(outputs)} tensors")
