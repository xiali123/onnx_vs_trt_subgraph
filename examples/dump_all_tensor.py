import os
import glob
import numpy as np
import onnx
from onnx_probe import GraphModel, AllTensorSelector, DumpBuilder, ORTRunner

model_path = os.path.join(os.path.dirname(__file__), "large_model.onnx")
model = GraphModel(model_path)

dump_dir = os.path.join(os.path.dirname(__file__), "dump_models_v1")

# Step 1: 按内存预算分割成多个独立子图 (默认512MB)
builder = DumpBuilder(model, AllTensorSelector())
builder.build(save_dir=dump_dir, memory_budget_mb=512)
# 如需按节点数分割：builder.build(save_dir=dump_dir, nodes_per_subgraph=100)

# Step 2: 级联推理 — 前一个子图的输出作为后一个子图的输入
#
# input_dict: 初始真值输入 {input_name: numpy_array}，不传则随机生成
input_dict = None  # 替换为 {"input": np.load("real_input.npy")}


def cascade_dump(model, dump_dir, input_dict=None):
    # ── 初始化 carry ──
    carry = {}
    if input_dict:
        carry.update(input_dict)
    else:
        for inp in model.inputs:
            shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
            carry[inp.name] = (np.random.randn(*shape) * 0.02).astype(np.float32)

    dump_files = sorted(glob.glob(os.path.join(dump_dir, "dump_*.onnx")))

    # 预扫描：每个子图需要哪些运行时输入（排除 initializer）
    future_runtime_inputs = []
    for df in dump_files:
        sub = onnx.load(df)
        init_names = {init.name for init in sub.graph.initializer}
        runtime_inputs = {inp.name for inp in sub.graph.input if inp.name not in init_names}
        future_runtime_inputs.append(runtime_inputs)

    for i, dump_file in enumerate(dump_files):
        sub = onnx.load(dump_file)
        input_names = [inp.name for inp in sub.graph.input]

        # 从 carry 中提取当前子图需要的输入
        inputs = {}
        for name in input_names:
            if name in carry:
                inputs[name] = carry[name]
            else:
                inits = {init.name: init for init in sub.graph.initializer}
                if name in inits:
                    continue  # initializer 已内嵌在 onnx 中
                raise KeyError(f"missing boundary input '{name}' for {os.path.basename(dump_file)}")

        runner = ORTRunner(dump_file)
        outputs = runner.dump(inputs, dump_dir)
        carry.update(outputs)

        # 释放后续子图不再需要的 tensor
        still_needed = set()
        for j in range(i + 1, len(dump_files)):
            still_needed |= future_runtime_inputs[j]
        for name in list(carry.keys()):
            if name not in still_needed:
                del carry[name]

        print(f"{os.path.basename(dump_file)}: {len(outputs)} tensors, "
              f"carry={len(carry)} tensors "
              f"({sum(v.nbytes for v in carry.values()) / 1024**2:.1f} MB)")


cascade_dump(model, dump_dir, input_dict)
