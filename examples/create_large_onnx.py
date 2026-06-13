import onnx
from onnx import helper, TensorProto
import numpy as np
import os

C, H, W = 16, 28, 28
shape = [1, C, H, W]

nodes = []
init = []
node_id = [0]  # mutable counter
tensor_id = [0]

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)


def t(name):
    """Generate unique tensor name."""
    tensor_id[0] += 1
    return f"{name}_{tensor_id[0]}"


def conv(x, out_c, kernel, name):
    k = kernel
    w_n = t(f"{name}_w")
    b_n = t(f"{name}_b")
    y = t(f"{name}_out")
    node_id[0] += 1
    init.append(helper.make_tensor(w_n, TensorProto.FLOAT, [out_c, C, k, k],
                np.random.randn(out_c, C, k, k).astype(np.float32).tobytes(), raw=True))
    init.append(helper.make_tensor(b_n, TensorProto.FLOAT, [out_c],
                np.random.randn(out_c).astype(np.float32).tobytes(), raw=True))
    nodes.append(helper.make_node(
        "Conv", [x, w_n, b_n], [y],
        kernel_shape=[k, k], pads=[k // 2] * 4, strides=[1, 1]
    ))
    return y


def relu(x, name):
    y = t(f"{name}_relu")
    node_id[0] += 1
    nodes.append(helper.make_node("Relu", [x], [y]))
    return y


def add(a, b, name):
    y = t(f"{name}_add")
    node_id[0] += 1
    nodes.append(helper.make_node("Add", [a, b], [y]))
    return y


def concat(xs, name):
    y = t(f"{name}_cat")
    node_id[0] += 1
    nodes.append(helper.make_node("Concat", xs, [y], axis=1))
    return y


def sigmoid(x, name):
    y = t(f"{name}_sig")
    node_id[0] += 1
    nodes.append(helper.make_node("Sigmoid", [x], [y]))
    return y


def tanh(x, name):
    y = t(f"{name}_tanh")
    node_id[0] += 1
    nodes.append(helper.make_node("Tanh", [x], [y]))
    return y


def gemm(x, name):
    w_n = t(f"{name}_w")
    b_n = t(f"{name}_b")
    y = t(f"{name}_out")
    node_id[0] += 1
    init.append(helper.make_tensor(w_n, TensorProto.FLOAT, [10, C * H * W],
                np.random.randn(10, C * H * W).astype(np.float32).tobytes(), raw=True))
    init.append(helper.make_tensor(b_n, TensorProto.FLOAT, [10],
                np.random.randn(10).astype(np.float32).tobytes(), raw=True))
    nodes.append(helper.make_node("Gemm", [x, w_n, b_n], [y], transB=1))
    return y


prev = "input"
skip_bank = []  # sliding window of past outputs for skip connections

N_BLOCKS = 330

for i in range(N_BLOCKS):
    # ── 3 parallel branches ──
    # Branch A: 3x3 Conv -> Relu
    a = conv(prev, C, 3, f"b{i}_a")
    a = relu(a, f"b{i}_a")

    # Branch B: 1x1 Conv -> Relu -> 3x3 Conv -> Relu
    b = conv(prev, C, 1, f"b{i}_b1")
    b = relu(b, f"b{i}_b1")
    b = conv(b, C, 3, f"b{i}_b2")
    b = relu(b, f"b{i}_b2")

    # Branch C: Sigmoid -> Tanh (nonlinear path)
    c = sigmoid(prev, f"b{i}_c1")
    c = tanh(c, f"b{i}_c2")
    # Scale back with 1x1 conv
    c = conv(c, C, 1, f"b{i}_c3")
    c = relu(c, f"b{i}_c3")

    # ── Merge A + B + C ──
    ab = add(a, b, f"b{i}_ab")
    abc = add(ab, c, f"b{i}_abc")
    merged = relu(abc, f"b{i}_merge")

    # ── Skip connections ──
    # Short skip: every 3 blocks, add skip from 3 blocks ago
    if i >= 3:
        merged = add(merged, skip_bank[-3], f"b{i}_short_skip")
        merged = relu(merged, f"b{i}_ss_relu")

    # Long skip: every 50 blocks, add skip from 50 blocks ago
    if i >= 50 and i % 50 == 0:
        merged = add(merged, skip_bank[-50], f"b{i}_long_skip")
        merged = relu(merged, f"b{i}_ls_relu")

    skip_bank.append(merged)
    prev = merged

# ── Head ──
# Flatten -> Gemm
from onnx import helper as _h
flat = t("flat")
node_id[0] += 1
nodes.append(helper.make_node("Flatten", [prev], [flat], axis=1))
out = gemm(flat, "head")
out = relu(out, "head")

out_vi = helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, 10])

graph = helper.make_graph(nodes, "complex_model", [inp], [out_vi], init)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

save_path = os.path.join(os.path.dirname(__file__), "large_model.onnx")
onnx.save(model, save_path)

print(f"Nodes: {len(nodes)}  Initializers: {len(init)}")
print(f"File: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")
print(f"Saved: {save_path}")
