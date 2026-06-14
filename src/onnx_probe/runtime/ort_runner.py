import os
import re
import json
import numpy as np
import onnx
import onnxruntime as ort


def _safe_name(name):
    """Replace path-unsafe characters with underscores."""
    safe = re.sub(r"[<>:\"/\\|?*\s]", "_", name)
    safe = re.sub(r"_+", "_", safe)
    return safe or "unnamed"


class ORTRunner:
    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
        )

        # 从 ONNX 模型提取输入的 elem_type，用于 dtype 转换
        onnx_model = onnx.load(model_path)
        self._input_elem_types = {}
        for vi in onnx_model.graph.input:
            self._input_elem_types[vi.name] = vi.type.tensor_type.elem_type

    def _prepare_inputs(self, inputs):
        """根据 session 的输入信息，从 inputs 中提取并转换为正确的 numpy 类型。"""
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

        prepared = {}
        for inp in self.session.get_inputs():
            if inp.name not in inputs:
                continue
            arr = np.asarray(inputs[inp.name])
            elem_type = self._input_elem_types.get(inp.name)
            if elem_type is not None:
                expected = TENSOR_TYPE_TO_NP_TYPE[elem_type]
                if arr.dtype != expected:
                    arr = arr.astype(expected)
            prepared[inp.name] = arr
        return prepared

    def run(self, inputs, output_names=None):
        if output_names is None:
            output_names = [
                x.name for x in self.session.get_outputs()
            ]

        prepared = self._prepare_inputs(inputs)

        outputs = self.session.run(
            output_names,
            prepared,
        )

        return dict(zip(output_names, outputs))

    def dump(self, inputs, save_dir, min_bytes=0):
        os.makedirs(save_dir, exist_ok=True)

        outputs = self.run(inputs)

        new_mapping = {}
        tensor_sizes = {}
        for name, value in outputs.items():
            if value.nbytes < min_bytes:
                continue
            safe = _safe_name(name)
            new_mapping[name] = f"{safe}.npy"
            tensor_sizes[name] = {
                "bytes": int(value.nbytes),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }

        # merge name_mapping
        map_path = os.path.join(save_dir, "name_mapping.json")
        if os.path.exists(map_path):
            with open(map_path, "r") as f:
                mapping = json.load(f)
        else:
            mapping = {}
        mapping.update(new_mapping)
        with open(map_path, "w") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        # merge tensor_sizes
        sizes_path = os.path.join(save_dir, "tensor_sizes.json")
        if os.path.exists(sizes_path):
            with open(sizes_path, "r") as f:
                sizes = json.load(f)
        else:
            sizes = {}
        sizes.update(tensor_sizes)
        with open(sizes_path, "w") as f:
            json.dump(sizes, f, indent=2, ensure_ascii=False)

        # save tensors
        for name, value in outputs.items():
            if name in new_mapping:
                np.save(
                    os.path.join(save_dir, new_mapping[name]),
                    value
                )

        return outputs
