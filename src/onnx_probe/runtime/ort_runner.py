import os
import re
import json
import numpy as np
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

    def run(self, inputs, output_names=None):
        if output_names is None:
            output_names = [
                x.name for x in self.session.get_outputs()
            ]

        outputs = self.session.run(
            output_names,
            inputs,
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
