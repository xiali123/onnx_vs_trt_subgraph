import os
import numpy as np
import onnxruntime as ort

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

    def dump(self, inputs, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        outputs = self.run(inputs)

        for name, value in outputs.items():
            safe_name = name.lstrip("/").replace("/", "_")
            np.save(
                os.path.join(save_dir, f"{safe_name}.npy"),
                value
            )

        return outputs
