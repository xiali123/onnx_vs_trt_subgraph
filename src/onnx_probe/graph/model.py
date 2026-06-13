import onnx
from onnx.shape_inference import infer_shapes
import onnx_graphsurgeon as gs


class GraphModel:

    def __init__(self, model_path=None):

        self.model_path = model_path

        self.onnx_model = None
        self.graph = None

        if model_path is not None:

            self.onnx_model = onnx.load(
                model_path
            )

            self.onnx_model = infer_shapes(self.onnx_model)

            self.graph = gs.import_onnx(
                self.onnx_model
            )

    @property
    def nodes(self):
        if self.graph is None:
            raise ValueError("Model not loaded, call __init__ with a model_path")
        return self.graph.nodes

    @property
    def tensors(self):
        if self.graph is None:
            raise ValueError("Model not loaded, call __init__ with a model_path")
        return self.graph.tensors()

    @property
    def inputs(self):
        if self.graph is None:
            raise ValueError("Model not loaded, call __init__ with a model_path")
        return self.graph.inputs

    @property
    def outputs(self):
        if self.graph is None:
            raise ValueError("Model not loaded, call __init__ with a model_path")
        return self.graph.outputs

    def clone(self):

        new_model = GraphModel()

        new_model.model_path = self.model_path

        new_model.onnx_model = infer_shapes(
            gs.export_onnx(self.graph)
        )

        new_model.graph = gs.import_onnx(
            new_model.onnx_model
        )

        return new_model

    def save(self, save_path):

        self.graph.cleanup()
        self.graph.toposort()

        onnx.save(
            gs.export_onnx(self.graph),
            save_path
        )
