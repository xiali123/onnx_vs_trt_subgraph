class TensorSelector:
    def select(self, graph):
        raise NotImplementedError

class AllTensorSelector(TensorSelector):
    def select(self, graph):
        outputs = []
        for node in graph.nodes:
            outputs.extend(node.outputs)
        return outputs

class OpTensorSelector(TensorSelector):
    def __init__(self, op_types):
        self.op_types = set(op_types)

    def select(self, graph):
        outputs = []
        for node in graph.nodes:
            if node.op in self.op_types:
                outputs.extend(node.outputs)
        return outputs
