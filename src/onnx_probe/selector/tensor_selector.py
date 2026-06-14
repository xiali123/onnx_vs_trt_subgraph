from onnx_probe.dumper.memory_estimator import MemoryEstimator


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


class MemoryThresholdSelector(TensorSelector):
    """Only select tensors whose byte size is >= *min_bytes*.

    This gives fine-grained control over dump granularity: large
    activation tensors are dumped while tiny shape/ index tensors are
    skipped, reducing dump I/O and downstream comparison noise.
    """

    def __init__(self, min_bytes=1024):
        self.min_bytes = min_bytes

    def select(self, graph):
        outputs = []
        for node in graph.nodes:
            for t in node.outputs:
                if MemoryEstimator.from_onnx_tensor(t) >= self.min_bytes:
                    outputs.append(t)
        return outputs


class CompositeSelector(TensorSelector):
    """Combine multiple selectors: a tensor is selected if ANY selector picks it."""

    def __init__(self, selectors):
        self.selectors = list(selectors)

    def select(self, graph):
        selected_sets = []
        for sel in self.selectors:
            selected_sets.append({t.name for t in sel.select(graph)})
        if not selected_sets:
            return []
        union_names = set.union(*selected_sets)
        # return the actual tensor objects from the first selector's graph
        tensor_map = {t.name: t for t in self.selectors[0].select(graph)
                      if hasattr(self.selectors[0], 'select')}
        # rebuild from graph to be safe
        all_tensors = {t.name: t for node in graph.nodes for t in node.outputs}
        return [all_tensors[name] for name in union_names if name in all_tensors]
