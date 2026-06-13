class SubgraphExtractor:
    @staticmethod
    def extract(graph_model, input_tensors, output_tensors):
        model = graph_model.clone()
        graph = model.graph
        tensor_map = graph.tensors()

        graph.inputs = [
            tensor_map[t if isinstance(t, str) else t.name]
            for t in input_tensors
        ]

        graph.outputs = [
            tensor_map[t if isinstance(t, str) else t.name]
            for t in output_tensors
        ]

        graph.cleanup()
        graph.toposort()
        return model
