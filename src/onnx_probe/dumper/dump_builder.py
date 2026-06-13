import os
import onnx
from onnx import helper


class DumpBuilder:
    def __init__(self, graph_model, tensor_selector=None):
        self.graph_model = graph_model
        self.tensor_selector = tensor_selector

    def build(self, save_dir, nodes_per_subgraph=100):
        os.makedirs(save_dir, exist_ok=True)

        onnx_model = self.graph_model.onnx_model
        graph = onnx_model.graph
        all_nodes = list(graph.node)

        # name -> ValueInfoProto
        type_map = {}
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            type_map[vi.name] = vi

        # name -> TensorProto (initializer)
        init_map = {init.name: init for init in graph.initializer}

        # which tensor names to output (None = all)
        if self.tensor_selector:
            selected = self.tensor_selector.select(self.graph_model.graph)
            selected_names = {t.name for t in selected}
        else:
            selected_names = None

        total_chunks = (len(all_nodes) + nodes_per_subgraph - 1) // nodes_per_subgraph
        zfill = len(str(total_chunks))

        for chunk_idx in range(0, len(all_nodes), nodes_per_subgraph):
            chunk = all_nodes[chunk_idx:chunk_idx + nodes_per_subgraph]

            # --- tensors produced & consumed within this chunk ---
            produced = set()
            chunk_output_names = []
            for n in chunk:
                for out_name in n.output:
                    produced.add(out_name)
                    if selected_names is None or out_name in selected_names:
                        chunk_output_names.append(out_name)

            consumed = set()
            for n in chunk:
                for inp_name in n.input:
                    consumed.add(inp_name)

            external_input_names = consumed - produced

            # --- subgraph inputs ---
            sub_inputs = []
            for name in external_input_names:
                if name in type_map:
                    sub_inputs.append(type_map[name])

            # --- subgraph outputs ---
            sub_outputs = []
            for name in chunk_output_names:
                if name in type_map:
                    sub_outputs.append(type_map[name])

            # --- initializers needed by this chunk ---
            needed_init_names = set()
            for n in chunk:
                for inp in n.input:
                    if inp in init_map:
                        needed_init_names.add(inp)
            sub_inits = [init_map[name] for name in needed_init_names]

            # --- build subgraph ---
            sub_idx = chunk_idx // nodes_per_subgraph
            sub_graph = helper.make_graph(
                list(chunk),
                f"subgraph_{sub_idx}",
                sub_inputs,
                sub_outputs,
                sub_inits,
            )
            sub_model = helper.make_model(
                sub_graph,
                opset_imports=onnx_model.opset_import,
            )

            save_path = os.path.join(save_dir, f"dump_{str(sub_idx).zfill(zfill)}.onnx")
            onnx.save(sub_model, save_path)
