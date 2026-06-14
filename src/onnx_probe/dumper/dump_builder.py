import os
import json
import onnx
from onnx import helper

from onnx_probe.dumper.memory_estimator import MemoryEstimator


class DumpBuilder:
    """Split an ONNX model into subgraphs, each fitting within a memory budget.

    Two modes (mutually exclusive):
      - ``memory_budget_mb`` — partition by peak memory (initializers + activations).
      - ``nodes_per_subgraph``  — fallback: partition by fixed node count.
    """

    def __init__(self, graph_model, tensor_selector=None):
        self.graph_model = graph_model
        self.tensor_selector = tensor_selector

    # ── public ──────────────────────────────────────────────────────────

    def build(self, save_dir, memory_budget_mb=512, nodes_per_subgraph=None):
        os.makedirs(save_dir, exist_ok=True)

        onnx_model = self.graph_model.onnx_model
        graph = onnx_model.graph
        all_nodes = list(graph.node)

        # ---- pre-compute lookup tables ----
        type_map = {}
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            type_map[vi.name] = vi

        init_map = {init.name: init for init in graph.initializer}

        selected_names = None
        if self.tensor_selector:
            selected = self.tensor_selector.select(self.graph_model.graph)
            selected_names = {t.name for t in selected}

        # ---- partition ----
        if nodes_per_subgraph is not None:
            chunks, chunk_memories = self._split_by_node_count(all_nodes, nodes_per_subgraph)
            strategy = "node_count"
        else:
            budget_bytes = int(memory_budget_mb * 1024 * 1024)
            chunks, chunk_memories = self._split_by_memory(
                all_nodes, type_map, init_map, budget_bytes
            )
            strategy = "memory_budget"

        total_chunks = len(chunks)
        zfill = len(str(max(total_chunks - 1, 0)))

        profile = {
            "strategy": strategy,
            "memory_budget_mb": memory_budget_mb if nodes_per_subgraph is None else None,
            "nodes_per_subgraph": nodes_per_subgraph,
            "total_chunks": total_chunks,
            "subgraphs": [],
        }

        for sub_idx, chunk_nodes in enumerate(chunks):
            sub_info = self._build_one(
                sub_idx, chunk_nodes, all_nodes,
                type_map, init_map, selected_names,
                onnx_model, save_dir, zfill,
            )
            if chunk_memories:
                sub_info["memory"] = chunk_memories[sub_idx]
            profile["subgraphs"].append(sub_info)

        # ---- save memory profile ----
        profile_path = os.path.join(save_dir, "dump_profile.json")
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    # ── partitioning strategies ────────────────────────────────────────

    def _split_by_node_count(self, all_nodes, nodes_per_subgraph):
        chunks = []
        for i in range(0, len(all_nodes), nodes_per_subgraph):
            chunks.append(all_nodes[i:i + nodes_per_subgraph])
        return chunks, []

    def _split_by_memory(self, all_nodes, type_map, init_map, budget_bytes):
        """Greedy topological partition: accumulate nodes until the peak memory
        (initializers + activations) exceeds *budget_bytes*, then cut."""
        chunks = []
        chunk_memories = []

        current_chunk = []
        current_produced = set()   # tensor names produced in this chunk
        current_init_names = set() # initializer names referenced in this chunk

        init_mem = 0  # bytes of unique initializers in chunk
        prod_mem = 0  # bytes of produced activations in chunk

        def _snapshot_memory():
            return {
                "init_memory_mb": round(init_mem / (1024 * 1024), 3),
                "activation_memory_mb": round(prod_mem / (1024 * 1024), 3),
                "total_memory_mb": round((init_mem + prod_mem) / (1024 * 1024), 3),
                "node_count": len(current_chunk),
            }

        def _finalize_chunk():
            nonlocal current_chunk, current_produced, current_init_names
            nonlocal init_mem, prod_mem
            chunks.append(current_chunk)
            chunk_memories.append(_snapshot_memory())
            current_chunk = []
            current_produced = set()
            current_init_names = set()
            init_mem = 0
            prod_mem = 0

        def _tensor_bytes_in(name):
            if name in type_map:
                return MemoryEstimator.from_value_info(type_map[name])
            return 0

        def _init_bytes(name):
            if name in init_map:
                return MemoryEstimator.from_tensor_proto(init_map[name])
            return 0

        for node in all_nodes:
            # --- memory that would be added by this node ---
            add_init = sum(
                _init_bytes(inp) for inp in node.input
                if inp not in current_init_names
            )
            add_prod = sum(
                _tensor_bytes_in(out) for out in node.output
                if out not in current_produced
            )

            would_be_total = init_mem + add_init + prod_mem + add_prod

            # cut if budget exceeded AND we already have nodes in the chunk
            if would_be_total > budget_bytes and current_chunk:
                _finalize_chunk()

            # --- commit node to current chunk ---
            current_chunk.append(node)

            for out in node.output:
                if out not in current_produced:
                    current_produced.add(out)
                    prod_mem += _tensor_bytes_in(out)

            for inp in node.input:
                if inp in init_map and inp not in current_init_names:
                    current_init_names.add(inp)
                    init_mem += _init_bytes(inp)

        if current_chunk:
            chunks.append(current_chunk)
            chunk_memories.append(_snapshot_memory())

        return chunks, chunk_memories

    # ── subgraph construction ──────────────────────────────────────────

    def _build_one(self, sub_idx, chunk_nodes, all_nodes,
                   type_map, init_map, selected_names,
                   onnx_model, save_dir, zfill):

        produced = set()
        chunk_output_names = []
        for n in chunk_nodes:
            for out_name in n.output:
                produced.add(out_name)
                if selected_names is None or out_name in selected_names:
                    chunk_output_names.append(out_name)

        consumed = set()
        for n in chunk_nodes:
            for inp_name in n.input:
                consumed.add(inp_name)

        external_input_names = consumed - produced

        sub_inputs = [type_map[name] for name in external_input_names if name in type_map]
        sub_outputs = [type_map[name] for name in chunk_output_names if name in type_map]

        needed_init_names = set()
        for n in chunk_nodes:
            for inp in n.input:
                if inp in init_map:
                    needed_init_names.add(inp)
        sub_inits = [init_map[name] for name in needed_init_names]

        sub_graph = helper.make_graph(
            list(chunk_nodes),
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

        return {
            "index": sub_idx,
            "node_count": len(chunk_nodes),
            "input_count": len(sub_inputs),
            "output_count": len(sub_outputs),
            "init_count": len(sub_inits),
            "file": os.path.basename(save_path),
        }
