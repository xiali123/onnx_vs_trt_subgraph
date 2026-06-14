import os
import json
import onnx
from onnx import helper
from onnx.shape_inference import infer_shapes
from onnx_trt_map import LayerMapper


class TRTPartitioner:
    def __init__(self, onnx_path, trt_layer_path):
        self._onnx_model = onnx.load(onnx_path)
        self._onnx_model = infer_shapes(self._onnx_model)
        onnx.checker.check_model(self._onnx_model)
        self._graph = self._onnx_model.graph

        self._mapper = LayerMapper(onnx_path, trt_layer_path)

        # ordered TRT layers that have ONNX mappings (skip NoOp/Reformat)
        self._ordered_layers = []
        for layer in self._mapper.layers:
            name = layer["Name"]
            if name in self._mapper.trt_to_onnx_map:
                self._ordered_layers.append(layer)

        # type info cache
        self._type_map = {}
        for vi in list(self._graph.input) + list(self._graph.output) + list(self._graph.value_info):
            self._type_map[vi.name] = vi
        self._init_map = {init.name: init for init in self._graph.initializer}

    # ── split ──

    def split(self, save_dir, nodes_per_subgraph=500):
        """Partition ONNX into subgraphs of roughly nodes_per_subgraph."""
        groups = self._partition_groups(nodes_per_subgraph)
        return self._build_subgraphs(groups, save_dir)

    def split_by_count(self, save_dir, num_subgraphs=10):
        total = sum(len(self._mapper.trt_to_onnx_map[l["Name"]]) for l in self._ordered_layers)
        target = total // num_subgraphs
        return self.split(save_dir, nodes_per_subgraph=target)

    # ── partition logic ──

    def _partition_groups(self, target):
        """Greedy partition: fill each group up to target, never break a TRT layer."""
        groups = []
        current_group = []
        current_weight = 0

        for layer in self._ordered_layers:
            weight = len(self._mapper.trt_to_onnx_map[layer["Name"]])

            if current_weight >= target and current_group:
                groups.append(current_group)
                current_group = []
                current_weight = 0

            current_group.append(layer)
            current_weight += weight

        if current_group:
            groups.append(current_group)

        return groups

    # ── subgraph builder ──

    def _build_subgraphs(self, groups, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        rows = []
        zfill = len(str(len(groups)))

        for gid, group in enumerate(groups):
            # collect all ONNX nodes for this group
            onnx_node_names = set()
            for layer in group:
                for n in self._mapper.trt_to_onnx_map[layer["Name"]]:
                    onnx_node_names.update(n.output)

            # collect ONNX proto nodes + their inputs/outputs
            chunk_nodes = []
            produced = set()
            consumed = set()
            for n in self._graph.node:
                if any(o in onnx_node_names for o in n.output):
                    chunk_nodes.append(n)
                    for o in n.output:
                        produced.add(o)
                    for inp in n.input:
                        consumed.add(inp)

            # pull in Constant nodes whose outputs feed chunk_nodes
            for n in self._graph.node:
                if n.op_type == "Constant" and any(o in consumed for o in n.output):
                    chunk_nodes.append(n)
                    for o in n.output:
                        produced.add(o)
                    for inp in n.input:
                        consumed.add(inp)

            # sort by original graph order (preserves topological sort)
            node_order = {}
            for i, n in enumerate(self._graph.node):
                # use output[0] as fallback for unnamed nodes (ONNX allows empty names)
                key = n.name or (n.output[0] if n.output else "")
                node_order[key] = i
            chunk_nodes.sort(key=lambda n: node_order.get(
                n.name or (n.output[0] if n.output else ""), 0))

            external_inputs = consumed - produced

            # build inputs
            sub_inputs = []
            for name in external_inputs:
                if name in self._type_map:
                    sub_inputs.append(self._type_map[name])

            # build outputs
            sub_outputs = []
            for name in produced:
                if name in self._type_map:
                    sub_outputs.append(self._type_map[name])

            # initializers
            needed_init = set()
            for n in chunk_nodes:
                for inp in n.input:
                    if inp in self._init_map:
                        needed_init.add(inp)
            sub_inits = [self._init_map[n] for n in needed_init]

            # build subgraph
            sub_graph = helper.make_graph(
                chunk_nodes,
                f"subgraph_{gid}",
                sub_inputs,
                sub_outputs,
                sub_inits,
            )
            sub_model = helper.make_model(
                sub_graph,
                opset_imports=self._onnx_model.opset_import,
            )

            fname = f"subgraph_{str(gid).zfill(zfill)}.onnx"
            onnx.save(sub_model, os.path.join(save_dir, fname))

            # rows for CSV
            for layer in group:
                name = layer["Name"]
                onnx_nodes = self._mapper.trt_to_onnx(name)
                rows.append({
                    "trt_name": name,
                    "trt_type": layer["LayerType"],
                    "subgraph": gid,
                    "subgraph_file": fname,
                    "onnx_count": len(onnx_nodes),
                    "onnx_nodes": ", ".join(o for n in onnx_nodes for o in n.output),
                    "trt_output": self._mapper.trt_outputs.get(name, ""),
                })

            print(f"[{gid}] {fname}: {len(chunk_nodes)} onnx nodes, {len(group)} TRT layers")

        # save partition CSV
        if rows:
            csv_path = os.path.join(save_dir, "partition.csv")
            import csv
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\nPartition table: {csv_path}")

        return save_dir
