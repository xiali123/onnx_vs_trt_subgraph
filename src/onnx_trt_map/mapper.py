import json
import os
import re
import onnx
from dataclasses import dataclass, field


_META_RE = re.compile(r"\[ONNX Layer:\s*(.+?)\]")


@dataclass
class MatchResult:
    matched: list = field(default_factory=list)         # [(trt_layer_dict, [npy_filename, ...]), ...]
    unmatched_layers: list = field(default_factory=list) # trt_layer_dict (has output tensor but no npy)
    orphan_npy: list = field(default_factory=list)       # npy filenames with no matching layer

# ONNX ops that TRT may absorb/fold into adjacent layers
_PASS_THROUGH_OPS = {"Flatten", "Reshape", "Squeeze", "Unsqueeze", "Identity", "Transpose"}


def _parse_metadata(metadata_str):
    if not metadata_str:
        return set()
    return set(_META_RE.findall(metadata_str))


class LayerMapper:
    def __init__(self, onnx_path, trt_layer_path):
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        self._nodes = list(onnx_model.graph.node)

        # tensor_name -> (node_index, node_proto)
        self._tensor_to_node = {}
        # node_name -> (node_index, node_proto)
        self._node_name_to_node = {}
        # tensor_name -> set of consumer node indices
        self._tensor_to_consumers = {}
        for i, n in enumerate(self._nodes):
            self._node_name_to_node[n.name] = (i, n)
            for out in n.output:
                self._tensor_to_node[out] = (i, n)
            for inp in n.input:
                self._tensor_to_consumers.setdefault(inp, set()).add(i)

        with open(trt_layer_path) as f:
            trt_data = json.load(f)
        self._layers = trt_data["Layers"]
        self._bindings = trt_data.get("Bindings", [])

        self._trt_to_onnx = {}   # trt_layer_name -> [onnx_node_proto]
        self._trt_output = {}    # trt_layer_name -> downstream_tensor_name
        self._onnx_to_trt = {}   # onnx_tensor_name -> trt_layer dict
        self._build_mapping()
        self._absorb_pass_through()

    # ── mapping ──

    def _build_mapping(self):
        for layer in self._layers:
            meta_names = _parse_metadata(layer.get("Metadata", ""))
            if not meta_names:
                continue

            matched = []
            for mname in meta_names:
                tname = mname.replace("node_of_", "", 1) if mname.startswith("node_of_") else mname
                node = self._tensor_to_node.get(tname)
                if node is None:
                    node = self._node_name_to_node.get(tname)
                if node:
                    matched.append(node)

            if not matched:
                continue

            # find most downstream node
            downstream = self._find_downstream(matched)

            name = layer["Name"]
            onnx_nodes = [n for _, n in matched]
            self._trt_to_onnx[name] = onnx_nodes
            if downstream:
                self._trt_output[name] = downstream.output[0]

            for _, n in matched:
                for out in n.output:
                    self._onnx_to_trt[out] = layer

    def _find_downstream(self, matched):
        """Return the (idx, node) whose outputs are NOT consumed by any other in matched."""
        best_idx = -1
        best_node = None
        for idx, node in matched:
            downstream = True
            for other_idx, other_node in matched:
                if other_idx == idx:
                    continue
                for out_name in node.output:
                    if out_name in other_node.input:
                        downstream = False
                        break
                if not downstream:
                    break
            if downstream and idx > best_idx:
                best_idx = idx
                best_node = node
        if best_node is None:
            best_idx = max(i for i, _ in matched)
            best_node = self._nodes[best_idx]
        return best_node

    def _absorb_pass_through(self):
        """Map pass-through ONNX ops (Flatten etc.) to the TRT layer of their consumer."""
        for i, node in enumerate(self._nodes):
            if node.op_type not in _PASS_THROUGH_OPS:
                continue
            already_mapped = any(o in self._onnx_to_trt for o in node.output)
            if already_mapped:
                continue
            # find which TRT layer handles the consumer
            for out_name in node.output:
                consumer_idxs = self._tensor_to_consumers.get(out_name, set())
                for cidx in consumer_idxs:
                    cnode = self._nodes[cidx]
                    for cout in cnode.output:
                        trt_layer = self._onnx_to_trt.get(cout)
                        if trt_layer:
                            # absorb this pass-through into the consumer's TRT layer
                            self._trt_to_onnx.setdefault(trt_layer["Name"], []).append(node)
                            for o in node.output:
                                self._onnx_to_trt[o] = trt_layer
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

    # ── query ──

    def trt_to_onnx(self, trt_layer_name):
        return self._trt_to_onnx.get(trt_layer_name, [])

    def onnx_to_trt(self, tensor_name):
        return self._onnx_to_trt.get(tensor_name)

    @property
    def mapping(self):
        return dict(self._trt_to_onnx)

    # ── stats ──

    def stats(self):
        total_trt = len(self._layers)
        mapped_trt = len(self._trt_to_onnx)
        total_onnx = len(self._nodes)
        mapped_onnx = len(self._onnx_to_trt)
        return {
            "trt_layers": total_trt,
            "trt_mapped": mapped_trt,
            "trt_unmapped": total_trt - mapped_trt,
            "onnx_nodes": total_onnx,
            "onnx_mapped": mapped_onnx,
            "onnx_unmapped": total_onnx - mapped_onnx,
        }

    # ── export ──

    def to_rows(self):
        rows = []
        for layer in self._layers:
            name = layer["Name"]
            onnx_nodes = self._trt_to_onnx.get(name, [])
            rows.append({
                "trt_name": name,
                "trt_type": layer["LayerType"],
                "onnx_count": len(onnx_nodes),
                "onnx_nodes": ", ".join(n.output[0] for n in onnx_nodes),
                "onnx_ops": ", ".join(n.op_type for n in onnx_nodes),
                "trt_output": self._trt_output.get(name, ""),
            })
        return rows

    def save_mapping(self, path):
        rows = self.to_rows()
        if path.endswith(".csv"):
            import csv
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        else:
            with open(path, "w") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
        return path

    # ── debug tensor file matching ──

    def match_dump_files(self, dump_dir):
        """Match .npy dump files to TRT layers by checking if the filename
        contains the TRT layer's output tensor name."""
        if not os.path.isdir(dump_dir):
            raise FileNotFoundError(f"dump dir not found: {dump_dir}")

        npy_files = sorted(
            f for f in os.listdir(dump_dir)
            if f.endswith(".npy") and f != "name_mapping.json"
        )

        # layer → files (each file keyed by output tensor name)
        layer_files = {}  # trt_layer_name -> [npy_filename]
        unmatched_npy = set(npy_files)

        for layer in self._layers:
            name = layer["Name"]
            output = self._trt_output.get(name, "")
            if not output:
                continue
            matches = [f for f in npy_files if output in f]
            if matches:
                layer_files[name] = matches
                unmatched_npy -= set(matches)

        matched = []
        unmatched_layers = []
        for layer in self._layers:
            name = layer["Name"]
            if name in layer_files:
                matched.append((layer, layer_files[name]))
            elif self._trt_output.get(name):
                # has output tensor but no matching npy
                unmatched_layers.append(layer)

        return MatchResult(
            matched=matched,
            unmatched_layers=unmatched_layers,
            orphan_npy=sorted(unmatched_npy),
        )

    def save_match_result(self, dump_dir, path):
        result = self.match_dump_files(dump_dir)

        rows = []
        for layer, npy_files in result.matched:
            rows.append({
                "trt_name": layer["Name"],
                "trt_type": layer["LayerType"],
                "trt_output": self._trt_output.get(layer["Name"], ""),
                "npy_files": ", ".join(npy_files),
                "npy_count": len(npy_files),
            })

        for layer in result.unmatched_layers:
            rows.append({
                "trt_name": layer["Name"],
                "trt_type": layer["LayerType"],
                "trt_output": self._trt_output.get(layer["Name"], ""),
                "npy_files": "",
                "npy_count": 0,
            })

        for npy in result.orphan_npy:
            rows.append({
                "trt_name": "(orphan)",
                "trt_type": "",
                "trt_output": "",
                "npy_files": npy,
                "npy_count": 1,
            })

        if path.endswith(".csv"):
            import csv
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        else:
            with open(path, "w") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
        return path
