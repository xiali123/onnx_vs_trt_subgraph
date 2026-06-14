import json
import os
import numpy as np
from dataclasses import dataclass, field
from onnx_trt_map import LayerMapper


_EPSILON = 1e-8


@dataclass
class CompareResult:
    rows: list = field(default_factory=list)      # per-layer comparison rows
    summary: dict = field(default_factory=dict)    # overall stats

    def top_errors(self, n=10, key="max_abs_diff"):
        return sorted(self.rows, key=lambda r: r.get(key, 0), reverse=True)[:n]


class LayerComparator:
    def __init__(self, onnx_path, trt_layer_path, onnx_dump_dir, trt_dump_dir):
        self._mapper = LayerMapper(onnx_path, trt_layer_path)

        # load name mappings
        with open(os.path.join(onnx_dump_dir, "name_mapping.json")) as f:
            self._onnx_map = json.load(f)   # tensor_name → npy_filename
        with open(os.path.join(trt_dump_dir, "name_mapping.json")) as f:
            self._trt_map = json.load(f)

        self._onnx_dir = onnx_dump_dir
        self._trt_dir = trt_dump_dir

    # ── compare ──

    def compare(self):
        rows = []
        matched = 0
        shape_mismatch = 0
        missing_trt = 0
        missing_onnx = 0

        for layer in self._mapper._layers:
            name = layer["Name"]
            trt_output = self._mapper._trt_output.get(name, "")
            onnx_nodes = self._mapper._trt_to_onnx.get(name, [])

            if not trt_output or not onnx_nodes:
                continue

            # find ONNX output tensor (most downstream node's output)
            onnx_output = onnx_nodes[-1].output[0]

            # resolve file names
            trt_file = self._trt_map.get(trt_output)
            onnx_file = self._onnx_map.get(onnx_output)

            row = {
                "trt_layer": name,
                "trt_type": layer["LayerType"],
                "trt_output": trt_output,
                "onnx_nodes": ", ".join(n.output[0] for n in onnx_nodes),
                "onnx_ops": ", ".join(n.op_type for n in onnx_nodes),
            }

            if not trt_file:
                missing_trt += 1
                row["status"] = "missing_trt_dump"
                rows.append(row)
                continue
            if not onnx_file:
                missing_onnx += 1
                row["status"] = "missing_onnx_dump"
                rows.append(row)
                continue

            # load and compare
            trt_data = np.load(os.path.join(self._trt_dir, trt_file))
            onnx_data = np.load(os.path.join(self._onnx_dir, onnx_file))

            if trt_data.shape != onnx_data.shape:
                shape_mismatch += 1
                row["status"] = "shape_mismatch"
                row["trt_shape"] = str(trt_data.shape)
                row["onnx_shape"] = str(onnx_data.shape)
                rows.append(row)
                continue

            diff = trt_data.astype(np.float64) - onnx_data.astype(np.float64)
            abs_diff = np.abs(diff)

            trt_flat = trt_data.ravel().astype(np.float64)
            onnx_flat = onnx_data.ravel().astype(np.float64)

            max_ad = float(abs_diff.max())
            mean_ad = float(abs_diff.mean())

            # relative diff
            onnx_max = float(np.abs(onnx_data).max())
            rel_diff = max_ad / max(onnx_max, _EPSILON)

            # cosine similarity
            denom = np.linalg.norm(trt_flat) * np.linalg.norm(onnx_flat)
            cos_sim = float(np.dot(trt_flat, onnx_flat) / max(denom, _EPSILON))

            matched += 1
            row.update({
                "status": "ok",
                "max_abs_diff": max_ad,
                "mean_abs_diff": mean_ad,
                "relative_diff": rel_diff,
                "cosine_sim": cos_sim,
                "allclose_1e-3": bool(np.allclose(trt_data, onnx_data, atol=1e-3)),
                "allclose_1e-5": bool(np.allclose(trt_data, onnx_data, atol=1e-5)),
                "shape": str(trt_data.shape),
            })
            rows.append(row)

        self._result = CompareResult(
            rows=rows,
            summary={
                "total_layers": len(self._mapper._layers),
                "compared": matched,
                "shape_mismatch": shape_mismatch,
                "missing_trt_dump": missing_trt,
                "missing_onnx_dump": missing_onnx,
                "allclose_1e-3": sum(1 for r in rows if r.get("allclose_1e-3", False)),
                "allclose_1e-5": sum(1 for r in rows if r.get("allclose_1e-5", False)),
            },
        )
        return self._result

    @property
    def result(self):
        return getattr(self, "_result", None)

    def save_report(self, path):
        if self._result is None:
            raise RuntimeError("Call compare() first")
        rows = self._result.rows
        if not rows:
            return
        if path.endswith(".csv"):
            import csv
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        else:
            with open(path, "w") as f:
                json.dump({"summary": self._result.summary, "rows": rows}, f, indent=2, ensure_ascii=False)
        return path
