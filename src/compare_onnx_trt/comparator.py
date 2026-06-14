import json
import os
import re
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

    # ── name resolution helpers ──

    @staticmethod
    def _safe_name(tensor_name):
        s = re.sub(r"[<>:\"/\\|?*\s]", "_", tensor_name)
        return re.sub(r"_+", "_", s)

    def _find_trt_dump(self, onnx_output_name, all_onnx_outputs):
        """Resolve TRT dump file for the ONNX tensor(s) produced by this layer.

        Tries: exact onnx_output_name → safe-name → all ONNX outputs →
        substring (preferring shortest match). Returns (filename, strategy, matched_key)
        or (None, reason, "").
        """
        candidates = [onnx_output_name] + [o for o in all_onnx_outputs if o != onnx_output_name]

        # 1) exact match with any candidate
        for c in candidates:
            if c in self._trt_map:
                return self._trt_map[c], "exact", c

        # 2) safe-name match
        target_safe = self._safe_name(onnx_output_name)
        for mapped_name, filename in self._trt_map.items():
            mapped_safe = self._safe_name(mapped_name)
            if target_safe == mapped_safe:
                return filename, "safe_name", mapped_name
            # also try safe names of all candidates
            for c in candidates[1:]:
                if self._safe_name(c) == mapped_safe:
                    return filename, "safe_name", mapped_name

        # 3) substring — prefer shortest match (less likely to be a false positive)
        best = None
        for mapped_name, filename in self._trt_map.items():
            mapped_safe = self._safe_name(mapped_name)
            if target_safe in mapped_safe or mapped_safe in target_safe:
                if best is None or len(mapped_safe) < len(best[1]):
                    best = (filename, mapped_safe, mapped_name)
        if best:
            return best[0], "substring", best[2]

        return None, f"no_trt_dump (exact={onnx_output_name}, safe={target_safe})", ""

    def _find_onnx_dump(self, onnx_output_name, all_onnx_outputs):
        """Resolve ONNX dump file. Tries: exact onnx_output_name → all node outputs → safe-name."""
        # 1) exact match
        if onnx_output_name in self._onnx_map:
            return self._onnx_map[onnx_output_name], onnx_output_name

        # 2) try all outputs of matched nodes
        for o in all_onnx_outputs:
            if o in self._onnx_map:
                return self._onnx_map[o], o

        # 3) safe-name fallback
        target_safe = self._safe_name(onnx_output_name)
        for mapped_name, filename in self._onnx_map.items():
            if self._safe_name(mapped_name) == target_safe:
                return filename, mapped_name

        return None, ""

    # ── compare ──

    def compare(self):
        rows = []
        matched = 0
        shape_mismatch = 0
        missing_trt = 0
        missing_onnx = 0
        skipped = 0

        for layer in self._mapper._layers:
            name = layer["Name"]
            onnx_output_name = self._mapper._trt_output.get(name, "")
            onnx_nodes = self._mapper._trt_to_onnx.get(name, [])

            # collect ALL outputs from matched ONNX nodes (for fallback lookup)
            all_onnx_outputs = []
            for n in onnx_nodes:
                all_onnx_outputs.extend(n.output)

            row = {
                "trt_layer": name,
                "trt_type": layer["LayerType"],
                "trt_output": onnx_output_name,
                "onnx_nodes": ", ".join(n.output[0] for n in onnx_nodes),
                "onnx_ops": ", ".join(n.op_type for n in onnx_nodes),
            }

            if not onnx_output_name or not onnx_nodes:
                skipped += 1
                row["status"] = "unmapped"
                rows.append(row)
                continue

            trt_file, strategy, trt_matched_key = self._find_trt_dump(
                onnx_output_name, all_onnx_outputs)
            onnx_file, onnx_matched_key = self._find_onnx_dump(
                onnx_output_name, all_onnx_outputs)

            row["trt_match_strategy"] = strategy
            row["trt_matched_key"] = trt_matched_key
            row["onnx_matched_key"] = onnx_matched_key

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

            onnx_max = float(np.abs(onnx_data).max())
            rel_diff = max_ad / max(onnx_max, _EPSILON)

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
                "skipped_unmapped": skipped,
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
