import json
import os
import re
from functools import lru_cache
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

    _SAFE_RE_ILLEGAL = re.compile(r"[<>:\"/\\|?*\s]")
    _SAFE_RE_COLLAPSE = re.compile(r"_+")

    def __init__(self, onnx_path, trt_layer_path, onnx_dump_dir, trt_dump_dir,
                 trt_mapping=None):
        self._mapper = LayerMapper(onnx_path, trt_layer_path)

        # load ONNX name mapping
        with open(os.path.join(onnx_dump_dir, "name_mapping.json")) as f:
            self._onnx_map = json.load(f)

        # TRT mapping: accept pre-built dict to skip disk round-trip
        if trt_mapping is not None:
            self._trt_map = trt_mapping
        else:
            with open(os.path.join(trt_dump_dir, "name_mapping.json")) as f:
                self._trt_map = json.load(f)

        self._onnx_dir = onnx_dump_dir
        self._trt_dir = trt_dump_dir
        self._npy_cache = {}  # path → array, avoids re-loading same file

        # pre-build safe-name indexes for O(1) lookup (avoids O(n) scans)
        self._onnx_safe_index = self._build_safe_index(self._onnx_map)
        self._trt_safe_index = self._build_safe_index(self._trt_map)

    @staticmethod
    def _build_safe_index(name_map):
        """{safe_name: original_name} — first-wins for O(1) safe-name resolution."""
        idx = {}
        for name in name_map:
            safe = LayerComparator._safe_name(name)
            idx.setdefault(safe, name)
        return idx

    # ── name resolution helpers ──

    @staticmethod
    @lru_cache(maxsize=4096)
    def _safe_name(tensor_name):
        s = LayerComparator._SAFE_RE_ILLEGAL.sub("_", tensor_name)
        return LayerComparator._SAFE_RE_COLLAPSE.sub("_", s)

    def _find_trt_dump(self, onnx_output_name, all_onnx_outputs):
        """Resolve TRT dump file for the ONNX tensor(s) produced by this layer.

        Tries: exact match → safe-name index (O(1)) → substring fallback.
        Returns (filename, strategy, matched_key) or (None, reason, "").
        """
        candidates = [onnx_output_name] + [o for o in all_onnx_outputs if o != onnx_output_name]

        # 1) exact match with any candidate
        for c in candidates:
            if c in self._trt_map:
                return self._trt_map[c], "exact", c

        # 2) safe-name match via pre-built index — O(1)
        target_safe = self._safe_name(onnx_output_name)
        matched = self._trt_safe_index.get(target_safe)
        if matched:
            return self._trt_map[matched], "safe_name", matched
        for c in candidates[1:]:
            matched = self._trt_safe_index.get(self._safe_name(c))
            if matched:
                return self._trt_map[matched], "safe_name", matched

        # 3) substring — last resort (O(n), but rarely reached)
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
        """Resolve ONNX dump file. Tries: exact match → all node outputs → safe-name index."""
        # 1) exact match
        if onnx_output_name in self._onnx_map:
            return self._onnx_map[onnx_output_name], onnx_output_name

        # 2) try all outputs of matched nodes
        for o in all_onnx_outputs:
            if o in self._onnx_map:
                return self._onnx_map[o], o

        # 3) safe-name via pre-built index — O(1)
        target_safe = self._safe_name(onnx_output_name)
        matched = self._onnx_safe_index.get(target_safe)
        if matched:
            return self._onnx_map[matched], matched

        return None, ""

    # ── compare ──

    def compare(self):
        rows = []
        matched = 0
        shape_mismatch = 0
        missing_trt = 0
        missing_onnx = 0
        dtype_conflict = 0
        skipped = 0

        for layer in self._mapper.layers:
            name = layer["Name"]
            onnx_output_name = self._mapper.trt_outputs.get(name, "")
            onnx_nodes = self._mapper.trt_to_onnx_map.get(name, [])

            # collect ALL outputs from matched ONNX nodes (for fallback lookup)
            all_onnx_outputs = []
            for n in onnx_nodes:
                all_onnx_outputs.extend(n.output)

            row = {
                "trt_layer": name,
                "trt_type": layer["LayerType"],
                "trt_output": onnx_output_name,
                "onnx_nodes": ", ".join(o for n in onnx_nodes for o in n.output),
                "onnx_ops": ", ".join(n.op_type for n in onnx_nodes),
                "onnx_node_count": len(onnx_nodes),
                "trt_parameter_type": layer.get("ParameterType", ""),
                "trt_origin": layer.get("Origin", ""),
                "trt_tactic": layer.get("TacticValue", ""),
                "trt_inputs": ", ".join(inp.get("Name", "") for inp in layer.get("Inputs", [])),
                "trt_input_dims": ", ".join(
                    "x".join(str(d) for d in inp.get("Dimensions", []))
                    for inp in layer.get("Inputs", [])),
                "trt_output_dims": ", ".join(
                    "x".join(str(d) for d in out.get("Dimensions", []))
                    for out in layer.get("Outputs", [])),
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

            # load and compare (cached — same tensor may be referenced by multiple layers)
            trt_path = os.path.join(self._trt_dir, trt_file)
            onnx_path_full = os.path.join(self._onnx_dir, onnx_file)
            trt_data = self._npy_cache.get(trt_path)
            if trt_data is None:
                trt_data = np.load(trt_path)
                self._npy_cache[trt_path] = trt_data
            onnx_data = self._npy_cache.get(onnx_path_full)
            if onnx_data is None:
                onnx_data = np.load(onnx_path_full)
                self._npy_cache[onnx_path_full] = onnx_data

            trt_is_int = np.issubdtype(trt_data.dtype, np.integer)
            onnx_is_int = np.issubdtype(onnx_data.dtype, np.integer)

            # detect implausible pairings: integer vs float → likely name mismatch
            if trt_is_int != onnx_is_int:
                dtype_conflict += 1
                row["status"] = "dtype_conflict"
                row["trt_dtype"] = str(trt_data.dtype)
                row["onnx_dtype"] = str(onnx_data.dtype)
                row["trt_shape"] = str(trt_data.shape)
                row["onnx_shape"] = str(onnx_data.shape)
                rows.append(row)
                continue

            if trt_data.shape != onnx_data.shape:
                shape_mismatch += 1
                row.update({
                    "status": "shape_mismatch",
                    "trt_shape": str(trt_data.shape),
                    "onnx_shape": str(onnx_data.shape),
                    "trt_dtype": str(trt_data.dtype),
                    "onnx_dtype": str(onnx_data.dtype),
                    "trt_elements": int(trt_data.size),
                    "onnx_elements": int(onnx_data.size),
                    "onnx_file": onnx_file,
                    "trt_file": trt_file,
                })
                rows.append(row)
                continue

            # integer tensors: exact comparison
            if trt_is_int:
                trt_flat = trt_data.ravel()
                onnx_flat = onnx_data.ravel()
                exact_match = bool(np.array_equal(trt_data, onnx_data))
                mismatch_mask = trt_flat != onnx_flat
                mismatch_count = int(np.sum(mismatch_mask))
                max_ad = float(np.abs(trt_flat.astype(np.int64) -
                                       onnx_flat.astype(np.int64)).max())

                # locate first mismatch position
                mismatch_idx = np.where(mismatch_mask)[0]
                first_mismatch_pos = (list(np.unravel_index(int(mismatch_idx[0]), trt_data.shape))
                                      if len(mismatch_idx) > 0 and trt_data.ndim > 1
                                      else [int(mismatch_idx[0])] if len(mismatch_idx) > 0
                                      else [-1])
                first_mismatch_trt = (int(trt_flat[mismatch_idx[0]])
                                     if len(mismatch_idx) > 0 else 0)
                first_mismatch_onnx = (int(onnx_flat[mismatch_idx[0]])
                                      if len(mismatch_idx) > 0 else 0)

                matched += 1
                row.update({
                    "status": "ok",
                    "max_abs_diff": max_ad,
                    "mean_abs_diff": float(mismatch_count) / max(trt_data.size, 1),
                    "relative_diff": float(mismatch_count) / max(trt_data.size, 1),
                    "cosine_sim": 1.0 if exact_match else 0.0,
                    "allclose_1e-3": exact_match,
                    "allclose_1e-5": exact_match,
                    "shape": str(trt_data.shape),
                    "trt_dtype": str(trt_data.dtype),
                    "onnx_dtype": str(onnx_data.dtype),
                    "num_elements": int(trt_data.size),
                    "trt_min": int(trt_flat.min()),
                    "trt_max": int(trt_flat.max()),
                    "trt_mean": float(trt_flat.mean()),
                    "onnx_min": int(onnx_flat.min()),
                    "onnx_max": int(onnx_flat.max()),
                    "onnx_mean": float(onnx_flat.mean()),
                    "mismatch_count": mismatch_count,
                    "first_mismatch_position": first_mismatch_pos,
                    "first_mismatch_trt_value": first_mismatch_trt,
                    "first_mismatch_onnx_value": first_mismatch_onnx,
                    "onnx_file": onnx_file,
                    "trt_file": trt_file,
                })
                rows.append(row)
                continue

            # float tensors: tolerance-based comparison
            diff = trt_data.astype(np.float64) - onnx_data.astype(np.float64)
            abs_diff = np.abs(diff)

            trt_flat = trt_data.ravel().astype(np.float64)
            onnx_flat = onnx_data.ravel().astype(np.float64)

            max_ad = float(abs_diff.max())
            mean_ad = float(abs_diff.mean())

            # locate the element with the largest error
            max_flat_idx = int(np.argmax(abs_diff))
            max_pos = (list(np.unravel_index(max_flat_idx, trt_data.shape))
                       if trt_data.ndim > 1 else [max_flat_idx])
            max_trt_val = float(trt_flat[max_flat_idx])
            max_onnx_val = float(onnx_flat[max_flat_idx])

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
                "trt_dtype": str(trt_data.dtype),
                "onnx_dtype": str(onnx_data.dtype),
                "num_elements": int(trt_data.size),
                "trt_min": float(trt_flat.min()),
                "trt_max": float(trt_flat.max()),
                "trt_mean": float(trt_flat.mean()),
                "onnx_min": float(onnx_flat.min()),
                "onnx_max": float(onnx_flat.max()),
                "onnx_mean": float(onnx_flat.mean()),
                "max_error_position": max_pos,
                "max_error_trt_value": max_trt_val,
                "max_error_onnx_value": max_onnx_val,
                "onnx_file": onnx_file,
                "trt_file": trt_file,
            })
            rows.append(row)

        ok_rows = [r for r in rows if r.get("status") == "ok"]
        self._result = CompareResult(
            rows=rows,
            summary={
                "total_layers": len(self._mapper.layers),
                "compared": matched,
                "skipped_unmapped": skipped,
                "dtype_conflict": dtype_conflict,
                "shape_mismatch": shape_mismatch,
                "missing_trt_dump": missing_trt,
                "missing_onnx_dump": missing_onnx,
                "allclose_1e-3": sum(1 for r in rows if r.get("allclose_1e-3", False)),
                "allclose_1e-5": sum(1 for r in rows if r.get("allclose_1e-5", False)),
                "max_abs_diff": max((r.get("max_abs_diff", 0) for r in ok_rows), default=0),
                "mean_abs_diff": sum(r.get("mean_abs_diff", 0) for r in ok_rows) / max(len(ok_rows), 1),
                "worst_layer": max(ok_rows, key=lambda r: r.get("max_abs_diff", 0)).get("trt_layer", "") if ok_rows else "",
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
        if path.endswith(".csv"):
            import csv
            if not rows:
                return path
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        else:
            with open(path, "w") as f:
                json.dump({"summary": self._result.summary, "rows": rows}, f, indent=2, ensure_ascii=False)
        return path
