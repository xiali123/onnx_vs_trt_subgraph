import os
import re
import json
import glob
import logging
import shutil
from collections import deque
import numpy as np
import onnx
from onnx import TensorProto
from dataclasses import dataclass, field

from onnx_probe import GraphModel, AllTensorSelector, DumpBuilder, ORTRunner
from trt_op import TRTBuilder
from subgraph_split_by_trt import TRTPartitioner
from compare_onnx_trt import LayerComparator

logger = logging.getLogger("verify_pipeline")


_ONNX_DTYPE_MAP = {
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.BOOL: bool,
}

# dtypes TRT does not support as network I/O → safe fallback
_TRT_COMPAT_DTYPE = {
    np.float64: np.float32,
}

def _parse_onnx_inputs(onnx_path):
    """Return {input_name: (shape, numpy_dtype)} for runtime graph inputs
    (excludes initializers which carry their own data)."""
    model = onnx.load(onnx_path)
    init_names = {init.name for init in model.graph.initializer}
    specs = {}
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        ts = inp.type.tensor_type
        shape = [d.dim_value if d.dim_value else 1 for d in ts.shape.dim]
        dtype = _ONNX_DTYPE_MAP.get(ts.elem_type, np.float32)
        specs[inp.name] = (tuple(shape), dtype)
    return specs


def _find_constant_value(model_path, tensor_name):
    """If tensor_name is the output of a Constant node, return its numpy value."""
    model = onnx.load(model_path)
    for node in model.graph.node:
        if node.op_type == "Constant" and tensor_name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    from onnx import numpy_helper
                    return numpy_helper.to_array(attr.t)
    return None


def _build_input_spec(subgraph_onnx, gt_dir, inputs_dir, trt_working_dir,
                      orig_onnx=None):
    """Prepare subgraph inputs from ground truth, return trtexec --loadInputs string.

    If orig_onnx is given, use it to resolve Constant node outputs that appear as
    subgraph inputs.
    Paths in the spec are relative to trt_working_dir (to avoid Windows C: colon issue).

    Ground-truth data is cast to match the ONNX model's input dtype before writing
    the bin file — that is the type trtexec --loadInputs expects (network binding
    type, not TRT internal precision).
    """
    os.makedirs(inputs_dir, exist_ok=True)

    input_specs = _parse_onnx_inputs(subgraph_onnx)

    with open(os.path.join(gt_dir, "name_mapping.json")) as f:
        gt_mapping = json.load(f)

    spec_parts = []
    for name, (shape, onnx_dtype) in input_specs.items():
        if name in gt_mapping:
            npy_file = gt_mapping[name]
            arr = np.load(os.path.join(gt_dir, npy_file))
        elif orig_onnx:
            const_val = _find_constant_value(orig_onnx, name)
            if const_val is not None:
                arr = const_val
            else:
                arr = (np.random.randn(*shape) * 0.02).astype(onnx_dtype)
        else:
            arr = np.random.randn(*shape).astype(onnx_dtype)

        # trtexec --loadInputs expects data matching the network binding type,
        # which is the ONNX input dtype (not TRT internal precision).
        # TRT does not support all dtypes (e.g. float64) — downgrade when needed.
        target_dtype = _TRT_COMPAT_DTYPE.get(onnx_dtype, onnx_dtype)
        if arr.dtype != target_dtype:
            arr = arr.astype(target_dtype)

        safe = name.replace("/", "_").replace(":", "_")
        bin_file = f"{safe}.bin"
        bin_path = os.path.join(inputs_dir, bin_file)
        arr.tofile(bin_path)

        # path relative to trt_working_dir
        rel_path = os.path.relpath(bin_path, trt_working_dir)
        spec_parts.append(f"{name}:{rel_path}")

    return ",".join(spec_parts)


_TRT_DUMP_PREFIX_RE = re.compile(r"^(\d{4,})_(.+)")

def _build_trt_dump_mapping_dict(trt_dump_dir, gt_dir=None):
    """Build {onnx_tensor_name: npy_filename} from TRT debug-tensor dump.

    Does NOT write a file — returns the dict directly so callers can pass it
    to LayerComparator without a disk round-trip.
    """
    npy_files = sorted(f for f in os.listdir(trt_dump_dir) if f.endswith(".npy"))

    # reverse lookup: safe_name_stem → onnx_name
    safe_to_onnx = {}
    if gt_dir:
        gt_mapping_path = os.path.join(gt_dir, "name_mapping.json")
        if os.path.exists(gt_mapping_path):
            with open(gt_mapping_path) as f:
                gt_mapping = json.load(f)
            for onnx_name, npy_file in gt_mapping.items():
                safe_to_onnx[os.path.splitext(npy_file)[0]] = onnx_name

    mapping = {}
    for f in npy_files:
        full_stem = os.path.splitext(f)[0]
        m = _TRT_DUMP_PREFIX_RE.match(full_stem)
        if m and m.group(2) in safe_to_onnx:
            stem = m.group(2)
        else:
            stem = full_stem

        onnx_name = safe_to_onnx.get(stem)
        if onnx_name is None:
            onnx_name = "/" + stem[1:] if stem.startswith("_") else stem
        mapping[onnx_name] = f

    return mapping


def _build_trt_dump_mapping(trt_dump_dir, gt_dir=None):
    """Write name_mapping.json and return its path (convenience wrapper)."""
    mapping = _build_trt_dump_mapping_dict(trt_dump_dir, gt_dir)
    path = os.path.join(trt_dump_dir, "name_mapping.json")
    with open(path, "w") as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)
    return path


def _parse_trt_export_output(json_path):
    """Parse trtexec --exportOutput JSON into {name: ndarray}.

    trtexec output format:
      [{"name": "out", "dimensions": "1x10", "values": [nan, 0.1, ...]}]
    Note: trtexec may emit NaN/Inf as bare identifiers (invalid JSON), so we
    sanitise the raw text before parsing.
    """
    with open(json_path) as f:
        raw_text = f.read()

    # trtexec writes nan / inf / -inf as bare identifiers → not valid JSON
    raw_text = re.sub(r'\bnan\b', '"__NaN__"', raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'\binf\b', '"__Inf__"', raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'\b-inf\b', '"__-Inf__"', raw_text, flags=re.IGNORECASE)

    raw = json.loads(raw_text)

    result = {}
    for entry in raw:
        name = entry["name"]
        dims_str = entry.get("dimensions", "")
        shape = [int(d) for d in dims_str.split("x")] if dims_str else []
        raw_values = entry.get("values", [])

        values = []
        for v in raw_values:
            if v == "__NaN__":
                values.append(np.nan)
            elif v == "__Inf__":
                values.append(np.inf)
            elif v == "__-Inf__":
                values.append(-np.inf)
            else:
                values.append(v)

        arr = np.array(values, dtype=np.float32).reshape(shape)
        result[name] = arr

    return result


def _compare_outputs_detailed(sub_onnx, gt_dir, trt_outputs, strict=False):
    """Compare subgraph TRT outputs against ground truth.

    Returns (max_abs_diff, errors, rows) where rows is a list of per-output
    comparison dicts suitable for comparison.json.
    """
    sub_model = onnx.load(sub_onnx)
    output_names = {o.name for o in sub_model.graph.output}

    with open(os.path.join(gt_dir, "name_mapping.json")) as f:
        gt_mapping = json.load(f)

    max_diff = 0.0
    errors = []
    rows = []
    for out_name in sorted(output_names):
        row = {"output_name": out_name}
        if out_name not in gt_mapping:
            if strict:
                errors.append(f"ground truth missing for '{out_name}'")
            row["status"] = "missing_ground_truth"
            rows.append(row)
            continue
        onnx_arr = np.load(os.path.join(gt_dir, gt_mapping[out_name]))

        if out_name not in trt_outputs:
            if strict:
                errors.append(f"TRT output '{out_name}' not found")
            row["status"] = "missing_trt_output"
            rows.append(row)
            continue

        trt_arr = trt_outputs[out_name]
        if trt_arr.shape != onnx_arr.shape:
            if strict:
                errors.append(f"shape mismatch for '{out_name}': "
                              f"TRT={trt_arr.shape} ONNX={onnx_arr.shape}")
            row["status"] = "shape_mismatch"
            row["ort_shape"] = str(onnx_arr.shape)
            row["trt_shape"] = str(trt_arr.shape)
            rows.append(row)
            continue

        diff = float(np.abs(onnx_arr.astype(np.float64) -
                            trt_arr.astype(np.float64)).max())
        max_diff = max(max_diff, diff)
        row["status"] = "ok"
        row["max_abs_diff"] = diff
        row["ort_shape"] = str(onnx_arr.shape)
        row["trt_shape"] = str(trt_arr.shape)
        row["ort_dtype"] = str(onnx_arr.dtype)
        row["trt_dtype"] = str(trt_arr.dtype)
        rows.append(row)

    return max_diff, errors, rows


def _compare_outputs(sub_onnx, gt_dir, trt_outputs, strict=False):
    """Compare subgraph TRT outputs against ground truth.
    Returns (max_abs_diff, errors)."""
    max_diff, errors, _ = _compare_outputs_detailed(
        sub_onnx, gt_dir, trt_outputs, strict=strict)
    return max_diff, errors


@dataclass
class VerifyReport:
    passed: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    threshold: float = 1e-3
    summary: dict = field(default_factory=dict)


class TRTVerifier:
    def __init__(
        self,
        onnx_path,
        save_dir,
        trtexec_path=None,
        threshold=1e-3,
        min_nodes=500,
        nodes_per_subgraph=None,
        memory_budget_mb=512,
        precision="fp16",
        verbose=True,
        input_dict=None,
    ):
        self.onnx_path = os.path.abspath(onnx_path)
        self.save_dir = save_dir
        self.trtexec_path = trtexec_path
        self.threshold = threshold
        self.min_nodes = min_nodes
        self.nodes_per_subgraph = nodes_per_subgraph
        self.memory_budget_mb = memory_budget_mb
        self.precision = precision
        self.verbose = verbose
        self.input_dict = input_dict

        self._report = None
        self._cache = {}
        self._cache_hits = 0

    # ── cache ──

    @property
    def _cache_path(self):
        return os.path.join(self.save_dir, "verify_cache.json")

    def _cache_header(self):
        """Return a fingerprint dict for this run — any change invalidates the cache."""
        return {
            "model": self.onnx_path,
            "threshold": self.threshold,
            "min_nodes": self.min_nodes,
            "precision": self.precision,
        }

    def _load_cache(self):
        path = self._cache_path
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("corrupt cache — starting fresh")
                self._cache = {}
        if not self._cache or self._cache.get("_header") != self._cache_header():
            logger.info("cache header mismatch or missing — fresh cache")
            self._cache = {"_header": self._cache_header(), "subgraphs": {}}
            self._cache_hits = 0
        else:
            self._cache_hits = len(self._cache.get("subgraphs", {}))
            logger.info("cache loaded: %d subgraph entries, header matches", self._cache_hits)

    def _save_cache(self):
        """Atomic write: temp file + rename to avoid corruption on crash."""
        tmp = self._cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._cache_path)

    def _cache_key(self, depth, idx):
        return f"depth_{depth}/sub_{idx}"

    def _cache_fingerprint(self, onnx_path, node_count):
        """File fingerprint: [file_size, node_count] (no ONNX load)."""
        try:
            return [os.stat(onnx_path).st_size, node_count]
        except Exception:
            return None

    def _cache_hit(self, cache_key, fingerprint):
        """Check cache for subgraph; returns cached entry or None."""
        cached = self._cache.get("subgraphs", {}).get(cache_key)
        if cached and cached.get("_fp") == fingerprint:
            return cached
        return None

    # ── public ──

    def _setup_logging(self):
        """Configure logger: console (level depends on *verbose*) + file (always DEBUG)."""
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)

        # file handler — full trace written to save_dir/verify.log
        fh = logging.FileHandler(
            os.path.join(self.save_dir, "verify.log"), encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(fh)

        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if self.verbose else logging.WARNING)
        ch.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))
        logger.addHandler(ch)

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self._setup_logging()
        self._load_cache()
        logger.info("verify pipeline start — %s", self.onnx_path)

        gt_dir = os.path.join(self.save_dir, "ground_truth")

        # S1: full ONNX ORT dump (skip if cached marker and output exist)
        if self._cache.get("s1_ground_truth") and os.path.exists(
                os.path.join(gt_dir, "name_mapping.json")):
            logger.info("[S1] ground truth already cached — skip")
        else:
            self._step_full_ort_dump(gt_dir, self.input_dict)
            self._cache["s1_ground_truth"] = True
            self._save_cache()

        # S2: full ONNX → TRT + layer info
        full_layer_info = os.path.join(self.save_dir, "full_layers.json")
        if self._cache.get("s2_full_trt") and os.path.exists(full_layer_info):
            logger.info("[S2] full TRT already cached — skip")
        else:
            self._step_full_trt_convert(full_layer_info)
            self._cache["s2_full_trt"] = True
            self._save_cache()

        report = VerifyReport(threshold=self.threshold)

        # S2.5: quick full-model output comparison — if they match, skip subgraph
        if self._full_output_match(gt_dir):
            report.summary = {"passed": 0, "failed": 0, "errors": 0,
                              "threshold": self.threshold, "full_output_match": True}
            self._report = report
            return report

        # S3: BFS subgraph verification
        self._verify(
            onnx_path=self.onnx_path,
            layer_info_path=full_layer_info,
            gt_dir=gt_dir,
            depth=0,
            report=report,
        )
        report.summary = {
            "passed": len(report.passed),
            "failed": len(report.failed),
            "errors": len(report.errors),
            "threshold": self.threshold,
            "full_output_match": False,
        }
        self._report = report
        return report

    @property
    def report(self):
        return self._report

    def save_report(self, path=None):
        if self._report is None:
            raise RuntimeError("Call run() first")
        path = path or os.path.join(self.save_dir, "reports", "verify_report.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "summary": self._report.summary,
                    "passed": self._report.passed,
                    "failed": self._report.failed,
                    "errors": self._report.errors,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return path

    # ── setup ──

    def _step_full_ort_dump(self, gt_dir, input_dict=None):
        os.makedirs(gt_dir, exist_ok=True)

        # remove stale dump files from previous runs
        for f in glob.glob(os.path.join(gt_dir, "dump_*.onnx")):
            os.remove(f)
        for f in glob.glob(os.path.join(gt_dir, "*.npy")):
            os.remove(f)
        mapping_path = os.path.join(gt_dir, "name_mapping.json")
        if os.path.exists(mapping_path):
            os.remove(mapping_path)

        model = GraphModel(self.onnx_path)
        builder = DumpBuilder(model, AllTensorSelector())
        if self.nodes_per_subgraph is not None and self.memory_budget_mb is None:
            builder.build(save_dir=gt_dir, nodes_per_subgraph=self.nodes_per_subgraph)
        else:
            builder.build(save_dir=gt_dir, memory_budget_mb=self.memory_budget_mb)

        # ── 初始化 carry ──
        carry = {}
        if input_dict:
            carry.update(input_dict)
        else:
            for inp in model.inputs:
                shape = tuple(
                    d if isinstance(d, int) and d > 0 else 1 for d in inp.shape
                )
                carry[inp.name] = (np.random.randn(*shape) * 0.02).astype(np.float32)

        # 保存模型输入到 ground truth
        for name, arr in carry.items():
            safe = re.sub(r"[<>:\"/\\|?*\s]", "_", name)
            safe = re.sub(r"_+", "_", safe)
            np.save(os.path.join(gt_dir, f"{safe}.npy"), arr)
            map_path = os.path.join(gt_dir, "name_mapping.json")
            if os.path.exists(map_path):
                with open(map_path, "r") as f:
                    mapping = json.load(f)
            else:
                mapping = {}
            mapping[name] = f"{safe}.npy"
            with open(map_path, "w") as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)

        dump_files = sorted(glob.glob(os.path.join(gt_dir, "dump_*.onnx")))

        # 预扫描：每个子图需要哪些运行时输入（排除 initializer）
        future_runtime_inputs = []
        for df in dump_files:
            sub = onnx.load(df)
            init_names = {init.name for init in sub.graph.initializer}
            runtime_inputs = {inp.name for inp in sub.graph.input if inp.name not in init_names}
            future_runtime_inputs.append(runtime_inputs)

        for i, dump_file in enumerate(dump_files):
            sub = onnx.load(dump_file)
            input_names = [inp.name for inp in sub.graph.input]

            # 从 carry 中提取当前子图需要的输入
            inputs = {}
            for name in input_names:
                if name in carry:
                    inputs[name] = carry[name]
                else:
                    inits = {init.name: init for init in sub.graph.initializer}
                    if name in inits:
                        continue
                    raise KeyError(
                        f"missing boundary input '{name}' for "
                        f"{os.path.basename(dump_file)}"
                    )

            runner = ORTRunner(dump_file)
            outputs = runner.dump(inputs, gt_dir)
            carry.update(outputs)

            # 释放后续子图不再需要的 tensor
            still_needed = set()
            for j in range(i + 1, len(dump_files)):
                still_needed |= future_runtime_inputs[j]
            for name in list(carry.keys()):
                if name not in still_needed:
                    del carry[name]

            logger.debug("[S1] %s: %d tensors dumped, carry=%d tensors (%.1f MB)",
                         os.path.basename(dump_file), len(outputs), len(carry),
                         sum(v.nbytes for v in carry.values()) / 1024**2)

        logger.info("[S1] ground truth ready: %s", gt_dir)

    def _step_full_trt_convert(self, layer_info_path):
        engine_path = os.path.join(os.path.dirname(layer_info_path),
                                   os.path.splitext(os.path.basename(self.onnx_path))[0] + ".engine")

        # prepare bin input from ground truth (uses same input as S1 ORT dump)
        gt_dir = os.path.join(self.save_dir, "ground_truth")
        load_inputs = {}
        if os.path.exists(os.path.join(gt_dir, "name_mapping.json")):
            input_spec = _build_input_spec(
                self.onnx_path, gt_dir,
                os.path.join(self.save_dir, "s2_inputs"),
                self.save_dir, orig_onnx=self.onnx_path)
            load_inputs = dict(p.split(":", 1) for p in input_spec.split(",") if p)

        export_output = os.path.join(self.save_dir, "full_output.json")

        builder = TRTBuilder(
            self.onnx_path,
            engine_path=engine_path,
            precision=self.precision,
            trtexec_path=self.trtexec_path,
            export_layer_info=layer_info_path,
            mark_debug_tensors=False,
            verbose=self.verbose,
        )
        builder.set_working_dir(self.save_dir)
        builder.build(
            load_inputs=load_inputs or None,
            iterations=1,
            export_output=export_output,
        )
        logger.info("[S2] full TRT engine + layer info ready: %s", layer_info_path)

    # ── full-model output comparison ──

    def _full_output_match(self, gt_dir):
        """Compare final ONNX vs TRT output via --exportOutput JSON.
        Returns True if within threshold — skips subgraph split."""
        export_json = os.path.join(self.save_dir, "full_output.json")
        if not os.path.exists(export_json):
            logger.warning("[S2.5] TRT output json not found: %s", export_json)
            return False

        try:
            trt_outputs = _parse_trt_export_output(export_json)
        except Exception as e:
            logger.warning("[S2.5] TRT output parse failed: %s", e)
            return False

        max_diff, errors = _compare_outputs(self.onnx_path, gt_dir, trt_outputs, strict=True)

        for err in errors:
            logger.warning("[S2.5] %s", err)
        if errors:
            return False

        if max_diff <= self.threshold:
            logger.info("[S2.5] full output matched (max_diff=%.6f) — "
                        "skip subgraph verification", max_diff)
            return True
        else:
            logger.info("[S2.5] full output mismatch (max_diff=%.6f) — "
                        "proceed to subgraph verification", max_diff)
            return False

    # ── cleanup ──

    @staticmethod
    def _cleanup_verify_dir(verify_dir, keep_dump=False, keep_npy=None,
                            keep_output=False):
        """Remove TRT artifacts from verify_dir to control disk usage.

        keep_dump   → keep .npy + .engine (leaf failure debugging)
        keep_npy    → set of npy paths to preserve even when keep_dump=False
        keep_output → keep sub_output.json (when output errors are large)
        NOTE: comparison.json and sub_layer_info.json are NEVER deleted.
        """
        if not os.path.isdir(verify_dir):
            return

        keep_npy = keep_npy or set()

        if not keep_dump:
            for f in glob.glob(os.path.join(verify_dir, "*.npy")):
                if f not in keep_npy:
                    os.remove(f)
            if not keep_npy:
                for f in glob.glob(os.path.join(verify_dir, "*.engine")):
                    os.remove(f)

        # metadata / temp files — always cleaned regardless of keep_dump
        for f in glob.glob(os.path.join(verify_dir, "name_mapping.json")):
            os.remove(f)
        for pattern in ["*.profile", "*.timing*", "timing_cache*", "graph_*.json"]:
            for f in glob.glob(os.path.join(verify_dir, pattern)):
                os.remove(f)
        if not keep_output:
            for f in glob.glob(os.path.join(verify_dir, "sub_output.json")):
                os.remove(f)
        inputs_dir = os.path.join(verify_dir, "inputs")
        if os.path.isdir(inputs_dir):
            shutil.rmtree(inputs_dir, ignore_errors=True)

    @staticmethod
    def _selective_cleanup_npy(verify_dir, comparison_rows, threshold):
        """Keep npy files for problematic layers (error > threshold or bad status).

        Returns the set of preserved npy paths (for logging / cache info).
        """
        # collect npy files referenced by layers that need investigation
        bad_npy = set()
        for row in comparison_rows:
            status = row.get("status", "")
            max_diff = row.get("max_abs_diff", 0)
            keep = (status != "ok" or                      # dtype_conflict, shape_mismatch, etc.
                    max_diff > threshold)                   # exceeds tolerance
            if keep:
                trt_file = row.get("trt_file")
                if trt_file:
                    bad_npy.add(os.path.join(verify_dir, trt_file))

        # delete all npy files except problematic ones
        total = 0
        kept = 0
        for f in glob.glob(os.path.join(verify_dir, "*.npy")):
            total += 1
            if f in bad_npy:
                kept += 1
            else:
                os.remove(f)

        # delete engine if no bad layers at all
        if not bad_npy:
            for f in glob.glob(os.path.join(verify_dir, "*.engine")):
                os.remove(f)

        if total:
            logger.debug("selective npy cleanup: kept %d/%d (errors + >%.1e)",
                         kept, total, threshold)
        return bad_npy

    # ── dataflow ordering ──

    @staticmethod
    def _dataflow_order(sub_onnx_files):
        """Topological sort subgraphs by producer→consumer dataflow.

        Returns list of (original_index, onnx_path, node_count) in dataflow order.
        The models are loaded once here — callers can reuse the *node_count*
        instead of loading the ONNX file a second time.
        """
        n = len(sub_onnx_files)

        # 1) extract {outputs}, {inputs}, node_count per subgraph
        outputs = []      # outputs[i] = set of tensor names produced
        inputs = []       # inputs[i]  = set of external inputs needed
        node_counts = []  # node_counts[i] = number of nodes
        for path in sub_onnx_files:
            model = onnx.load(path)
            init_names = {init.name for init in model.graph.initializer}
            produced = set()
            consumed = set()
            for node in model.graph.node:
                for o in node.output:
                    produced.add(o)
                for inp in node.input:
                    if inp not in init_names:
                        consumed.add(inp)
            outputs.append(produced)
            inputs.append(consumed - produced)
            node_counts.append(len(model.graph.node))

        if n <= 1:
            return [(0, sub_onnx_files[0], node_counts[0])] if n == 1 else []

        # 2) tensor → index of subgraph that produces it
        producer_of = {}
        for i, prod in enumerate(outputs):
            for t in prod:
                producer_of[t] = i

        # 3) build adjacency: i → j if subgraph j consumes a tensor produced by i
        children = [[] for _ in range(n)]
        indegree = [0] * n
        for j in range(n):
            for t in inputs[j]:
                i = producer_of.get(t)
                if i is not None and i != j:
                    children[i].append(j)
                    indegree[j] += 1

        # 4) Kahn topological sort — roots first
        queue = deque(i for i in range(n) if indegree[i] == 0)
        order = []
        while queue:
            i = queue.popleft()
            order.append((i, sub_onnx_files[i], node_counts[i]))
            for j in children[i]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    queue.append(j)

        # append any remaining (cycles or disconnected) in original order
        seen = {i for i, _, _ in order}
        for i in range(n):
            if i not in seen:
                order.append((i, sub_onnx_files[i], node_counts[i]))

        return order

    # ── recursive verify (BFS) ──

    def _verify(self, onnx_path, layer_info_path, gt_dir, depth, report):
        sub_dir = os.path.join(self.save_dir, f"depth_{depth}")
        split_dir = os.path.join(sub_dir, "split")
        os.makedirs(split_dir, exist_ok=True)

        # parent node count — used to detect no-progress splits (infinite loop guard)
        try:
            parent_node_count = len(onnx.load(onnx_path).graph.node)
        except Exception:
            parent_node_count = 0

        # A. partition
        try:
            partitioner = TRTPartitioner(onnx_path, layer_info_path)
            partitioner.split(save_dir=split_dir,
                              nodes_per_subgraph=self.nodes_per_subgraph or self.min_nodes)
        except Exception as e:
            logger.error("[verify:%d] partition failed: %s", depth, e)
            report.errors.append({
                "depth": depth,
                "subgraph": onnx_path, "node_count": 0,
                "error": f"partition failed: {e}",
            })
            return

        sub_onnx_files = sorted(glob.glob(os.path.join(split_dir, "subgraph_*.onnx")))
        if not sub_onnx_files:
            logger.info("[verify:%d] no subgraphs — done", depth)
            return

        # dataflow-aware topological ordering
        ordered = self._dataflow_order(sub_onnx_files)
        logger.info("[verify:%d] %d subgraphs to verify (dataflow order)",
                    depth, len(ordered))

        retry_list = []

        # B. verify ALL subgraphs at this depth (dataflow order)
        for i, sub_onnx, node_count in ordered:
            if node_count == 0:
                try:
                    sub_model = onnx.load(sub_onnx)
                    node_count = len(sub_model.graph.node)
                except Exception as e:
                    logger.warning("[verify:%d.%d] SKIP — failed to load %s: %s",
                                   depth, i, os.path.basename(sub_onnx), e)
                    report.errors.append({
                        "depth": depth, "idx": i,
                        "subgraph": sub_onnx, "node_count": 0,
                        "error": f"failed to load subgraph: {e}",
                    })
                    continue

            logger.info("[verify:%d.%d] %s: %d nodes",
                        depth, i, os.path.basename(sub_onnx), node_count)

            # ── cache lookup (fingerprint = file_size + node_count) ──
            cache_key = self._cache_key(depth, i)
            fp = self._cache_fingerprint(sub_onnx, node_count)
            cached = self._cache_hit(cache_key, fp) if fp else None

            # a subgraph that matches the parent size cannot be split further
            can_split = node_count < parent_node_count
            if not can_split and node_count >= self.min_nodes:
                logger.warning("[verify:%d.%d] split did not reduce size "
                               "(%d == parent %d) — forcing leaf",
                               depth, i, node_count, parent_node_count)

            if cached:
                logger.info("[verify:%d.%d] cached (%s) — skip", depth, i,
                            cached.get("status"))
                if cached["status"] == "passed":
                    report.passed.append(cached.get("entry", {}))
                elif cached["status"] == "queued" and can_split:
                    retry_list.append((sub_onnx, i))
                else:
                    report.failed.append(cached.get("entry", {}))
                continue

            try:
                passed, entry = self._verify_one(sub_onnx, gt_dir, depth, i, report,
                                                 node_count)
            except Exception as e:
                logger.error("[verify:%d.%d] ERROR — %s", depth, i, e)
                report.errors.append({
                    "depth": depth, "idx": i,
                    "subgraph": sub_onnx, "node_count": node_count,
                    "error": f"_verify_one crashed: {e}",
                })
                continue

            verify_dir = os.path.join(self.save_dir, f"depth_{depth}",
                                      f"verify_sub_{i}")

            # ── update cache ──
            cache_entry = {k: v for k, v in entry.items()
                           if k not in ("top_errors", "_bad_npy")}
            bad_npy = entry.get("_bad_npy", set())
            has_output_error = entry.get("output_max_diff",
                                         0) is not None and entry.get(
                "output_max_diff", 0) > self.threshold
            if passed:
                report.passed.append(entry)
                self._cleanup_verify_dir(verify_dir, keep_npy=bad_npy,
                                         keep_output=has_output_error)
                status = "passed"
            elif node_count >= self.min_nodes and can_split:
                logger.info("[verify:%d.%d] queued for deeper verification", depth, i)
                retry_list.append((sub_onnx, i))
                self._cleanup_verify_dir(verify_dir, keep_output=has_output_error)
                status = "queued"
            else:
                report.failed.append(entry)
                self._cleanup_verify_dir(verify_dir, keep_dump=True,
                                         keep_output=True)
                status = "failed"

            self._cache.setdefault("subgraphs", {})[cache_key] = {
                "_fp": fp, "status": status,
                "max_abs_diff": entry.get("max_abs_diff"), "entry": cache_entry}
            self._save_cache()

        # C. BFS: recurse into all queued subgraphs
        for sub_onnx, idx in retry_list:
            verify_dir = os.path.join(
                self.save_dir, f"depth_{depth}",
                f"verify_sub_{idx}"
            )
            sub_layer_info = os.path.join(verify_dir, "sub_layer_info.json")
            if os.path.exists(sub_layer_info):
                self._verify(sub_onnx, sub_layer_info, gt_dir, depth + 1, report)
                # deeper verification done — parent's artifacts no longer needed
                self._cleanup_verify_dir(verify_dir)
            else:
                logger.warning("[verify:%d] skip — no layer info for %s",
                               depth, os.path.basename(sub_onnx))

    # ── verify single subgraph ──

    def _verify_one(self, sub_onnx, gt_dir, depth, idx, report, node_count):
        """Verify one subgraph. Returns (passed: bool, entry: dict)."""
        verify_dir = os.path.join(self.save_dir, f"depth_{depth}",
                                  f"verify_sub_{idx}")
        os.makedirs(verify_dir, exist_ok=True)

        inputs_dir = os.path.join(verify_dir, "inputs")
        trt_dump_dir = verify_dir

        sub_layer_info = os.path.join(verify_dir, "sub_layer_info.json")

        # 叶子节点 → dump 所有中间 tensor 做逐层对比
        # 非叶子节点 → 只 export 最终输出，判断是否需要继续递归
        is_leaf = node_count < self.min_nodes
        try:
            input_spec = _build_input_spec(sub_onnx, gt_dir, inputs_dir, verify_dir,
                                           orig_onnx=self.onnx_path)
            load_inputs = dict(
                pair.split(":", 1) for pair in input_spec.split(",") if pair
            )
        except Exception as e:
            logger.error("[verify:%d.%d] input spec build failed: %s", depth, idx, e)
            report.errors.append({
                "depth": depth, "idx": idx,
                "subgraph": sub_onnx, "node_count": node_count,
                "error": f"input spec build failed: {e}",
            })
            return False, {"depth": depth, "idx": idx, "node_count": node_count}

        # Only leaf nodes need debug tensors for per-layer comparison.
        # Non-leaf nodes only need final output to decide whether to split further.
        builder = TRTBuilder(
            sub_onnx,
            precision=self.precision,
            trtexec_path=self.trtexec_path,
            export_layer_info=sub_layer_info,
            mark_debug_tensors=is_leaf,
            verbose=self.verbose,
        )
        builder.set_working_dir(verify_dir)

        export_output = os.path.join(verify_dir, "sub_output.json")
        try:
            builder.build(
                load_inputs=load_inputs,
                save_debug_tensors=is_leaf,
                export_output=export_output,
            )
        except RuntimeError as e:
            logger.error("[verify:%d.%d] TRT build failed: %s", depth, idx, e)
            report.errors.append({
                "depth": depth, "idx": idx,
                "subgraph": sub_onnx, "node_count": node_count,
                "error": str(e),
            })
            return False, {"depth": depth, "idx": idx, "node_count": node_count}

        # per-layer comparison (only for leaf nodes with debug tensors)
        summary = {}
        comparison_path = None
        max_diff = 0.0
        _bad_npy = set()
        if is_leaf:
            try:
                trt_mapping = _build_trt_dump_mapping_dict(trt_dump_dir, gt_dir=gt_dir)
                comparator = LayerComparator(sub_onnx, sub_layer_info, gt_dir, trt_dump_dir,
                                             trt_mapping=trt_mapping)
                result = comparator.compare()
                comparison_path = comparator.save_report(
                    os.path.join(verify_dir, "comparison.json"))
                logger.debug("[verify:%d.%d] comparison saved: %s (layers=%d, "
                             "matched=%d, max_diff=%.6f)",
                             depth, idx, os.path.basename(comparison_path or ""),
                             len(result.rows), result.summary.get("compared", 0),
                             result.summary.get("max_abs_diff", 0))
                max_diff = max(
                    (r.get("max_abs_diff", 0) for r in result.rows), default=0
                )
                summary = result.summary
                _bad_npy = self._selective_cleanup_npy(verify_dir, result.rows,
                                                       self.threshold)
            except Exception as e:
                logger.error("[verify:%d.%d] comparison failed: %s", depth, idx, e)
                report.errors.append({
                    "depth": depth, "idx": idx,
                    "subgraph": sub_onnx, "node_count": node_count,
                    "error": f"comparison failed: {e}",
                })
                return False, {"depth": depth, "idx": idx, "node_count": node_count}

        # final output comparison (network outputs may not be in debug tensors)
        output_max_diff = 0.0
        output_details = {}
        if os.path.exists(export_output):
            try:
                trt_outputs = _parse_trt_export_output(export_output)
                output_max_diff, output_errors, output_details = _compare_outputs_detailed(
                    sub_onnx, gt_dir, trt_outputs, strict=False)
                summary["output_max_diff"] = output_max_diff
                summary["output_errors"] = len(output_errors)
            except Exception as e:
                logger.debug("[verify:%d.%d] output compare skipped: %s", depth, idx, e)
                summary["output_max_diff"] = None
                summary["output_errors"] = 0

        max_diff = max(max_diff, output_max_diff)

        # Non-leaf nodes: generate comparison.json from output comparison data
        if not is_leaf and output_details:
            comparison_path = os.path.join(verify_dir, "comparison.json")
            with open(comparison_path, "w") as f:
                json.dump({"summary": summary, "rows": output_details}, f,
                          indent=2, ensure_ascii=False)

        entry = {
            "depth": depth,
            "idx": idx,
            "subgraph": sub_onnx,
            "node_count": node_count,
            "is_leaf": is_leaf,
            "max_abs_diff": max_diff,
            "output_max_diff": output_max_diff,
            "summary": summary,
            "sub_layer_info": sub_layer_info,
            "comparison": comparison_path,
            "_bad_npy": _bad_npy,
        }

        if max_diff > self.threshold:
            if not is_leaf:
                logger.info("[verify:%d.%d] max_abs_diff=%.6f > threshold, "
                            "queue for re-split", depth, idx, max_diff)
                return False, entry
            else:
                logger.warning("[verify:%d.%d] max_abs_diff=%.6f > threshold, "
                               "record failed (leaf)", depth, idx, max_diff)
                entry["top_errors"] = comparator.result.top_errors(n=10)
                return False, entry
        else:
            logger.info("[verify:%d.%d] max_abs_diff=%.6f <= threshold, passed",
                        depth, idx, max_diff)
            return True, entry
