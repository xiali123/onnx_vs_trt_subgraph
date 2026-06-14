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


def _compare_outputs(sub_onnx, gt_dir, trt_outputs, strict=False):
    """Compare subgraph TRT outputs against ground truth.

    Returns (max_abs_diff, errors) where errors is a list of description strings.
    In strict mode missing outputs and shape mismatches are counted as errors;
    in lenient mode they are silently skipped (useful for partial subgraphs).
    """
    sub_model = onnx.load(sub_onnx)
    output_names = {o.name for o in sub_model.graph.output}

    with open(os.path.join(gt_dir, "name_mapping.json")) as f:
        gt_mapping = json.load(f)

    max_diff = 0.0
    errors = []
    for out_name in output_names:
        if out_name not in gt_mapping:
            if strict:
                errors.append(f"ground truth missing for '{out_name}'")
            continue
        onnx_arr = np.load(os.path.join(gt_dir, gt_mapping[out_name]))

        if out_name not in trt_outputs:
            if strict:
                errors.append(f"TRT output '{out_name}' not found")
            continue

        trt_arr = trt_outputs[out_name]
        if trt_arr.shape != onnx_arr.shape:
            if strict:
                errors.append(f"shape mismatch for '{out_name}': "
                              f"TRT={trt_arr.shape} ONNX={onnx_arr.shape}")
            continue

        diff = float(np.abs(onnx_arr.astype(np.float64) -
                            trt_arr.astype(np.float64)).max())
        max_diff = max(max_diff, diff)

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
        logger.info("verify pipeline start — %s", self.onnx_path)

        gt_dir = os.path.join(self.save_dir, "ground_truth")

        # S1: full ONNX ORT dump
        self._step_full_ort_dump(gt_dir, self.input_dict)

        # S2: full ONNX → TRT + layer info
        full_layer_info = os.path.join(self.save_dir, "full_layers.json")
        self._step_full_trt_convert(full_layer_info)

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
    def _cleanup_verify_dir(verify_dir, keep_layer_info=True, keep_dump=False):
        """Remove TRT artifacts from verify_dir to control disk usage.

        keep_layer_info=True  → keep sub_layer_info.json (needed for deeper split)
        keep_dump=True        → keep .npy + .engine + comparison.json (leaf failure debugging)
        """
        if not os.path.isdir(verify_dir):
            return

        if not keep_dump:
            # delete large artifacts: debug tensor dumps + engine
            for f in glob.glob(os.path.join(verify_dir, "*.npy")):
                os.remove(f)
            for f in glob.glob(os.path.join(verify_dir, "*.engine")):
                os.remove(f)
            for f in glob.glob(os.path.join(verify_dir, "name_mapping.json")):
                os.remove(f)
            # trtexec profile artifacts
            for pattern in ["*.profile", "*.timing*", "timing_cache*", "graph_*.json"]:
                for f in glob.glob(os.path.join(verify_dir, pattern)):
                    os.remove(f)
            # comparison.json (regenerated at deeper levels if needed)
            for f in glob.glob(os.path.join(verify_dir, "comparison.json")):
                os.remove(f)
            # input bin files (consumed by trtexec, no longer needed)
            inputs_dir = os.path.join(verify_dir, "inputs")
            if os.path.isdir(inputs_dir):
                shutil.rmtree(inputs_dir, ignore_errors=True)

        if not keep_layer_info:
            for f in glob.glob(os.path.join(verify_dir, "sub_layer_info.json")):
                os.remove(f)
            for f in glob.glob(os.path.join(verify_dir, "sub_output.json")):
                os.remove(f)

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

        # A. partition
        try:
            partitioner = TRTPartitioner(onnx_path, layer_info_path)
            partitioner.split(save_dir=split_dir, nodes_per_subgraph=self.min_nodes)
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

            if passed:
                report.passed.append(entry)
                self._cleanup_verify_dir(verify_dir, keep_layer_info=False)
            elif node_count >= self.min_nodes:
                logger.info("[verify:%d.%d] queued for deeper verification", depth, i)
                retry_list.append((sub_onnx, i))
                # 非叶子：删除 npy/engine，保留 sub_layer_info 给更深层用
                self._cleanup_verify_dir(verify_dir, keep_layer_info=True)
            else:
                report.failed.append(entry)
                # 叶子失败：保留所有 npy + engine 用于排查
                self._cleanup_verify_dir(verify_dir, keep_layer_info=True,
                                         keep_dump=True)

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
                self._cleanup_verify_dir(verify_dir, keep_layer_info=False)
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

        builder = TRTBuilder(
            sub_onnx,
            precision=self.precision,
            trtexec_path=self.trtexec_path,
            export_layer_info=sub_layer_info,
            mark_debug_tensors=True,
            verbose=self.verbose,
        )
        builder.set_working_dir(verify_dir)

        # build TRT engine with debug tensors for per-layer comparison
        try:
            builder.build(
                load_inputs=load_inputs,
                save_debug_tensors=True,
            )
        except RuntimeError as e:
            logger.error("[verify:%d.%d] TRT build failed: %s", depth, idx, e)
            report.errors.append({
                "depth": depth, "idx": idx,
                "subgraph": sub_onnx, "node_count": node_count,
                "error": str(e),
            })
            return False, {"depth": depth, "idx": idx, "node_count": node_count}

        # per-layer comparison (both leaf and non-leaf)
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
        except Exception as e:
            logger.error("[verify:%d.%d] comparison failed: %s", depth, idx, e)
            report.errors.append({
                "depth": depth, "idx": idx,
                "subgraph": sub_onnx, "node_count": node_count,
                "error": f"comparison failed: {e}",
            })
            return False, {"depth": depth, "idx": idx, "node_count": node_count}

        entry = {
            "depth": depth,
            "idx": idx,
            "subgraph": sub_onnx,
            "node_count": node_count,
            "is_leaf": is_leaf,
            "max_abs_diff": max_diff,
            "summary": summary,
            "sub_layer_info": sub_layer_info,
            "comparison": comparison_path,
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
