import os
import json
import glob
import numpy as np
import onnx
from onnx import TensorProto
from dataclasses import dataclass, field

from onnx_probe import GraphModel, AllTensorSelector, DumpBuilder, ORTRunner
from trt_op import TRTBuilder
from onnx_trt_map import LayerMapper
from subgraph_split_by_trt import TRTPartitioner
from compare_onnx_trt import LayerComparator


_ONNX_DTYPE_MAP = {
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.BOOL: bool,
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


def _build_input_spec(subgraph_onnx, gt_dir, inputs_dir, trt_working_dir, orig_onnx=None):
    """Prepare subgraph inputs from ground truth, return trtexec --loadInputs string.
    If orig_onnx is given, use it to resolve Constant node outputs that appear as
    subgraph inputs.
    Paths in the spec are relative to trt_working_dir (to avoid Windows C: colon issue)."""
    os.makedirs(inputs_dir, exist_ok=True)

    input_specs = _parse_onnx_inputs(subgraph_onnx)

    with open(os.path.join(gt_dir, "name_mapping.json")) as f:
        gt_mapping = json.load(f)

    spec_parts = []
    for name, (shape, dtype) in input_specs.items():
        if name in gt_mapping:
            npy_file = gt_mapping[name]
            arr = np.load(os.path.join(gt_dir, npy_file))
        elif orig_onnx:
            const_val = _find_constant_value(orig_onnx, name)
            if const_val is not None:
                arr = const_val
            else:
                arr = np.random.randn(*shape).astype(dtype)
        else:
            arr = np.random.randn(*shape).astype(dtype)

        safe = name.replace("/", "_").replace(":", "_")
        bin_file = f"{safe}.bin"
        bin_path = os.path.join(inputs_dir, bin_file)
        arr.tofile(bin_path)

        # path relative to trt_working_dir
        rel_path = os.path.relpath(bin_path, trt_working_dir)
        spec_parts.append(f"{name}:{rel_path}")

    return ",".join(spec_parts)


def _build_trt_dump_mapping(trt_dump_dir):
    """Create name_mapping.json in trt_dump_dir from whatever .npy files exist."""
    npy_files = sorted(f for f in os.listdir(trt_dump_dir) if f.endswith(".npy"))
    mapping = {}
    for f in npy_files:
        name = os.path.splitext(f)[0]
        mapping[name] = f
    path = os.path.join(trt_dump_dir, "name_mapping.json")
    with open(path, "w") as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)
    return path


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
        min_nodes=20,
        nodes_per_subgraph=100,
        precision="fp16",
        verbose=True,
    ):
        self.onnx_path = os.path.abspath(onnx_path)
        self.save_dir = save_dir
        self.trtexec_path = trtexec_path
        self.threshold = threshold
        self.min_nodes = min_nodes
        self.nodes_per_subgraph = nodes_per_subgraph
        self.precision = precision
        self.verbose = verbose

        self._report = None

    # ── public ──

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)

        gt_dir = os.path.join(self.save_dir, "ground_truth")

        # S1: full ONNX ORT dump
        self._step_full_ort_dump(gt_dir)

        # S2: full ONNX → TRT + layer info
        full_layer_info = os.path.join(self.save_dir, "full_layers.json")
        self._step_full_trt_convert(full_layer_info)

        # S3: verify recursively (also generates full LayerMapper internally)
        report = VerifyReport(threshold=self.threshold)
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

    def _step_full_ort_dump(self, gt_dir):
        os.makedirs(gt_dir, exist_ok=True)

        model = GraphModel(self.onnx_path)
        builder = DumpBuilder(model, AllTensorSelector())
        builder.build(save_dir=gt_dir, nodes_per_subgraph=self.nodes_per_subgraph)

        carry = {}
        for inp in model.inputs:
            shape = tuple(
                d if isinstance(d, int) and d > 0 else 1 for d in inp.shape
            )
            carry[inp.name] = np.random.randn(*shape).astype(np.float32)

        for dump_file in sorted(glob.glob(os.path.join(gt_dir, "dump_*.onnx"))):
            sub = onnx.load(dump_file)
            input_names = [inp.name for inp in sub.graph.input]

            inputs = {}
            for name in input_names:
                if name not in carry:
                    raise KeyError(
                        f"missing boundary input '{name}' for "
                        f"{os.path.basename(dump_file)}"
                    )
                inputs[name] = carry[name]

            runner = ORTRunner(dump_file)
            outputs = runner.dump(inputs, gt_dir)
            carry.update(outputs)
            if self.verbose:
                print(
                    f"[S1] {os.path.basename(dump_file)}: "
                    f"{len(outputs)} tensors dumped"
                )

        print(f"[S1] ground truth ready: {gt_dir}")

    def _step_full_trt_convert(self, layer_info_path):
        engine_path = os.path.join(os.path.dirname(layer_info_path),
                                   os.path.splitext(os.path.basename(self.onnx_path))[0] + ".engine")

        builder = TRTBuilder(
            self.onnx_path,
            engine_path=engine_path,
            precision=self.precision,
            trtexec_path=self.trtexec_path,
            export_layer_info=layer_info_path,
            verbose=self.verbose,
        )
        builder.set_working_dir(os.path.dirname(layer_info_path))
        builder.build()
        print(f"[S2] full TRT engine + layer info ready: {layer_info_path}")

    # ── recursive verify ──

    def _verify(self, onnx_path, layer_info_path, gt_dir, depth, report):
        sub_dir = os.path.join(self.save_dir, f"depth_{depth}")
        split_dir = os.path.join(sub_dir, "split")
        os.makedirs(split_dir, exist_ok=True)

        # partition
        partitioner = TRTPartitioner(onnx_path, layer_info_path)
        partitioner.split(save_dir=split_dir, nodes_per_subgraph=self.min_nodes)

        # pick first subgraph (sorted by name = most upstream)
        sub_onnx_files = sorted(glob.glob(os.path.join(split_dir, "subgraph_*.onnx")))
        if not sub_onnx_files:
            print(f"[verify:{depth}] no subgraphs — done")
            return
        sub_onnx = sub_onnx_files[0]

        # count onnx nodes
        sub_model = onnx.load(sub_onnx)
        node_count = len(sub_model.graph.node)
        print(f"[verify:{depth}] subgraph_0: {node_count} nodes")

        verify_dir = os.path.join(sub_dir, "verify_sub_0")
        os.makedirs(verify_dir, exist_ok=True)

        inputs_dir = os.path.join(verify_dir, "inputs")
        trt_dump_dir = os.path.join(verify_dir, "trt_dump")
        os.makedirs(trt_dump_dir, exist_ok=True)

        sub_layer_info = os.path.join(verify_dir, "sub_layer_info.json")

        if node_count >= self.min_nodes:
            # ── >= min_nodes: ORT dump already available (ground truth),
            #     just run TRT with debug tensors ──

            # prepare subgraph inputs
            input_spec = _build_input_spec(sub_onnx, gt_dir, inputs_dir, verify_dir,
                                           orig_onnx=self.onnx_path)
            load_inputs = dict(
                pair.split(":", 1) for pair in input_spec.split(",") if pair
            )

            # build TRT engine + dump debug tensors
            builder = TRTBuilder(
                sub_onnx,
                precision=self.precision,
                trtexec_path=self.trtexec_path,
                export_layer_info=sub_layer_info,
                verbose=self.verbose,
            )
            builder.set_working_dir(verify_dir)
            try:
                builder.build(
                    load_inputs=load_inputs,
                    save_debug_tensors=True,
                )
            except RuntimeError as e:
                report.errors.append(
                    {
                        "depth": depth,
                        "subgraph": sub_onnx,
                        "node_count": node_count,
                        "error": str(e),
                    }
                )
                return

        else:
            # ── < min_nodes: TRT with markUnfusedTensorsAsDebugTensors ──

            input_spec = _build_input_spec(sub_onnx, gt_dir, inputs_dir, verify_dir,
                                           orig_onnx=self.onnx_path)
            load_inputs = dict(
                pair.split(":", 1) for pair in input_spec.split(",") if pair
            )

            builder = TRTBuilder(
                sub_onnx,
                precision=self.precision,
                trtexec_path=self.trtexec_path,
                export_layer_info=sub_layer_info,
                mark_debug_tensors=True,
                verbose=self.verbose,
            )
            builder.set_working_dir(verify_dir)
            try:
                builder.build(
                    load_inputs=load_inputs,
                    save_debug_tensors=True,
                )
            except RuntimeError as e:
                report.errors.append(
                    {
                        "depth": depth,
                        "subgraph": sub_onnx,
                        "node_count": node_count,
                        "error": str(e),
                    }
                )
                return

        # build TRT dump name_mapping
        _build_trt_dump_mapping(trt_dump_dir)

        # LayerMapper for subgraph
        mapper = LayerMapper(sub_onnx, sub_layer_info)

        # LayerComparator
        comparator = LayerComparator(sub_onnx, sub_layer_info, gt_dir, trt_dump_dir)
        result = comparator.compare()

        # save comparison
        comparator.save_report(os.path.join(verify_dir, "comparison.json"))

        # check threshold
        max_diff = max(
            (r.get("max_abs_diff", 0) for r in result.rows), default=0
        )

        entry = {
            "depth": depth,
            "subgraph": sub_onnx,
            "node_count": node_count,
            "max_abs_diff": max_diff,
            "summary": result.summary,
        }

        if max_diff > self.threshold:
            if node_count >= self.min_nodes:
                print(
                    f"[verify:{depth}] max_abs_diff={max_diff:.6f} > threshold, "
                    f"recurse deeper"
                )
                self._verify(
                    onnx_path=sub_onnx,
                    layer_info_path=sub_layer_info,
                    gt_dir=gt_dir,
                    depth=depth + 1,
                    report=report,
                )
            else:
                print(
                    f"[verify:{depth}] max_abs_diff={max_diff:.6f} > threshold, "
                    f"record failed (leaf)"
                )
                entry["top_errors"] = comparator.result.top_errors(n=10)
                report.failed.append(entry)
        else:
            print(f"[verify:{depth}] max_abs_diff={max_diff:.6f} <= threshold, passed")
            report.passed.append(entry)
