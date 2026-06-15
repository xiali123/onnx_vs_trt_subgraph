"""
Batch ONNX vs TRT comparison (full model, no subgraph splitting).

Usage:
    python batch_compare.py --onnx-dir <dir> --gt-dir <dir> [--save-dir <dir>]

For each .onnx in --onnx-dir: find matching inputs from --gt-dir, run TRT and ORT,
compare outputs, write per-model result and summary.
"""

import os
import sys
import json
import glob
import argparse
import logging
import numpy as np
import onnx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from verify_pipeline.verifier import (
    _parse_onnx_inputs,
    _build_input_spec,
    _parse_trt_export_output,
    _TRT_COMPAT_DTYPE,
    _ONNX_DTYPE_MAP,
)
from trt_op import TRTBuilder
from onnx_probe import ORTRunner

logger = logging.getLogger("batch_compare")


def _load_gt_inputs(gt_dir):
    """Load all tensors from ground truth directory. Returns {name: ndarray}."""
    mapping_path = os.path.join(gt_dir, "name_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"name_mapping.json not found in {gt_dir}")
    with open(mapping_path) as f:
        mapping = json.load(f)
    tensors = {}
    for name, npy_file in mapping.items():
        npy_path = os.path.join(gt_dir, npy_file)
        if os.path.exists(npy_path):
            tensors[name] = np.load(npy_path)
    return tensors


def _load_inputs_for_onnx(onnx_path, gt_dir):
    """Return {input_name: ndarray} for the given ONNX model from ground truth.

    Inputs not found in ground truth get random fallback data (with warning).
    """
    input_specs = _parse_onnx_inputs(onnx_path)
    gt_tensors = _load_gt_inputs(gt_dir)

    inputs = {}
    for name, (shape, onnx_dtype) in input_specs.items():
        if name in gt_tensors:
            arr = gt_tensors[name]
        else:
            logger.warning("%s: input '%s' not in ground truth — using random fallback",
                           os.path.basename(onnx_path), name)
            arr = (np.random.randn(*shape) * 0.02).astype(onnx_dtype)

        # match ONNX dtype; downgrade float64 → float32 for TRT compat
        target_dtype = _TRT_COMPAT_DTYPE.get(onnx_dtype, onnx_dtype)
        if arr.dtype != target_dtype:
            arr = arr.astype(target_dtype)
        inputs[name] = arr

    return inputs


def _compare_tensors(ort_outputs, trt_outputs, threshold=1e-3):
    """Compare ORT outputs vs TRT outputs.

    Returns (passed, max_diff, details) where details is {name: {max_diff, passed, ...}}.
    """
    all_names = set(ort_outputs.keys()) | set(trt_outputs.keys())
    details = {}
    max_diff = 0.0

    for name in sorted(all_names):
        if name not in ort_outputs:
            details[name] = {"status": "missing_ort"}
            continue
        if name not in trt_outputs:
            details[name] = {"status": "missing_trt"}
            continue

        ort_arr = ort_outputs[name]
        trt_arr = trt_outputs[name]

        if ort_arr.shape != trt_arr.shape:
            diff = float("inf")
            details[name] = {
                "status": "shape_mismatch",
                "ort_shape": str(ort_arr.shape),
                "trt_shape": str(trt_arr.shape),
                "max_abs_diff": diff,
            }
            max_diff = max(max_diff, diff)
            continue

        diff = float(np.abs(ort_arr.astype(np.float64) -
                            trt_arr.astype(np.float64)).max())
        max_diff = max(max_diff, diff)

        details[name] = {
            "status": "ok" if diff <= threshold else "mismatch",
            "max_abs_diff": diff,
            "ort_shape": str(ort_arr.shape),
            "trt_shape": str(trt_arr.shape),
            "ort_dtype": str(ort_arr.dtype),
            "trt_dtype": str(trt_arr.dtype),
        }

    passed = max_diff <= threshold
    return passed, max_diff, details


def process_one(onnx_path, gt_dir, save_dir, precision, threshold,
                trtexec_path, verbose):
    """Process a single ONNX: TRT + ORT inference, compare, return result dict."""
    onnx_name = os.path.splitext(os.path.basename(onnx_path))[0]
    # Use absolute paths throughout — _build_input_spec uses os.path.relpath
    # which fails on Windows when mixing relative and absolute paths.
    def _fwdslash(p):
        return os.path.abspath(p).replace("\\", "/")

    work_dir = _fwdslash(os.path.join(save_dir, onnx_name))
    os.makedirs(work_dir, exist_ok=True)

    logger.info("=== %s ===", onnx_name)

    # 1. load inputs from ground truth
    gt_dir_abs = _fwdslash(gt_dir)
    onnx_path_abs = _fwdslash(onnx_path)
    logger.info("[%s] loading inputs from %s", onnx_name, gt_dir_abs)
    ort_inputs = _load_inputs_for_onnx(onnx_path_abs, gt_dir_abs)
    logger.info("[%s] %d inputs loaded", onnx_name, len(ort_inputs))

    # 2. prepare TRT inputs (bin files + --loadInputs string)
    inputs_dir = os.path.join(work_dir, "inputs")
    input_spec = _build_input_spec(onnx_path_abs, gt_dir_abs,
                                   inputs_dir, work_dir,
                                   orig_onnx=onnx_path_abs)

    # 3. TRT inference
    logger.info("[%s] running TRT inference...", onnx_name)
    engine_path = os.path.join(work_dir, onnx_name + ".engine").replace("\\", "/")
    export_output = os.path.join(work_dir, "trt_output.json").replace("\\", "/")

    trt_builder = TRTBuilder(
        onnx_path_abs,
        engine_path=engine_path,
        precision=precision,
        trtexec_path=trtexec_path,
        mark_debug_tensors=False,
        strongly_typed=(precision != "fp16"),
        verbose=verbose,
    )
    trt_builder.set_working_dir(work_dir)

    load_inputs = {}
    for spec in input_spec.split(","):
        if ":" in spec:
            k, v = spec.split(":", 1)
            load_inputs[k] = v

    try:
        trt_builder.build(
            load_inputs=load_inputs or None,
            iterations=1,
            export_output=export_output,
        )
    except RuntimeError as e:
        logger.error("[%s] TRT build failed: %s", onnx_name, e)
        return {
            "onnx": onnx_path,
            "status": "trt_build_failed",
            "error": str(e),
        }

    trt_outputs = {}
    if os.path.exists(export_output):
        trt_outputs = _parse_trt_export_output(export_output)
    logger.info("[%s] TRT: %d outputs", onnx_name, len(trt_outputs))

    # 4. ORT inference
    logger.info("[%s] running ORT inference...", onnx_name)
    ort_runner = ORTRunner(onnx_path)
    ort_outputs = ort_runner.run(ort_inputs)
    logger.info("[%s] ORT: %d outputs", onnx_name, len(ort_outputs))

    # save ORT outputs
    ort_dir = os.path.join(work_dir, "ort_output")
    os.makedirs(ort_dir, exist_ok=True)
    for name, arr in ort_outputs.items():
        safe = name.replace("/", "_").replace(":", "_")
        np.save(os.path.join(ort_dir, f"{safe}.npy"), arr)

    # 5. compare
    passed, max_diff, details = _compare_tensors(ort_outputs, trt_outputs, threshold)
    status = "passed" if passed else "failed"
    logger.info("[%s] max_abs_diff=%.6f threshold=%.0e → %s",
                onnx_name, max_diff, threshold, status)

    result = {
        "onnx": onnx_path,
        "status": status,
        "max_abs_diff": max_diff,
        "threshold": threshold,
        "num_inputs": len(ort_inputs),
        "num_ort_outputs": len(ort_outputs),
        "num_trt_outputs": len(trt_outputs),
        "per_output": details,
    }

    # save per-model result
    result_path = os.path.join(work_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch ONNX vs TRT comparison (full model, no subgraph split)")
    parser.add_argument("--onnx-dir", required=True,
                        help="Directory containing .onnx files")
    parser.add_argument("--gt-dir", required=True,
                        help="Ground truth directory (name_mapping.json + .npy files)")
    parser.add_argument("--save-dir", default="batch_compare_out",
                        help="Output directory (default: batch_compare_out)")
    parser.add_argument("--threshold", type=float, default=1e-3,
                        help="Max allowed abs diff (default: 1e-3)")
    parser.add_argument("--precision", default="fp16",
                        help="TRT precision (default: fp16)")
    parser.add_argument("--trtexec-path", default=None,
                        help="Path to trtexec binary")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # logging setup
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(os.path.join(args.save_dir, "batch_compare.log"),
                             encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    # scan ONNX files
    onnx_files = sorted(glob.glob(os.path.join(args.onnx_dir, "*.onnx")))
    if not onnx_files:
        logger.error("No .onnx files found in %s", args.onnx_dir)
        sys.exit(1)
    logger.info("Found %d ONNX file(s) in %s", len(onnx_files), args.onnx_dir)

    # process each ONNX
    results = []
    passed_count = 0
    failed_count = 0

    for onnx_path in onnx_files:
        result = process_one(
            onnx_path=onnx_path,
            gt_dir=args.gt_dir,
            save_dir=args.save_dir,
            precision=args.precision,
            threshold=args.threshold,
            trtexec_path=args.trtexec_path,
            verbose=args.verbose,
        )
        results.append(result)
        if result["status"] == "passed":
            passed_count += 1
        else:
            failed_count += 1

    # summary
    summary = {
        "total": len(results),
        "passed": passed_count,
        "failed": failed_count,
        "threshold": args.threshold,
        "precision": args.precision,
        "onnx_dir": args.onnx_dir,
        "gt_dir": args.gt_dir,
        "details": [
            {
                "onnx": r["onnx"],
                "status": r["status"],
                "max_abs_diff": r.get("max_abs_diff"),
            }
            for r in results
        ],
    }

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=== DONE ===")
    logger.info("Total: %d  Passed: %d  Failed: %d", len(results), passed_count,
                failed_count)
    logger.info("Summary: %s", summary_path)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
