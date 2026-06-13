"""
Example: ONNX -> TensorRT engine conversion + inference with trt_op.

Usage:
    # dry-run (print commands, no trtexec required)
    python trt_convert.py

    # real run (requires trtexec + ONNX model)
    python trt_convert.py --real --onnx large_model.onnx
"""
import os
import argparse

from trt_op import TRTBuilder


def demo_dry_run():
    print("=" * 60)
    print("DRY-RUN: trtexec command preview")
    print("=" * 60)

    base = os.path.dirname(__file__)

    # ── build (with random inputs) ──
    builder = TRTBuilder(
        os.path.join(base, "large_model.onnx"),
        engine_path=os.path.join(base, "large_model.engine"),
        precision="fp16",
        export_profile=os.path.join(base, "profile.json"),
        export_layer_info=os.path.join(base, "layers.json"),
        static_plugins="/home/kevin.xia/libNv12ToRgbPlugin.so",
    )
    builder.set_timing_cache(os.path.join(base, "timing.cache"))

    print("\n[build] — no inputs, trtexec uses random data")
    print(" ".join(builder._build_args()))

    # ── run (with explicit inputs) ──
    builder2 = TRTBuilder(
        os.path.join(base, "large_model.onnx"),
        engine_path=os.path.join(base, "large_model.engine"),
    )

    print("\n[run] — with explicit inputs")
    from trt_op.builder import TRTBuilder as _B
    b = _B(os.path.join(base, "large_model.onnx"))
    print(f"  trtexec --loadEngine={b.engine_path} --loadInputs=input:input.bin "
          f"--iterations=100 --saveAllDebugTensors --exportOutput=output.json")


def demo_real(onnx_path):
    print("=" * 60)
    print("REAL RUN: building engine + inference")
    print("=" * 60)

    base = os.path.splitext(onnx_path)[0]

    builder = TRTBuilder(
        onnx_path,
        engine_path=base + ".engine",
        precision="fp16",
        strongly_typed=True,
        max_aux_streams=0,
        export_profile=base + "_profile.json",
        export_layer_info=base + "_layers.json",
        dump_profile=True,
        separate_profile_run=True,
        verbose=True,
    )
    builder.set_timing_cache(os.path.join(os.path.dirname(__file__), "timing.cache"))

    # build + run with built-in random inputs (no --loadInputs)
    print("\n--- Build + run (random inputs) ---")
    result = builder.build(iterations=50, save_debug_tensors=True)
    print(f"Engine saved to: {builder.engine_path}")
    print(f"Latency: {result.latency_ms:.3f} ms  ({result.p50_ms:.2f} / {result.p90_ms:.2f} / {result.p99_ms:.2f})")
    print(f"Throughput: {result.throughput:.1f} qps")

    # run with explicit inputs
    print("\n--- Run with explicit inputs ---")
    result = builder.run(
        load_inputs={"input": "input.bin"},
        iterations=50,
        save_debug_tensors=True,
        export_output=base + "_output.json",
    )
    print(f"Latency: {result.latency_ms:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Actually run trtexec")
    parser.add_argument("--onnx", default=None, help="Path to ONNX model")
    args = parser.parse_args()

    if args.real:
        if not args.onnx:
            parser.error("--real requires --onnx <path>")
        demo_real(args.onnx)
    else:
        demo_dry_run()
