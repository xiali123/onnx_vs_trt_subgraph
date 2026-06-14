"""
Demo: TRTVerifier — ONNX vs TRT precision verification pipeline.

Requires: trtexec + onnxruntime
Usage:
    python verify_demo.py                # dry-run: show what would happen
    python verify_demo.py --real         # real run (needs trtexec)
    python verify_demo.py --real --onnx path/to/model.onnx
"""
import os
import sys
import argparse
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from verify_pipeline import TRTVerifier


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def demo_dry_run():
    """Print pipeline structure without executing."""
    print("=" * 60)
    print("DRY-RUN: TRTVerifier pipeline preview")
    print("=" * 60)

    onnx_path = os.path.join(THIS_DIR, "large_model.onnx")
    save_dir = os.path.join(THIS_DIR, "verify_out_v1")

    verifier = TRTVerifier(
        onnx_path=onnx_path,
        save_dir=save_dir,
        threshold=1e-3,
        min_nodes=20,
        nodes_per_subgraph=100,
    )

    print(f"ONNX         : {onnx_path}")
    print(f"Output       : {save_dir}")
    print(f"Threshold    : {verifier.threshold}")
    print(f"Min nodes    : {verifier.min_nodes}")
    print(f"Precision    : {verifier.precision}")
    print()
    print("Pipeline steps:")
    print("  S1: DumpBuilder → dump_*.onnx → ORTRunner cascade dump")
    print("       → ground_truth/*.npy + name_mapping.json")
    print("  S2: TRTBuilder.build() → full engine + full_layers.json")
    print()
    print("  [verify:{0}] TRTPartitioner.split(nodes_per_subgraph=20)")
    print("       → split/subgraph_0.onnx")
    print("       if nodes >= 20:")
    print("         → data_op prep inputs → bin")
    print("         → TRTBuilder(subgraph.onnx).build(save_debug_tensors=True)")
    print("         → LayerMapper + LayerComparator")
    print("         → if max_abs_diff > threshold: recurse deeper")
    print("       else:")
    print("         → TRTBuilder with mark_debug_tensors + save_debug_tensors")
    print("         → LayerMapper + LayerComparator")
    print("         → if max_abs_diff > threshold: record failed report")
    print()
    print("Output tree:")
    print(f"  {save_dir}/")
    print(f"    ground_truth/            # S1: all intermediate tensor .npy")
    print(f"    full_layers.json         # S2: TRT layer info")
    print(f"    depth_0/                 # recursive verify")
    print(f"      split/subgraph_0.onnx  # TRTPartitioner output")
    print(f"      verify_sub_0/")
    print(f"        inputs/              # data_op bin files")
    print(f"        trt_dump/            # TRT debug tensor .npy")
    print(f"        sub_layer_info.json  # subgraph TRT layer info")
    print(f"        comparison.json      # LayerComparator result")
    print(f"    reports/verify_report.json")


def demo_real(onnx_path):
    """Run the full pipeline."""
    print("=" * 60)
    print("REAL RUN: TRTVerifier pipeline")
    print("=" * 60)

    save_dir = os.path.join(THIS_DIR, "verify_out_v3")

    verifier = TRTVerifier(
        onnx_path=onnx_path,
        save_dir=save_dir,
        threshold=1e-3,
        min_nodes=500,
        nodes_per_subgraph=1000,
        verbose=True,
    )

    t0 = time.time()
    try:
        report = verifier.run()
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Elapsed     : {elapsed:.1f}s")
    print(f"Threshold   : {report.threshold}")
    print(f"Passed      : {report.summary['passed']}")
    print(f"Failed      : {report.summary['failed']}")
    print(f"Errors      : {report.summary['errors']}")

    if report.passed:
        print()
        print("--- Passed ---")
        for e in report.passed:
            print(f"  depth={e['depth']} nodes={e['node_count']} "
                  f"max_diff={e['max_abs_diff']:.6f}")

    if report.failed:
        print()
        print("--- Failed (leaf subgraphs < min_nodes) ---")
        for e in report.failed:
            print(f"  depth={e['depth']} nodes={e['node_count']} "
                  f"max_diff={e['max_abs_diff']:.6f}")
            if e.get("top_errors"):
                for row in e["top_errors"][:3]:
                    print(f"    {row.get('trt_layer','?')}: "
                          f"max_abs_diff={row.get('max_abs_diff','?')} "
                          f"cosine_sim={row.get('cosine_sim','?')}")

    if report.errors:
        print()
        print("--- Errors ---")
        for e in report.errors:
            print(f"  depth={e['depth']} {os.path.basename(e['subgraph'])}: {e['error']}")

    # save final report
    report_path = verifier.save_report()
    print(f"\nFull report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Actually run trtexec")
    parser.add_argument("--onnx", default=None, help="Path to ONNX model (default: tiny_test.onnx)")
    args = parser.parse_args()

    if args.real:
        onnx_path = args.onnx or os.path.join(THIS_DIR, "tiny_test.onnx")
        if not os.path.exists(onnx_path):
            print(f"ONNX model not found: {onnx_path}")
            print("Run create_onnx.py first or specify --onnx <path>")
            sys.exit(1)
        sys.exit(demo_real(onnx_path))
    else:
        demo_dry_run()
