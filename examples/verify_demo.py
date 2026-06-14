"""
Demo: TRTVerifier — ONNX vs TRT precision verification pipeline.

Requires: trtexec + onnxruntime
Usage:
    python verify_demo.py                              # dry-run
    python verify_demo.py --real                       # real run (needs trtexec)
    python verify_demo.py --real --onnx path/to/model.onnx
    python verify_demo.py --real --pkl path/to/inputs.pkl   # 真值输入
"""
import os
import sys
import argparse
import pickle
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
        min_nodes=20,                 # leaf threshold: smaller → stop splitting
        nodes_per_subgraph=200,       # partition target: subgraph size
        memory_budget_mb=512,         # S1 memory limit for ORT dump
    )

    print(f"ONNX              : {onnx_path}")
    print(f"Output            : {save_dir}")
    print(f"Threshold         : {verifier.threshold}")
    print(f"Leaf threshold    : {verifier.min_nodes} nodes")
    print(f"Partition target  : {verifier.nodes_per_subgraph} nodes/subgraph")
    print(f"Precision         : {verifier.precision}")
    print()
    print("Pipeline steps:")
    print("  S1: DumpBuilder → dump_*.onnx → ORTRunner cascade dump")
    print("       → ground_truth/*.npy + name_mapping.json  (cached after first run)")
    print("  S2: TRTBuilder.build() → full engine + full_layers.json  (cached)")
    print("  S2.5: Full-model output comparison → skip subgraphs if match")
    print()
    print("  S3: BFS subgraph verification (dataflow-ordered, cache-aware)")
    print("       TRTPartitioner.split(nodes_per_subgraph=200)")
    print("         → split/subgraph_*.onnx")
    print("       Leaf threshold: 20 nodes (smaller → stop splitting)")
    print("       For each subgraph:")
    print("         → check verify_cache.json (skip if fp matches)")
    print("         → TRTBuilder.build(save_debug_tensors=True)")
    print("         → LayerMapper + LayerComparator → comparison.json")
    print("         → if max_abs_diff > threshold && nodes >= 20:")
    print("             queue for deeper split")
    print("         → if max_abs_diff > threshold && nodes < 20:")
    print("             record failed (leaf)")
    print()
    print("Output tree:")
    print(f"  {save_dir}/")
    print(f"    verify_cache.json        # incremental cache (safe to delete)")
    print(f"    verify.log               # full debug log")
    print(f"    ground_truth/            # S1: all intermediate tensor .npy")
    print(f"    full_layers.json         # S2: TRT layer info")
    print(f"    depth_0/                 # recursive verify")
    print(f"      split/subgraph_0.onnx  # TRTPartitioner output")
    print(f"      verify_sub_0/")
    print(f"        inputs/              # input bin files (cleaned after use)")
    print(f"        comparison.json      # per-layer diff report")
    print(f"        sub_layer_info.json  # subgraph TRT layer info (retry only)")
    print(f"    reports/verify_report.json")


def demo_real(onnx_path, pkl_path=None):
    """Run the full pipeline."""
    print("=" * 60)
    print("REAL RUN: TRTVerifier pipeline")
    print("=" * 60)

    input_dict = None
    if pkl_path:
        print(f"Loading ground-truth inputs from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            input_dict = pickle.load(f)
        print(f"  {len(input_dict)} input tensors loaded")

    save_dir = os.path.join(THIS_DIR, "verify_out_v3")

    verifier = TRTVerifier(
        onnx_path=onnx_path,
        save_dir=save_dir,
        threshold=1e-3,
        min_nodes=500,
        memory_budget_mb=512,
        verbose=True,
        input_dict=input_dict,
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
    print(f"Cache hits  : {getattr(verifier, '_cache_hits', 0)}")

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
    parser.add_argument("--pkl", default=None, help="Path to pickle file with ground-truth inputs {name: ndarray}")
    args = parser.parse_args()

    if args.real:
        onnx_path = args.onnx or os.path.join(THIS_DIR, "tiny_test.onnx")
        if not os.path.exists(onnx_path):
            print(f"ONNX model not found: {onnx_path}")
            print("Run create_onnx.py first or specify --onnx <path>")
            sys.exit(1)
        if args.pkl and not os.path.exists(args.pkl):
            print(f"Pickle file not found: {args.pkl}")
            sys.exit(1)
        sys.exit(demo_real(onnx_path, args.pkl))
    else:
        demo_dry_run()
