import os
import re
import subprocess
import shutil
from dataclasses import dataclass


def _find_trtexec(trtexec_path=None):
    """Resolve trtexec: explicit path > env var > system PATH."""
    if trtexec_path:
        return trtexec_path
    env_path = os.environ.get("TRTEXEC_PATH", "")
    if env_path:
        return env_path
    system_path = shutil.which("trtexec")
    if system_path:
        return system_path
    return "trtexec"  # let subprocess fail naturally if not found


def _shape_str(shape):
    return "x".join(str(d) for d in shape)


@dataclass
class Result:
    stdout: str
    latency_ms: float = 0.0
    throughput: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p99_ms: float = 0.0


class TRTBuilder:
    def __init__(
        self,
        onnx_path,
        engine_path=None,
        precision="fp16",
        mempoolsize="workspace:2G",
        builder_opt_level=3,
        verbose=True,
        trtexec_path=None,
        mark_debug_tensors=True,
        is_noTF32=True,
        strongly_typed=True,
        max_aux_streams=0,
        export_profile=None,
        export_layer_info=None,
        profiling_verbosity="detailed",
        dump_profile=True,
        static_plugins=None,
        separate_profile_run=True,
    ):
        self.onnx_path = onnx_path
        self.engine_path = engine_path or os.path.splitext(onnx_path)[0] + ".engine"
        self.precision = precision
        self.mempoolsize = mempoolsize
        self.builder_opt_level = builder_opt_level
        self.verbose = verbose
        self._trtexec_path = trtexec_path
        self.mark_debug_tensors = mark_debug_tensors
        self.is_noTF32 = is_noTF32
        self.strongly_typed = strongly_typed
        self.max_aux_streams = max_aux_streams
        self.export_profile = export_profile
        self.export_layer_info = export_layer_info
        self.profiling_verbosity = profiling_verbosity
        self.dump_profile = dump_profile
        self.static_plugins = static_plugins
        self.separate_profile_run = separate_profile_run
        self.working_dir = None

        self._profiles = []
        self._calib_dir = None
        self._timing_cache = None

    # ── config ──

    def add_profile(self, name, min, opt, max):
        self._profiles.append((name, min, opt, max))
        return self

    def set_calibration(self, calib_dir):
        self._calib_dir = calib_dir
        return self

    def set_timing_cache(self, path):
        self._timing_cache = path
        return self

    def set_working_dir(self, path):
        self.working_dir = path
        return self

    # ── build ──

    @property
    def _trtexec(self):
        return _find_trtexec(self._trtexec_path)

    def _base_args(self, load_engine=False):
        args = [self._trtexec]
        if load_engine:
            args += ["--loadEngine=" + self.engine_path]
        else:
            args += ["--onnx=" + self.onnx_path, "--saveEngine=" + self.engine_path]
        return args

    def _build_args(self):
        args = self._base_args(load_engine=False)

        # precision
        if self.precision == "fp16":
            args.append("--fp16")
        elif self.precision == "int8":
            args.append("--int8")

        args += [
            "--memPoolSize=" + self.mempoolsize,
            "--builderOptimizationLevel=" + str(self.builder_opt_level),
        ]

        # dynamic shapes
        if self._profiles:
            mins, opts, maxs = [], [], []
            for name, mn, opt, mx in self._profiles:
                mins.append(f"{name}:{_shape_str(mn)}")
                opts.append(f"{name}:{_shape_str(opt)}")
                maxs.append(f"{name}:{_shape_str(mx)}")
            args += [
                "--minShapes=" + ",".join(mins),
                "--optShapes=" + ",".join(opts),
                "--maxShapes=" + ",".join(maxs),
            ]

        if self._calib_dir:
            args.append("--calib=" + self._calib_dir)
        if self._timing_cache:
            args.append("--timingCacheFile=" + self._timing_cache)
        if self.mark_debug_tensors:
            args.append("--markUnfusedTensorsAsDebugTensors")
        if self.is_noTF32:
            args.append("--noTF32")
        if self.strongly_typed:
            args.append("--stronglyTyped")
        if self.max_aux_streams is not None:
            args.append("--maxAuxStreams=" + str(self.max_aux_streams))
        if self.export_profile:
            args.append("--exportProfile=" + self.export_profile)
        if self.export_layer_info:
            args.append("--exportLayerInfo=" + self.export_layer_info)
        if self.profiling_verbosity:
            args.append("--profilingVerbosity=" + self.profiling_verbosity)
        if self.dump_profile:
            args.append("--dumpProfile")
        if self.static_plugins:
            plugins = self.static_plugins
            if isinstance(plugins, (list, tuple)):
                plugins = ",".join(plugins)
            args.append("--staticPlugins=" + plugins)
        if self.separate_profile_run:
            args.append("--separateProfileRun")
        if self.verbose:
            args.append("--verbose")

        return args

    def build(self, load_inputs=None, iterations=100,
              save_debug_tensors=False, export_output=None):
        """Build engine + run inference (uses random inputs if load_inputs=None)."""
        args = self._build_args()
        args.append("--iterations=" + str(iterations))

        if load_inputs:
            specs = ",".join(f"{k}:{v}" for k, v in load_inputs.items())
            args.append("--loadInputs=" + specs)
        if save_debug_tensors:
            args.append("--saveAllDebugTensors")
        if export_output:
            args.append("--exportOutput=" + export_output)

        if self.verbose:
            print("[trtexec]", " ".join(args))
        r = subprocess.run(args, capture_output=True, text=True, cwd=self.working_dir)
        if r.returncode != 0:
            raise RuntimeError(f"trtexec build failed:\n{r.stderr}")
        return _parse_latency(r.stdout)

    # ── run ──

    def run(self, load_inputs=None, iterations=100,
            save_debug_tensors=False, export_output=None):
        """Load existing engine + run inference (uses random inputs if load_inputs=None)."""
        args = self._base_args(load_engine=True)

        if load_inputs:
            specs = ",".join(f"{k}:{v}" for k, v in load_inputs.items())
            args.append("--loadInputs=" + specs)
        args.append("--iterations=" + str(iterations))
        if save_debug_tensors:
            args.append("--saveAllDebugTensors")
        if export_output:
            args.append("--exportOutput=" + export_output)

        if self.verbose:
            print("[trtexec]", " ".join(args))
        r = subprocess.run(args, capture_output=True, text=True, cwd=self.working_dir)
        if r.returncode != 0:
            raise RuntimeError(f"trtexec run failed:\n{r.stderr}")
        return _parse_latency(r.stdout)


# ── result parsing ──

_LATENCY_RE = re.compile(r"mean\s*=\s*([\d.]+)\s*ms")
_THROUGHPUT_RE = re.compile(r"throughput\s*[:=]\s*([\d.]+)\s*qps")
_PERCENTILE_RE = re.compile(r"P(\d+)\s*=\s*([\d.]+)\s*ms")


def _parse_latency(stdout):
    result = Result(stdout=stdout)

    m = _LATENCY_RE.search(stdout)
    if m:
        result.latency_ms = float(m.group(1))

    m = _THROUGHPUT_RE.search(stdout)
    if m:
        result.throughput = float(m.group(1))

    for m in _PERCENTILE_RE.finditer(stdout):
        p = int(m.group(1))
        val = float(m.group(2))
        if p == 50:
            result.p50_ms = val
        elif p == 90:
            result.p90_ms = val
        elif p == 99:
            result.p99_ms = val

    return result
