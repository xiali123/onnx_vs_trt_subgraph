import json
import os
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ConvertResult:
    name: str
    bin_file: str
    shape: tuple
    dtype: str
    nbytes: int


@dataclass
class ConvertManifest:
    entries: list = field(default_factory=list)
    input_spec: str = ""
    save_dir: str = ""
    total_bytes: int = 0


class NpyToBin:
    def __init__(self, dump_dir, mapping_file=None):
        self.dump_dir = dump_dir
        self._mapping_path = mapping_file or os.path.join(dump_dir, "name_mapping.json")
        self._output_dir = dump_dir
        self._spec_sep = ","

        with open(self._mapping_path) as f:
            self._mapping = json.load(f)

    def set_output_dir(self, path):
        os.makedirs(path, exist_ok=True)
        self._output_dir = path
        return self

    def convert(self):
        entries = []
        total_bytes = 0

        for orig_name, npy_file in self._mapping.items():
            npy_path = os.path.join(self.dump_dir, npy_file)
            arr = np.load(npy_path)

            bin_file = os.path.splitext(npy_file)[0] + ".bin"
            bin_path = os.path.join(self._output_dir, bin_file)
            arr.tofile(bin_path)

            nbytes = arr.nbytes
            total_bytes += nbytes

            entries.append(ConvertResult(
                name=orig_name,
                bin_file=bin_file,
                shape=arr.shape,
                dtype=str(arr.dtype),
                nbytes=nbytes,
            ))

        spec_parts = []
        for e in entries:
            spec_parts.append(f"{e.name}:{e.bin_file}")

        self._manifest = ConvertManifest(
            entries=entries,
            input_spec=self._spec_sep.join(spec_parts),
            save_dir=self._output_dir,
            total_bytes=total_bytes,
        )
        return self._manifest

    @property
    def manifest(self):
        return getattr(self, "_manifest", None)

    @property
    def input_spec(self):
        if self._manifest is None:
            return ""
        return self._manifest.input_spec

    def save_manifest(self, path=None):
        if self._manifest is None:
            raise RuntimeError("Call convert() first")
        path = path or os.path.join(self._output_dir, "conversion_manifest.json")
        data = {
            "save_dir": self._manifest.save_dir,
            "total_bytes": self._manifest.total_bytes,
            "input_spec": self._manifest.input_spec,
            "entries": [
                {
                    "name": e.name,
                    "bin_file": e.bin_file,
                    "shape": list(e.shape),
                    "dtype": e.dtype,
                    "nbytes": e.nbytes,
                }
                for e in self._manifest.entries
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path
