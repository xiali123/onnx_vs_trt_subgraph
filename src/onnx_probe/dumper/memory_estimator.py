import numpy as np
from onnx import TensorProto

_ONNX_DTYPE_TO_NUMPY = {
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.BOOL: np.bool_,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT16: np.int16,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
    TensorProto.BFLOAT16: np.float16,
    TensorProto.FLOAT8E4M3FN: np.float32,
    TensorProto.FLOAT8E5M2: np.float32,
}

_ONNX_DTYPE_SIZE = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.BFLOAT16: 2,
    TensorProto.FLOAT8E4M3FN: 4,
    TensorProto.FLOAT8E5M2: 4,
}


class MemoryEstimator:
    """Static utilities for calculating ONNX tensor memory sizes."""

    @staticmethod
    def dtype_bytes(elem_type):
        """Return element size in bytes for an ONNX data type."""
        size = _ONNX_DTYPE_SIZE.get(elem_type)
        if size is not None:
            return size
        dtype = _ONNX_DTYPE_TO_NUMPY.get(elem_type, np.float32)
        return np.dtype(dtype).itemsize

    @staticmethod
    def num_elements(shape):
        """Number of elements from a shape tuple. Dynamic dims (0 or None) → 1."""
        n = 1
        for d in shape:
            n *= (d if d else 1)
        return max(n, 1)

    @staticmethod
    def tensor_bytes(shape, elem_type):
        """Total bytes for a tensor given its shape and ONNX elem_type."""
        return MemoryEstimator.num_elements(shape) * MemoryEstimator.dtype_bytes(elem_type)

    @staticmethod
    def from_value_info(vi):
        """Memory bytes from a ValueInfoProto (graph input/output/value_info)."""
        ts = vi.type.tensor_type
        shape = [d.dim_value for d in ts.shape.dim]
        return MemoryEstimator.tensor_bytes(shape, ts.elem_type)

    @staticmethod
    def from_tensor_proto(tp):
        """Memory bytes from a TensorProto (initializer)."""
        return MemoryEstimator.tensor_bytes(list(tp.dims), tp.data_type)

    @staticmethod
    def from_onnx_tensor(tensor):
        """Memory bytes from an onnx_graphsurgeon Tensor (has shape + dtype attrs)."""
        shape = list(tensor.shape) if tensor.shape else []
        if hasattr(tensor, 'dtype') and tensor.dtype is not None:
            return MemoryEstimator.num_elements(shape) * np.dtype(tensor.dtype).itemsize
        return 0
