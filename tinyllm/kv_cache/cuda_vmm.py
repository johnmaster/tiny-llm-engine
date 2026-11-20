"""CUDA virtual-memory-backed tensors for incrementally committed KV pages.

The CUDA driver reserves one stable virtual range per tensor. Physical memory
is mapped in allocation-granularity-sized chunks as KV blocks are touched.
PyTorch views the reserved range through DLPack; Triton continues to receive
ordinary strided CUDA tensors with stable data pointers.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import logging
from typing import Any


_CUDA_SUCCESS = 0
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_DL_DEVICE_CUDA = 2
_DL_DATA_TYPE_FLOAT = 2
_DL_DATA_TYPE_BFLOAT = 4
logger = logging.getLogger(__name__)


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compression_type", ctypes.c_ubyte),
        ("gpu_direct_rdma_capable", ctypes.c_ubyte),
        ("usage", ctypes.c_ubyte),
        ("reserved", ctypes.c_ubyte * 5),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requested_handle_types", ctypes.c_uint),
        ("location", _CUmemLocation),
        ("win32_handle_metadata", ctypes.c_void_p),
        ("alloc_flags", _CUmemAllocationFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_uint)]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorPointer = ctypes.POINTER(_DLManagedTensor)
_DLPackDeleter = ctypes.CFUNCTYPE(None, _DLManagedTensorPointer)
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLPackDeleter),
]


@dataclass
class _DLPackOwner:
    buffer: "CudaVirtualBuffer"
    managed: _DLManagedTensor
    shape: Any
    strides: Any


_DLPACK_OWNERS: dict[int, _DLPackOwner] = {}


def _dlmanaged_deleter(managed_ptr: _DLManagedTensorPointer) -> None:
    key = ctypes.addressof(managed_ptr.contents)
    owner = _DLPACK_OWNERS.pop(key, None)
    if owner is not None:
        try:
            owner.buffer.close()
        except Exception:
            # Exceptions cannot be propagated through a ctypes DLPack callback.
            pass


_DLPACK_DELETER = _DLPackDeleter(_dlmanaged_deleter)


def _capsule_destructor(capsule: ctypes.py_object) -> None:
    """Release a DLPack tensor if its capsule was never consumed by PyTorch."""
    name = _PY_CAPSULE_GET_NAME(capsule)
    if name != b"dltensor":
        return
    managed_ptr = _PY_CAPSULE_GET_POINTER(capsule, b"dltensor")
    if managed_ptr:
        _dlmanaged_deleter(ctypes.cast(managed_ptr, _DLManagedTensorPointer))


_CapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
_CAPSULE_DESTRUCTOR = _CapsuleDestructor(_capsule_destructor)
_PY_CAPSULE_NEW = ctypes.pythonapi.PyCapsule_New
_PY_CAPSULE_NEW.restype = ctypes.py_object
_PY_CAPSULE_NEW.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_PY_CAPSULE_GET_NAME = ctypes.pythonapi.PyCapsule_GetName
_PY_CAPSULE_GET_NAME.restype = ctypes.c_char_p
_PY_CAPSULE_GET_NAME.argtypes = [ctypes.py_object]
_PY_CAPSULE_GET_POINTER = ctypes.pythonapi.PyCapsule_GetPointer
_PY_CAPSULE_GET_POINTER.restype = ctypes.c_void_p
_PY_CAPSULE_GET_POINTER.argtypes = [ctypes.py_object, ctypes.c_char_p]


class _CudaDriver:
    def __init__(self):
        library = ctypes.util.find_library("cuda") or "libcuda.so.1"
        try:
            self.lib = ctypes.CDLL(library)
        except OSError as exc:
            raise RuntimeError(
                "CUDA virtual memory requires the NVIDIA CUDA driver library (libcuda)"
            ) from exc
        self._bind("cuInit", [ctypes.c_uint])
        self._bind("cuMemGetAllocationGranularity", [
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(_CUmemAllocationProp),
            ctypes.c_int,
        ])
        self._bind("cuMemAddressReserve", [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_uint64, ctypes.c_ulonglong,
        ])
        self._bind("cuMemAddressFree", [ctypes.c_uint64, ctypes.c_size_t])
        self._bind("cuMemCreate", [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
            ctypes.POINTER(_CUmemAllocationProp), ctypes.c_ulonglong,
        ])
        self._bind("cuMemMap", [
            ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_uint64, ctypes.c_ulonglong,
        ])
        self._bind("cuMemSetAccess", [
            ctypes.c_uint64, ctypes.c_size_t,
            ctypes.POINTER(_CUmemAccessDesc), ctypes.c_size_t,
        ])
        self._bind("cuMemUnmap", [ctypes.c_uint64, ctypes.c_size_t])
        self._bind("cuMemRelease", [ctypes.c_uint64])
        self._bind("cuGetErrorName", [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)])
        self._check(self.lib.cuInit(0), "cuInit")

    def _bind(self, name: str, argtypes: list[Any]) -> None:
        function = getattr(self.lib, name)
        function.restype = ctypes.c_int
        function.argtypes = argtypes

    def _check(self, result: int, operation: str) -> None:
        if result == _CUDA_SUCCESS:
            return
        error_name = ctypes.c_char_p()
        if hasattr(self.lib, "cuGetErrorName"):
            self.lib.cuGetErrorName(result, ctypes.byref(error_name))
        label = error_name.value.decode() if error_name.value else f"CUDA error {result}"
        raise RuntimeError(f"{operation} failed: {label}")


_DRIVER: _CudaDriver | None = None


def _driver() -> _CudaDriver:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = _CudaDriver()
    return _DRIVER


def _allocation_prop(device_id: int) -> _CUmemAllocationProp:
    prop = _CUmemAllocationProp()
    prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requested_handle_types = 0
    prop.location = _CUmemLocation(_CU_MEM_LOCATION_TYPE_DEVICE, device_id)
    return prop


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class CudaVirtualBuffer:
    """Stable CUDA virtual range whose physical backing grows by KV blocks."""

    def __init__(self, *, torch: Any, shape: tuple[int, ...], dtype: Any,
                 device: Any):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA VMM was requested but CUDA is unavailable")
        torch.cuda.init()
        self.torch = torch
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.device_id = (self.device.index if self.device.index is not None
                          else torch.cuda.current_device())
        torch.cuda.set_device(self.device_id)
        self.element_size = torch.empty((), dtype=dtype).element_size()
        self.logical_bytes = self.element_size
        for dimension in shape:
            self.logical_bytes *= dimension
        self._driver = _driver()
        prop = _allocation_prop(self.device_id)
        granularity = ctypes.c_size_t()
        self._driver._check(
            self._driver.lib.cuMemGetAllocationGranularity(
                ctypes.byref(granularity), ctypes.byref(prop),
                _CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "cuMemGetAllocationGranularity",
        )
        self.granularity = granularity.value
        self.reserved_bytes = _align_up(self.logical_bytes, self.granularity)
        address = ctypes.c_uint64()
        self._driver._check(
            self._driver.lib.cuMemAddressReserve(
                ctypes.byref(address), self.reserved_bytes, self.granularity, 0, 0
            ),
            "cuMemAddressReserve",
        )
        self.address = address.value
        self.committed_bytes = 0
        self._mapped: list[tuple[int, int, int]] = []
        self._closed = False

    @property
    def committed_blocks(self) -> int:
        block_bytes = self.logical_bytes // self.shape[0]
        return min(self.shape[0], (self.committed_bytes + block_bytes - 1) // block_bytes)

    def commit_blocks(self, num_blocks: int) -> None:
        if self._closed:
            raise RuntimeError("cannot commit a closed CUDA virtual buffer")
        self.torch.cuda.set_device(self.device_id)
        if num_blocks < 0 or num_blocks > self.shape[0]:
            raise ValueError("requested block count is outside the reserved CUDA range")
        target = _align_up(self.logical_bytes * num_blocks // self.shape[0], self.granularity)
        if target > self.reserved_bytes:
            target = self.reserved_bytes
        if target <= self.committed_bytes:
            return

        prop = _allocation_prop(self.device_id)
        access = _CUmemAccessDesc(
            _CUmemLocation(_CU_MEM_LOCATION_TYPE_DEVICE, self.device_id),
            _CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        )
        offset = self.committed_bytes
        while offset < target:
            size = target - offset
            handle = ctypes.c_uint64()
            self._driver._check(
                self._driver.lib.cuMemCreate(
                    ctypes.byref(handle), size, ctypes.byref(prop), 0
                ),
                "cuMemCreate",
            )
            address = self.address + offset
            try:
                self._driver._check(
                    self._driver.lib.cuMemMap(address, size, 0, handle.value, 0),
                    "cuMemMap",
                )
                self._driver._check(
                    self._driver.lib.cuMemSetAccess(
                        address, size, ctypes.byref(access), 1
                    ),
                    "cuMemSetAccess",
                )
            except Exception:
                self._driver.lib.cuMemUnmap(address, size)
                self._driver.lib.cuMemRelease(handle.value)
                raise
            self._mapped.append((address, size, handle.value))
            offset += size
        self.committed_bytes = target

    def as_torch_tensor(self) -> Any:
        dtype_name = str(self.dtype)
        if dtype_name == "torch.bfloat16":
            code, bits = _DL_DATA_TYPE_BFLOAT, 16
        elif dtype_name == "torch.float16":
            code, bits = _DL_DATA_TYPE_FLOAT, 16
        elif dtype_name == "torch.float32":
            code, bits = _DL_DATA_TYPE_FLOAT, 32
        else:
            raise TypeError(f"CUDA VMM does not support KV dtype {dtype_name}")

        shape = (ctypes.c_int64 * len(self.shape))(*self.shape)
        strides_values = [1] * len(self.shape)
        for index in range(len(self.shape) - 2, -1, -1):
            strides_values[index] = strides_values[index + 1] * self.shape[index + 1]
        strides = (ctypes.c_int64 * len(self.shape))(*strides_values)
        managed = _DLManagedTensor()
        managed.dl_tensor = _DLTensor(
            ctypes.c_void_p(self.address),
            _DLDevice(_DL_DEVICE_CUDA, self.device_id),
            len(self.shape),
            _DLDataType(code, bits, 1),
            ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64)),
            ctypes.cast(strides, ctypes.POINTER(ctypes.c_int64)),
            0,
        )
        managed.manager_ctx = ctypes.c_void_p(ctypes.addressof(managed))
        managed.deleter = _DLPACK_DELETER
        key = ctypes.addressof(managed)
        _DLPACK_OWNERS[key] = _DLPackOwner(self, managed, shape, strides)
        capsule = _PY_CAPSULE_NEW(
            ctypes.cast(ctypes.pointer(managed), ctypes.c_void_p),
            b"dltensor",
            ctypes.cast(_CAPSULE_DESTRUCTOR, ctypes.c_void_p),
        )
        try:
            return self.torch.utils.dlpack.from_dlpack(capsule)
        except Exception:
            _dlmanaged_deleter(ctypes.pointer(managed))
            raise

    def close(self, *, synchronize: bool = True) -> None:
        if self._closed:
            return
        self.torch.cuda.set_device(self.device_id)
        if synchronize:
            self.torch.cuda.synchronize(self.device_id)
        self._closed = True
        for address, size, handle in reversed(self._mapped):
            self._driver._check(self._driver.lib.cuMemUnmap(address, size), "cuMemUnmap")
            self._driver._check(self._driver.lib.cuMemRelease(handle), "cuMemRelease")
        self._mapped.clear()
        self.committed_bytes = 0
        self._driver._check(
            self._driver.lib.cuMemAddressFree(self.address, self.reserved_bytes),
            "cuMemAddressFree",
        )


def probe_cuda_vmm(torch: Any, device: Any, dtype: Any) -> tuple[bool, str | None]:
    """Exercise driver VMM and DLPack before an engine accepts requests."""
    buffer: CudaVirtualBuffer | None = None
    tensor: Any = None
    try:
        buffer = CudaVirtualBuffer(
            torch=torch, shape=(1,), dtype=dtype, device=device
        )
        buffer.commit_blocks(1)
        tensor = buffer.as_torch_tensor()
        tensor.zero_()
        torch.cuda.synchronize(device)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if tensor is not None:
            del tensor
        if buffer is not None:
            try:
                buffer.close()
            except Exception as exc:
                logger.debug("Failed to clean up CUDA VMM probe: %s", exc)
