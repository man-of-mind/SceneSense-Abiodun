"""Minimal read-only NVML binding over ``ctypes`` (no package install).

Only query entry points are bound; nothing here changes clocks, power limits,
compute mode, MPS or scheduling.  Every query returns ``None`` (and the caller
records ``unsupported``) when the driver does not support it.

NVML utilization is the driver's own sampled counter: ``utilization.gpu`` is
the fraction of the last driver sample period (1/6 s to 1 s depending on the
product) in which one or more kernels ran, so a 100-ms reader observes a
moving window, not an instantaneous value.  Per-process utilization comes from
``nvmlDeviceGetProcessUtilization`` samples newer than the previous call.
"""

from __future__ import annotations

import ctypes
from typing import Any, Optional

NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_NOT_FOUND = 6
NVML_CLOCK_GRAPHICS, NVML_CLOCK_SM, NVML_CLOCK_MEM = 0, 1, 2
NVML_TEMPERATURE_GPU = 0


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _ProcessInfo(ctypes.Structure):          # nvmlProcessInfo_v2_t (== v3 layout)
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class _ProcessUtilSample(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("timeStamp", ctypes.c_ulonglong),
                ("smUtil", ctypes.c_uint), ("memUtil", ctypes.c_uint),
                ("encUtil", ctypes.c_uint), ("decUtil", ctypes.c_uint)]


class NvmlUnavailable(RuntimeError):
    pass


class NvmlDeviceV1:
    """One device handle; every method is a pure read."""

    def __init__(self, index: int = 0, library: str = "libnvidia-ml.so.1") -> None:
        try:
            self._lib = ctypes.CDLL(library)
        except OSError as exc:
            raise NvmlUnavailable(f"cannot load {library}: {exc}") from exc
        if self._lib.nvmlInit_v2() != NVML_SUCCESS:
            raise NvmlUnavailable("nvmlInit_v2 failed")
        self._handle = ctypes.c_void_p()
        if self._lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index),
                                                    ctypes.byref(self._handle)) != NVML_SUCCESS:
            raise NvmlUnavailable(f"no NVML device {index}")
        self._last_proc_ts = ctypes.c_ulonglong(0)
        self.unsupported: set[str] = set()

    def _uint(self, fn_name: str, *args: Any) -> Optional[int]:
        if fn_name in self.unsupported:
            return None
        fn = getattr(self._lib, fn_name, None)
        value = ctypes.c_uint()
        if fn is None or fn(self._handle, *args, ctypes.byref(value)) != NVML_SUCCESS:
            self.unsupported.add(fn_name)
            return None
        return int(value.value)

    def name(self) -> Optional[str]:
        buffer = ctypes.create_string_buffer(96)
        if self._lib.nvmlDeviceGetName(self._handle, buffer, ctypes.c_uint(96)) != NVML_SUCCESS:
            return None
        return buffer.value.decode(errors="replace")

    def driver_version(self) -> Optional[str]:
        buffer = ctypes.create_string_buffer(96)
        if self._lib.nvmlSystemGetDriverVersion(buffer, ctypes.c_uint(96)) != NVML_SUCCESS:
            return None
        return buffer.value.decode(errors="replace")

    def device_sample(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        util = _Utilization()
        if self._lib.nvmlDeviceGetUtilizationRates(self._handle,
                                                   ctypes.byref(util)) == NVML_SUCCESS:
            out["util_gpu_pct"], out["util_mem_pct"] = int(util.gpu), int(util.memory)
        else:
            self.unsupported.add("nvmlDeviceGetUtilizationRates")
        memory = _Memory()
        if self._lib.nvmlDeviceGetMemoryInfo(self._handle,
                                             ctypes.byref(memory)) == NVML_SUCCESS:
            out["mem_used_bytes"], out["mem_total_bytes"] = int(memory.used), int(memory.total)
        out["clock_sm_mhz"] = self._uint("nvmlDeviceGetClockInfo", ctypes.c_int(NVML_CLOCK_SM))
        out["clock_mem_mhz"] = self._uint("nvmlDeviceGetClockInfo", ctypes.c_int(NVML_CLOCK_MEM))
        out["power_mw"] = self._uint("nvmlDeviceGetPowerUsage")
        out["temp_c"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_int(NVML_TEMPERATURE_GPU))
        pstate = ctypes.c_int()
        if self._lib.nvmlDeviceGetPerformanceState(self._handle,
                                                   ctypes.byref(pstate)) == NVML_SUCCESS:
            out["pstate"] = int(pstate.value)
        return out

    def _processes(self, fn_name: str) -> Optional[list[dict[str, int]]]:
        if fn_name in self.unsupported:
            return None
        fn = getattr(self._lib, fn_name, None)
        if fn is None:
            self.unsupported.add(fn_name)
            return None
        count = ctypes.c_uint(64)
        array = (_ProcessInfo * 64)()
        if fn(self._handle, ctypes.byref(count), array) != NVML_SUCCESS:
            self.unsupported.add(fn_name)
            return None
        return [{"pid": int(array[i].pid), "used_bytes": int(array[i].usedGpuMemory)}
                for i in range(int(count.value))]

    def process_memory(self) -> dict[str, Any]:
        return {"compute": self._processes("nvmlDeviceGetComputeRunningProcesses_v3"),
                "graphics": self._processes("nvmlDeviceGetGraphicsRunningProcesses_v3")}

    def process_utilization(self) -> Optional[list[dict[str, int]]]:
        """Per-process SM/memory utilization samples newer than the last call."""
        name = "nvmlDeviceGetProcessUtilization"
        if name in self.unsupported:
            return None
        count = ctypes.c_uint(0)
        rc = self._lib.nvmlDeviceGetProcessUtilization(self._handle, None, ctypes.byref(count),
                                                       self._last_proc_ts)
        if rc == NVML_ERROR_NOT_FOUND:
            return []
        if rc not in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE):
            self.unsupported.add(name)
            return None
        size = max(int(count.value), 1)
        array = (_ProcessUtilSample * size)()
        count = ctypes.c_uint(size)
        rc = self._lib.nvmlDeviceGetProcessUtilization(self._handle, array, ctypes.byref(count),
                                                       self._last_proc_ts)
        if rc == NVML_ERROR_NOT_FOUND:
            return []
        if rc != NVML_SUCCESS:
            return None
        rows = [{"pid": int(array[i].pid), "ts_us": int(array[i].timeStamp),
                 "sm_pct": int(array[i].smUtil), "mem_pct": int(array[i].memUtil)}
                for i in range(int(count.value))]
        if rows:
            self._last_proc_ts = ctypes.c_ulonglong(max(r["ts_us"] for r in rows))
        return rows

    def close(self) -> None:
        try:
            self._lib.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
