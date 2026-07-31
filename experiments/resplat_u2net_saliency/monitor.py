from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

import psutil


GPU_QUERY = (
    "utilization.gpu,memory.used,power.draw,temperature.gpu"
)


@dataclass
class HardwareSample:
    timestamp: float
    cpu_util: float
    ram_used_mb: float
    process_rss_mb: float
    disk_read_mb: float
    disk_write_mb: float
    gpu_util: float | None = None
    gpu_mem_mb: float | None = None
    gpu_power_w: float | None = None
    gpu_temp_c: float | None = None


def _query_nvidia_smi(gpu_index: int = 0) -> dict[str, float | None]:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        f"--query-gpu={GPU_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2)
    except Exception:
        return {
            "gpu_util": None,
            "gpu_mem_mb": None,
            "gpu_power_w": None,
            "gpu_temp_c": None,
        }
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        return {
            "gpu_util": None,
            "gpu_mem_mb": None,
            "gpu_power_w": None,
            "gpu_temp_c": None,
        }
    keys = ["gpu_util", "gpu_mem_mb", "gpu_power_w", "gpu_temp_c"]
    out: dict[str, float | None] = {}
    for key, value in zip(keys, parts):
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = None
    return out


class HardwareMonitor:
    def __init__(self, interval_s: float = 0.5, gpu_index: int = 0) -> None:
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.samples: list[HardwareSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self._disk0 = psutil.disk_io_counters()

    def sample_once(self) -> HardwareSample:
        disk = psutil.disk_io_counters()
        gpu = _query_nvidia_smi(self.gpu_index)
        sample = HardwareSample(
            timestamp=time.time(),
            cpu_util=psutil.cpu_percent(interval=None),
            ram_used_mb=psutil.virtual_memory().used / 1024**2,
            process_rss_mb=self._process.memory_info().rss / 1024**2,
            disk_read_mb=(disk.read_bytes - self._disk0.read_bytes) / 1024**2,
            disk_write_mb=(disk.write_bytes - self._disk0.write_bytes) / 1024**2,
            **gpu,
        )
        self.samples.append(sample)
        return sample

    def start(self) -> "HardwareMonitor":
        if self._thread is not None:
            return self
        psutil.cpu_percent(interval=None)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
            self._thread = None
        if not self.samples:
            self.sample_once()
        return self.summary()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, float | None]:
        def mean(name: str) -> float | None:
            vals = [getattr(sample, name) for sample in self.samples]
            vals = [v for v in vals if v is not None]
            return float(sum(vals) / len(vals)) if vals else None

        def peak(name: str) -> float | None:
            vals = [getattr(sample, name) for sample in self.samples]
            vals = [v for v in vals if v is not None]
            return float(max(vals)) if vals else None

        return {
            "gpu_util_mean": mean("gpu_util"),
            "gpu_mem_peak_mb": peak("gpu_mem_mb"),
            "gpu_power_mean_w": mean("gpu_power_w"),
            "gpu_temp_mean_c": mean("gpu_temp_c"),
            "cpu_util_mean": mean("cpu_util"),
            "ram_peak_mb": peak("ram_used_mb"),
            "process_rss_peak_mb": peak("process_rss_mb"),
            "disk_read_mb": peak("disk_read_mb"),
            "disk_write_mb": peak("disk_write_mb"),
        }

    def wandb_payload(self, prefix: str = "hardware") -> dict[str, Any]:
        summary = self.summary()
        return {f"{prefix}/{key}": value for key, value in summary.items() if value is not None}

