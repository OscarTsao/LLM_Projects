from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple


def _read_proc_stat() -> Tuple[int, int]:
    """Return (idle, total) jiffies from /proc/stat for aggregate CPU.
    If not available, returns (0, 0)."""
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            line = f.readline()
        if not line.startswith("cpu "):
            return 0, 0
        parts = line.split()
        # cpu user nice system idle iowait irq softirq steal guest guest_nice
        values = list(map(int, parts[1:]))
        idle = values[3] + values[4]  # idle + iowait
        total = sum(values[:8])  # up to steal
        return idle, total
    except Exception:
        return 0, 0


def _read_meminfo() -> Tuple[float, float]:
    """Return (used_mb, total_mb) using /proc/meminfo."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val_kb = int(parts[1])
                    info[key] = val_kb
        total_kb = info.get("MemTotal", 0)
        free_kb = info.get("MemFree", 0) + info.get("Buffers", 0) + info.get("Cached", 0)
        used_kb = max(0, total_kb - free_kb)
        return used_kb / 1024.0, total_kb / 1024.0
    except Exception:
        return 0.0, 0.0


def _read_gpu() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (util_pct, mem_used_mb, mem_total_mb) for GPU 0 using nvidia-smi.
    If nvidia-smi is not available, returns (None, None, None)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # Take first GPU line
        line = out.splitlines()[0]
        util_s, used_s, total_s = [s.strip() for s in line.split(",")]
        return float(util_s), float(used_s), float(total_s)
    except Exception:
        return None, None, None


@dataclass
class ResourceSnapshot:
    cpu_idle: int
    cpu_total: int


class ResourceMonitor:
    """Lightweight per-epoch resource monitor.

    Usage:
      monitor = ResourceMonitor()
      monitor.begin_epoch()
      ... train & validate ...
      monitor.log_epoch_end(epoch)
    """

    def __init__(self) -> None:
        self._last_cpu: Optional[ResourceSnapshot] = None

    def begin_epoch(self) -> None:
        idle, total = _read_proc_stat()
        self._last_cpu = ResourceSnapshot(idle=idle, cpu_total=total)

    def _cpu_utilization_pct(self) -> Optional[float]:
        prev = self._last_cpu
        if prev is None:
            return None
        idle, total = _read_proc_stat()
        delta_total = max(1, total - prev.cpu_total)
        delta_idle = max(0, idle - prev.cpu_idle)
        util = 100.0 * (1.0 - (delta_idle / float(delta_total)))
        return max(0.0, min(100.0, util))

    def collect(self) -> dict[str, Any]:
        cpu_util = self._cpu_utilization_pct()
        ram_used_mb, ram_total_mb = _read_meminfo()
        gpu_util, gpu_mem_used_mb, gpu_mem_total_mb = _read_gpu()
        metrics: dict[str, Any] = {
            "cpu.util_pct": cpu_util if cpu_util is not None else -1.0,
            "ram.used_mb": ram_used_mb,
            "ram.total_mb": ram_total_mb,
        }
        if gpu_util is not None:
            metrics.update(
                {
                    "gpu.util_pct": gpu_util,
                    "gpu.vram_used_mb": gpu_mem_used_mb,
                    "gpu.vram_total_mb": gpu_mem_total_mb,
                }
            )
        return metrics

    def log_epoch_end(self, epoch: int) -> None:
        """Log current resource metrics to MLflow at the end of an epoch."""
        try:
            import mlflow  # lazy import

            metrics = self.collect()
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v), step=epoch)
        except Exception:
            # Best-effort; never break training due to monitoring
            pass

