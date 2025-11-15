"""System resource monitoring and logging for MLflow."""

import time
from typing import Optional

import mlflow
import torch

# Try to import psutil (optional dependency)
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import pynvml for detailed GPU metrics (optional)
try:
    import pynvml

    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, Exception):
    PYNVML_AVAILABLE = False


class SystemMetricsLogger:
    """
    Comprehensive system metrics logger for MLflow.

    Tracks:
    - CPU usage and memory
    - GPU utilization, memory, temperature (if available)
    - Disk I/O (if available)
    - Network I/O (if available)
    """

    def __init__(
        self,
        log_cpu: bool = True,
        log_gpu: bool = True,
        log_disk: bool = False,
        log_network: bool = False,
        gpu_device: int = 0,
    ):
        """
        Initialize system metrics logger.

        Args:
            log_cpu: Enable CPU/memory logging
            log_gpu: Enable GPU logging
            log_disk: Enable disk I/O logging
            log_network: Enable network I/O logging
            gpu_device: GPU device index to monitor
        """
        self.log_cpu = log_cpu and PSUTIL_AVAILABLE
        self.log_gpu = log_gpu and torch.cuda.is_available()
        self.log_disk = log_disk and PSUTIL_AVAILABLE
        self.log_network = log_network and PSUTIL_AVAILABLE
        self.gpu_device = gpu_device

        # Get GPU handle if pynvml available
        self.gpu_handle = None
        if self.log_gpu and PYNVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device)
            except Exception:
                pass

        # Baseline metrics for delta calculations
        self.baseline_disk_io = None
        self.baseline_net_io = None
        self.last_log_time = None

        if PSUTIL_AVAILABLE:
            if self.log_disk:
                self.baseline_disk_io = psutil.disk_io_counters()
            if self.log_network:
                self.baseline_net_io = psutil.net_io_counters()

    def log_metrics(self, step: Optional[int] = None, prefix: str = "system"):
        """
        Log all enabled system metrics to MLflow.

        Args:
            step: Optional step number for MLflow logging
            prefix: Prefix for metric names (default: "system")
        """
        metrics = {}
        current_time = time.time()

        # CPU and memory metrics
        if self.log_cpu:
            metrics[f"{prefix}/cpu_percent"] = psutil.cpu_percent(interval=0.1)
            metrics[f"{prefix}/cpu_count"] = psutil.cpu_count()

            mem = psutil.virtual_memory()
            metrics[f"{prefix}/memory_percent"] = mem.percent
            metrics[f"{prefix}/memory_available_gb"] = mem.available / 1e9
            metrics[f"{prefix}/memory_used_gb"] = mem.used / 1e9
            metrics[f"{prefix}/memory_total_gb"] = mem.total / 1e9

        # GPU metrics
        if self.log_gpu:
            metrics[f"{prefix}/gpu_memory_allocated_gb"] = (
                torch.cuda.memory_allocated(self.gpu_device) / 1e9
            )
            metrics[f"{prefix}/gpu_memory_reserved_gb"] = (
                torch.cuda.memory_reserved(self.gpu_device) / 1e9
            )
            metrics[f"{prefix}/gpu_memory_cached_gb"] = (
                torch.cuda.memory_reserved(self.gpu_device) / 1e9
            )

            # Detailed GPU metrics via pynvml
            if self.gpu_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    metrics[f"{prefix}/gpu_utilization_percent"] = util.gpu
                    metrics[f"{prefix}/gpu_memory_utilization_percent"] = util.memory

                    temp = pynvml.nvmlDeviceGetTemperature(
                        self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    metrics[f"{prefix}/gpu_temperature_celsius"] = temp

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    metrics[f"{prefix}/gpu_memory_used_pynvml_gb"] = mem_info.used / 1e9
                    metrics[f"{prefix}/gpu_memory_free_gb"] = mem_info.free / 1e9
                except Exception:
                    pass

        # Disk I/O metrics (delta since baseline)
        if self.log_disk and self.baseline_disk_io is not None:
            current_disk = psutil.disk_io_counters()
            metrics[f"{prefix}/disk_read_mb"] = (
                current_disk.read_bytes - self.baseline_disk_io.read_bytes
            ) / 1e6
            metrics[f"{prefix}/disk_write_mb"] = (
                current_disk.write_bytes - self.baseline_disk_io.write_bytes
            ) / 1e6

        # Network I/O metrics (delta since baseline)
        if self.log_network and self.baseline_net_io is not None:
            current_net = psutil.net_io_counters()
            metrics[f"{prefix}/network_sent_mb"] = (
                current_net.bytes_sent - self.baseline_net_io.bytes_sent
            ) / 1e6
            metrics[f"{prefix}/network_recv_mb"] = (
                current_net.bytes_recv - self.baseline_net_io.bytes_recv
            ) / 1e6

        # Log to MLflow
        if metrics and mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)

        self.last_log_time = current_time
        return metrics

    def __del__(self):
        """Cleanup pynvml on deletion."""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
