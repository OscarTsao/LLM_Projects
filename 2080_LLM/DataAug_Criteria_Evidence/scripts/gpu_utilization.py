#!/usr/bin/env python3
"""
Monitor GPU utilization during training with augmentation.

Measures:
- GPU utilization percentage
- GPU memory usage
- GPU temperature
- Average utilization over training steps

Requires: pynvml (install with: pip install nvidia-ml-py)
Falls back gracefully if not available or no GPU.
"""

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import torch

# Try to import pynvml
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("WARNING: nvidia-ml-py not available. Install with: pip install nvidia-ml-py")


class GPUMonitor:
    """Monitor GPU utilization and statistics."""

    def __init__(self, device_id: int = 0):
        """Initialize GPU monitor.

        Args:
            device_id: CUDA device ID to monitor
        """
        self.device_id = device_id
        self.initialized = False
        self.handle = None

        if not PYNVML_AVAILABLE:
            return

        if not torch.cuda.is_available():
            return

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.initialized = True
        except Exception as e:
            print(f"WARNING: Failed to initialize NVML: {e}")

    def get_stats(self) -> dict[str, Any] | None:
        """Get current GPU statistics.

        Returns:
            Dictionary with GPU stats or None if not available
        """
        if not self.initialized:
            return None

        try:
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            # Get temperature
            temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )

            return {
                "gpu_util_percent": util.gpu,
                "memory_util_percent": util.memory,
                "memory_used_mb": mem_info.used / (1024**2),
                "memory_total_mb": mem_info.total / (1024**2),
                "memory_free_mb": mem_info.free / (1024**2),
                "temperature_c": temp,
                "timestamp": time.time(),
            }
        except Exception as e:
            print(f"WARNING: Failed to get GPU stats: {e}")
            return None

    def close(self):
        """Shutdown NVML."""
        if self.initialized:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()


def simulate_training_step(batch_size: int = 32, device: str = "cuda") -> float:
    """Simulate a training step to generate GPU load.

    Args:
        batch_size: Batch size for simulation
        device: Device to run on

    Returns:
        Step time in seconds
    """
    if device == "cpu":
        # CPU simulation
        start = time.perf_counter()
        data = torch.randn(batch_size, 512, 768)
        output = data.mean(dim=1)
        loss = output.mean()
        return time.perf_counter() - start

    # GPU simulation
    start = time.perf_counter()
    data = torch.randn(batch_size, 512, 768, device=device)
    output = data.mean(dim=1)
    loss = output.mean()
    loss.backward()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def monitor_training(
    num_steps: int = 100,
    batch_size: int = 32,
    log_interval: int = 10,
    device_id: int = 0,
) -> dict[str, Any]:
    """Monitor GPU during simulated training.

    Args:
        num_steps: Number of training steps to simulate
        batch_size: Batch size for simulation
        log_interval: Log GPU stats every N steps
        device_id: CUDA device ID

    Returns:
        Dictionary with monitoring results
    """
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("GPU Utilization Monitoring")
    print("=" * 80)
    print(f"Device: {device.upper()}")
    print(f"Steps: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Log interval: {log_interval}")
    print()

    if device == "cpu":
        print("WARNING: No GPU available. Running on CPU.")
        return {
            "device": "cpu",
            "stats": [],
            "summary": {"avg_gpu_util": 0, "avg_memory_util": 0},
        }

    monitor = GPUMonitor(device_id=device_id)

    if not monitor.initialized:
        print("WARNING: GPU monitoring not available.")
        return {
            "device": device,
            "stats": [],
            "summary": {"avg_gpu_util": 0, "avg_memory_util": 0},
        }

    stats_log = []
    step_times = []

    print(
        f"{'Step':>6} {'GPU%':>6} {'Mem%':>6} {'Mem(MB)':>10} {'Temp(C)':>8} {'Step(ms)':>10}"
    )
    print("-" * 80)

    for step in range(num_steps):
        # Simulate training step
        step_time = simulate_training_step(batch_size, device)
        step_times.append(step_time * 1000)  # Convert to ms

        # Log GPU stats at intervals
        if step % log_interval == 0:
            stats = monitor.get_stats()
            if stats:
                stats_log.append(
                    {
                        "step": step,
                        **stats,
                    }
                )

                print(
                    f"{step:>6} "
                    f"{stats['gpu_util_percent']:>6.1f} "
                    f"{stats['memory_util_percent']:>6.1f} "
                    f"{stats['memory_used_mb']:>10.1f} "
                    f"{stats['temperature_c']:>8.1f} "
                    f"{step_time * 1000:>10.2f}"
                )

    monitor.close()

    # Calculate summary statistics
    if stats_log:
        avg_gpu_util = sum(s["gpu_util_percent"] for s in stats_log) / len(stats_log)
        avg_mem_util = sum(s["memory_util_percent"] for s in stats_log) / len(stats_log)
        max_gpu_util = max(s["gpu_util_percent"] for s in stats_log)
        max_temp = max(s["temperature_c"] for s in stats_log)
        avg_step_time = sum(step_times) / len(step_times)

        summary = {
            "avg_gpu_util_percent": avg_gpu_util,
            "avg_memory_util_percent": avg_mem_util,
            "max_gpu_util_percent": max_gpu_util,
            "max_temperature_c": max_temp,
            "avg_step_time_ms": avg_step_time,
            "num_steps": num_steps,
            "num_samples": len(stats_log),
        }
    else:
        summary = {
            "avg_gpu_util_percent": 0,
            "avg_memory_util_percent": 0,
        }

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if stats_log:
        print(f"Average GPU utilization:  {summary['avg_gpu_util_percent']:.1f}%")
        print(f"Maximum GPU utilization:  {summary['max_gpu_util_percent']:.1f}%")
        print(f"Average memory util:      {summary['avg_memory_util_percent']:.1f}%")
        print(f"Maximum temperature:      {summary['max_temperature_c']:.1f}°C")
        print(f"Average step time:        {summary['avg_step_time_ms']:.2f}ms")

        # Check target
        target_util = 85.0
        if summary["avg_gpu_util_percent"] >= target_util:
            print(
                f"\n✅ PASS: GPU utilization ({summary['avg_gpu_util_percent']:.1f}%) "
                f">= target ({target_util}%)"
            )
        else:
            print(
                f"\n⚠️  WARNING: GPU utilization ({summary['avg_gpu_util_percent']:.1f}%) "
                f"< target ({target_util}%)"
            )
    else:
        print("No GPU stats collected.")

    return {
        "device": device,
        "device_id": device_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": stats_log,
        "summary": summary,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor GPU utilization during training"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for simulation (default: 32)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log GPU stats every N steps (default: 10)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device ID (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gpu_utilization_report.json"),
        help="Output JSON file",
    )

    args = parser.parse_args()

    results = monitor_training(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        device_id=args.device_id,
    )

    # Save results
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
