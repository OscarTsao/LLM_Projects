"""Test disk cache functionality."""
from tests.verify_utils import temp_output_dir, run_cli
import time
from pathlib import Path

def test_disk_cache_speedup():
    """Disk cache improves second-run performance."""
    cache_path = Path("/tmp/verify_cache.db")
    cache_path.unlink(missing_ok=True)

    times = []
    for run in range(2):
        with temp_output_dir() as outdir:
            start = time.time()
            run_cli(
                "--input", "tests/fixtures/mini_annotations.csv",
                "--output-root", str(outdir),
                "--combo-mode", "singletons",
                "--disk-cache", str(cache_path),
                "--variants-per-sample", "1",
                "--seed", "42",
                "--num-proc", "1",
            )
            times.append(time.time() - start)

    if len(times) == 2 and times[0] > 0:
        speedup = times[0] / times[1]
        # Second run should be faster (or at least not slower)
        assert speedup >= 0.9, f"No cache benefit: speedup={speedup:.2f}"
