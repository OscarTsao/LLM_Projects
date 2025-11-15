#!/usr/bin/env python3
"""
Generate comprehensive verification reports from test and benchmark results.
Outputs both JSON (verification_summary.json) and Markdown (verification_report.md).
"""

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_environment_info():
    """Gather environment and hardware information."""
    env_info = {
        "python": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.platform(),
        "cpu_cores": None,
        "cuda_available": False,
        "gpu_name": None
    }

    # Get CPU core count
    try:
        import multiprocessing
        env_info["cpu_cores"] = multiprocessing.cpu_count()
    except Exception:
        pass

    # Check CUDA availability
    try:
        import torch
        env_info["cuda_available"] = torch.cuda.is_available()
        if env_info["cuda_available"]:
            try:
                env_info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                env_info["gpu_name"] = "Unknown GPU"
    except ImportError:
        pass

    return env_info

def parse_test_results(test_json_path: Path):
    """Parse pytest JSON report."""
    if not test_json_path.exists():
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "error": 0,
            "xfailed": 0,
            "xpassed": 0,
            "tests": []
        }

    try:
        with open(test_json_path) as f:
            data = json.load(f)

        summary = data.get("summary", {})
        tests = data.get("tests", [])

        return {
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "skipped": summary.get("skipped", 0),
            "error": summary.get("error", 0),
            "xfailed": summary.get("xfailed", 0),
            "xpassed": summary.get("xpassed", 0),
            "tests": tests
        }
    except Exception as e:
        print(f"Warning: Could not parse test results: {e}", file=sys.stderr)
        return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0, "tests": []}

def parse_bench_results(bench_json_path: Path):
    """Parse benchmark results."""
    if not bench_json_path.exists():
        return {}

    try:
        with open(bench_json_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not parse benchmark results: {e}", file=sys.stderr)
        return {}

def categorize_tests(tests):
    """Categorize tests by functionality."""
    categories = {
        "registry": [],
        "cli": [],
        "evidence_only": [],
        "determinism": [],
        "variants": [],
        "combos": [],
        "sharding": [],
        "manifests": [],
        "quality_filter": [],
        "skip_handling": [],
        "gpu_cpu": [],
        "disk_cache": [],
        "no_training": [],
        "linting": [],
        "all_methods": []
    }

    for test in tests:
        nodeid = test.get("nodeid", "")

        if "test_01_registry" in nodeid:
            categories["registry"].append(test)
        elif "test_02_cli" in nodeid:
            categories["cli"].append(test)
        elif "test_03_evidence" in nodeid:
            categories["evidence_only"].append(test)
        elif "test_04_determinism" in nodeid:
            categories["determinism"].append(test)
        elif "test_05_variants" in nodeid:
            categories["variants"].append(test)
        elif "test_06_combos" in nodeid:
            categories["combos"].append(test)
        elif "test_07_sharding" in nodeid:
            categories["sharding"].append(test)
        elif "test_08_manifests" in nodeid:
            categories["manifests"].append(test)
        elif "test_09_quality" in nodeid:
            categories["quality_filter"].append(test)
        elif "test_10_skip" in nodeid:
            categories["skip_handling"].append(test)
        elif "test_11_gpu" in nodeid:
            categories["gpu_cpu"].append(test)
        elif "test_12_disk" in nodeid:
            categories["disk_cache"].append(test)
        elif "test_13_no_training" in nodeid:
            categories["no_training"].append(test)
        elif "test_14_linting" in nodeid:
            categories["linting"].append(test)
        elif "test_15_all_methods" in nodeid:
            categories["all_methods"].append(test)

    return categories

def category_status(category_tests):
    """Get pass/fail status for a category."""
    if not category_tests:
        return "N/A", 0, 0

    passed = sum(1 for t in category_tests if t.get("outcome") == "passed")
    failed = sum(1 for t in category_tests if t.get("outcome") == "failed")
    total = len(category_tests)

    if failed > 0:
        return "FAIL", passed, total
    elif passed == total:
        return "PASS", passed, total
    else:
        return "PARTIAL", passed, total

def generate_summary_json(env_info, test_results, bench_results):
    """Generate verification_summary.json."""

    categories = categorize_tests(test_results.get("tests", []))

    # Analyze specific test categories
    reg_status, _, _ = category_status(categories["registry"])
    det_status, _, _ = category_status(categories["determinism"])
    evi_status, _, _ = category_status(categories["evidence_only"])
    qf_status, _, _ = category_status(categories["quality_filter"])
    shard_status, _, _ = category_status(categories["sharding"])
    manifest_status, _, _ = category_status(categories["manifests"])

    # Extract benchmark metrics
    cpu_throughput = bench_results.get("cpu_methods", {}).get("throughput_rows_per_sec", 0)
    gpu_throughput = bench_results.get("gpu_methods", {}).get("throughput_rows_per_sec", 0)
    cache_speedup = bench_results.get("disk_cache", {}).get("speedup_factor", 0)
    mp_speedup = bench_results.get("multiprocessing", {}).get("speedup_factor", 0)

    # Determine overall status
    failed = test_results.get("failed", 0)
    error = test_results.get("error", 0)
    bench_failed = (
        not bench_results.get("cpu_methods", {}).get("success", False) or
        not bench_results.get("disk_cache", {}).get("success", False) or
        not bench_results.get("multiprocessing", {}).get("success", False)
    )

    if failed > 0 or error > 0 or bench_failed:
        overall = "FAIL"
    else:
        overall = "PASS"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "env": env_info,
        "registry": {
            "expected": 28,
            "status": reg_status
        },
        "tests": {
            "total": test_results.get("total", 0),
            "passed": test_results.get("passed", 0),
            "failed": test_results.get("failed", 0),
            "skipped": test_results.get("skipped", 0),
            "error": test_results.get("error", 0),
            "xfailed": test_results.get("xfailed", 0)
        },
        "determinism": {
            "status": det_status
        },
        "evidence_only": {
            "status": evi_status
        },
        "quality_filter": {
            "status": qf_status
        },
        "sharding": {
            "status": shard_status
        },
        "manifest_integrity": {
            "status": manifest_status
        },
        "performance": {
            "cpu_rows_per_sec": cpu_throughput,
            "gpu_rows_per_sec": gpu_throughput,
            "disk_cache_speedup": cache_speedup,
            "multiprocessing_speedup": mp_speedup
        },
        "overall": overall
    }

    return summary

def generate_markdown_report(env_info, test_results, bench_results, categories):
    """Generate verification_report.md."""

    report = f"""# DSM-5 Data Augmentation Verification Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Executive Summary

"""

    # Overall status
    failed = test_results.get("failed", 0)
    passed = test_results.get("passed", 0)
    total = test_results.get("total", 0)

    if failed > 0:
        report += f"**Status:** FAIL ({failed}/{total} tests failed)\n\n"
    else:
        report += f"**Status:** PASS (All {passed} tests passed)\n\n"

    # Environment section
    report += """---

## Environment & Hardware

| Component | Details |
|-----------|---------|
"""
    report += f"| Python Version | {env_info['python']} |\n"
    report += f"| Operating System | {env_info['os']} ({env_info['os_version']}) |\n"
    report += f"| CPU Cores | {env_info['cpu_cores'] or 'Unknown'} |\n"
    report += f"| CUDA Available | {env_info['cuda_available']} |\n"
    if env_info['cuda_available']:
        report += f"| GPU | {env_info['gpu_name'] or 'Unknown'} |\n"

    # Test results section
    report += """
---

## Test Results

### Summary by Category

| Category | Status | Passed | Total |
|----------|--------|--------|-------|
"""

    category_names = {
        "registry": "Method Registry",
        "cli": "CLI Interface",
        "evidence_only": "Evidence-Only Mode",
        "determinism": "Deterministic Output",
        "variants": "Variants Per Sample",
        "combos": "Method Combinations",
        "sharding": "Sharding Support",
        "manifests": "Manifest Integrity",
        "quality_filter": "Quality Filtering",
        "skip_handling": "Skip Handling",
        "gpu_cpu": "GPU/CPU Execution",
        "disk_cache": "Disk Caching",
        "no_training": "No Training Code",
        "linting": "Code Quality",
        "all_methods": "All Methods Test"
    }

    for cat_key, cat_name in category_names.items():
        cat_tests = categories.get(cat_key, [])
        status, passed_count, total_count = category_status(cat_tests)

        if status == "PASS":
            icon = "✅"
        elif status == "FAIL":
            icon = "❌"
        elif status == "PARTIAL":
            icon = "⚠️"
        else:
            icon = "-"

        report += f"| {cat_name} | {icon} {status} | {passed_count} | {total_count} |\n"

    # Performance benchmarks
    report += """
---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
"""

    cpu_methods = bench_results.get("cpu_methods", {})
    gpu_methods = bench_results.get("gpu_methods", {})
    disk_cache = bench_results.get("disk_cache", {})
    multiproc = bench_results.get("multiprocessing", {})

    if cpu_methods.get("success"):
        report += f"| CPU Throughput | {cpu_methods['throughput_rows_per_sec']:.1f} rows/sec |\n"
        report += f"| CPU Duration | {cpu_methods['duration_sec']:.2f}s ({cpu_methods['total_rows']} rows) |\n"
    else:
        report += "| CPU Throughput | FAILED |\n"

    if gpu_methods.get("cuda_available"):
        if gpu_methods.get("success"):
            report += f"| GPU Throughput | {gpu_methods['throughput_rows_per_sec']:.1f} rows/sec |\n"
            report += f"| GPU Duration | {gpu_methods['duration_sec']:.2f}s ({gpu_methods['total_rows']} rows) |\n"
        else:
            report += "| GPU Throughput | FAILED |\n"
    else:
        report += "| GPU Throughput | SKIPPED (CUDA not available) |\n"

    if disk_cache.get("success"):
        report += f"| Disk Cache Speedup | {disk_cache['speedup_factor']:.2f}x |\n"
        report += f"| Cache Cold Run | {disk_cache['first_run_sec']:.2f}s |\n"
        report += f"| Cache Warm Run | {disk_cache['second_run_sec']:.2f}s |\n"
    else:
        report += "| Disk Cache Speedup | FAILED |\n"

    if multiproc.get("success"):
        report += f"| Multiprocessing Speedup | {multiproc['speedup_factor']:.2f}x |\n"
        report += f"| Single Process | {multiproc['num_proc_1_sec']:.2f}s |\n"
        report += f"| 4 Processes | {multiproc['num_proc_4_sec']:.2f}s |\n"
    else:
        report += "| Multiprocessing Speedup | FAILED |\n"

    # Failed tests section
    failed_tests = [t for t in test_results.get("tests", []) if t.get("outcome") == "failed"]

    if failed_tests:
        report += """
---

## Failed Tests

"""
        for test in failed_tests:
            nodeid = test.get("nodeid", "unknown")
            call = test.get("call", {})
            longrepr = call.get("longrepr", "No details available")

            report += f"### {nodeid}\n\n"
            report += "```\n"
            report += str(longrepr)[:500]  # Truncate long errors
            report += "\n```\n\n"

    # Recommendations section
    report += """---

## Recommendations

"""

    if failed > 0:
        report += f"- ❌ {failed} test(s) failed. Review failures above and check test_results.json for full details.\n"
    else:
        report += "- ✅ All tests passed successfully.\n"

    if not cpu_methods.get("success", False):
        report += "- ⚠️ CPU benchmarks failed. Check that all required packages are installed.\n"

    if gpu_methods.get("cuda_available") and not gpu_methods.get("success"):
        report += "- ⚠️ GPU benchmarks failed despite CUDA being available. Check GPU setup.\n"

    if not disk_cache.get("success", False):
        report += "- ⚠️ Disk cache benchmarks failed. Check disk cache configuration.\n"

    if not multiproc.get("success", False):
        report += "- ⚠️ Multiprocessing benchmarks failed. Check multiprocessing setup.\n"

    if failed == 0 and cpu_methods.get("success") and disk_cache.get("success") and multiproc.get("success"):
        report += "- ✅ Verification complete. All systems operational.\n"

    report += "\n---\n\n"
    report += "*Report generated by tools/verify/generate_report.py*\n"

    return report

def main():
    """Generate verification reports."""
    print("=" * 70)
    print("Generating Verification Reports")
    print("=" * 70)

    # Gather data
    print("\n[1/4] Gathering environment information...")
    env_info = get_environment_info()

    print("[2/4] Parsing test results...")
    test_json_path = PROJECT_ROOT / "test_results.json"
    test_results = parse_test_results(test_json_path)

    print("[3/4] Parsing benchmark results...")
    bench_json_path = PROJECT_ROOT / "tools" / "verify" / "bench_results.json"
    bench_results = parse_bench_results(bench_json_path)

    # Generate reports
    print("[4/4] Generating reports...")

    categories = categorize_tests(test_results.get("tests", []))

    summary = generate_summary_json(env_info, test_results, bench_results)
    summary_path = PROJECT_ROOT / "verification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    report = generate_markdown_report(env_info, test_results, bench_results, categories)
    report_path = PROJECT_ROOT / "verification_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print("\n" + "=" * 70)
    print("Reports Generated")
    print("=" * 70)
    print(f"Summary JSON: {summary_path}")
    print(f"Report MD:    {report_path}")
    print(f"\nOverall Status: {summary['overall']}")

    return 0 if summary['overall'] == 'PASS' else 1

if __name__ == "__main__":
    sys.exit(main())
