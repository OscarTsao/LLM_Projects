#!/usr/bin/env python3
"""
Security vulnerability scanner using pip-audit.

Scans all installed packages for known vulnerabilities from PyPI Advisory Database
and OSV (Open Source Vulnerabilities).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_pip_audit(
    output_format: str = "json",
    severity_threshold: str = "low",
    ignore_vulns: list[str] = None,
) -> dict[str, Any]:
    """
    Run pip-audit and return results.

    Args:
        output_format: Output format (json, markdown, cyclonedx)
        severity_threshold: Minimum severity to report (low, medium, high, critical)
        ignore_vulns: List of vulnerability IDs to ignore

    Returns:
        Dictionary with audit results
    """
    cmd = ["pip-audit", "--format", output_format, "--progress-spinner=off"]

    if ignore_vulns:
        for vuln_id in ignore_vulns:
            cmd.extend(["--ignore-vuln", vuln_id])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise on vulnerabilities found
        )

        if output_format == "json":
            return json.loads(result.stdout) if result.stdout else {}
        return {"stdout": result.stdout, "stderr": result.stderr}

    except subprocess.CalledProcessError as e:
        print(f"Error running pip-audit: {e}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing pip-audit JSON: {e}", file=sys.stderr)
        return {}


def filter_by_severity(results: dict[str, Any], threshold: str) -> dict[str, Any]:
    """Filter vulnerabilities by severity threshold."""
    severity_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    threshold_level = severity_levels.get(threshold.lower(), 0)

    if "dependencies" not in results:
        return results

    filtered_deps = []
    for dep in results["dependencies"]:
        filtered_vulns = [
            v
            for v in dep.get("vulns", [])
            if severity_levels.get(v.get("severity", "").lower(), 0) >= threshold_level
        ]
        if filtered_vulns:
            dep["vulns"] = filtered_vulns
            filtered_deps.append(dep)

    results["dependencies"] = filtered_deps
    return results


def generate_report(results: dict[str, Any], output_file: Path = None) -> str:
    """Generate human-readable security report."""
    if not results or "dependencies" not in results:
        return "✅ No vulnerabilities found!"

    deps = results["dependencies"]
    if not deps:
        return "✅ No vulnerabilities found!"

    total_vulns = sum(len(d.get("vulns", [])) for d in deps)

    report_lines = [
        "=" * 80,
        f"SECURITY AUDIT REPORT - {total_vulns} vulnerabilities found",
        "=" * 80,
        "",
    ]

    for dep in deps:
        dep_name = dep.get("name", "unknown")
        dep_version = dep.get("version", "unknown")
        vulns = dep.get("vulns", [])

        report_lines.append(f"📦 {dep_name} {dep_version}")
        report_lines.append("-" * 80)

        for vuln in vulns:
            vuln_id = vuln.get("id", "N/A")
            severity = vuln.get("severity", "unknown").upper()
            description = vuln.get("description", "No description")
            fix_versions = vuln.get("fix_versions", [])

            report_lines.append(f"  🔴 {vuln_id} [{severity}]")
            report_lines.append(f"     {description}")
            if fix_versions:
                report_lines.append(f"     Fix: Upgrade to {', '.join(fix_versions)}")
            report_lines.append("")

    report = "\n".join(report_lines)

    if output_file:
        output_file.write_text(report)
        print(f"Report saved to: {output_file}")

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Security vulnerability scanner")
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "cyclonedx"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--severity",
        choices=["low", "medium", "high", "critical"],
        default="low",
        help="Minimum severity threshold",
    )
    parser.add_argument(
        "--ignore",
        nargs="+",
        help="Vulnerability IDs to ignore",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path",
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with code 1 if critical vulnerabilities found",
    )

    args = parser.parse_args()

    print("Running security audit...")
    results = run_pip_audit(args.format, args.severity, args.ignore or [])

    if args.format == "json":
        filtered = filter_by_severity(results, args.severity)
        report = generate_report(filtered, args.output)
        print(report)

        # Check for critical vulnerabilities
        if args.fail_on_critical:
            critical_count = sum(
                1
                for dep in filtered.get("dependencies", [])
                for vuln in dep.get("vulns", [])
                if vuln.get("severity", "").lower() == "critical"
            )
            if critical_count > 0:
                print(
                    f"\n❌ Found {critical_count} critical vulnerabilities!",
                    file=sys.stderr,
                )
                sys.exit(1)
    else:
        print(results.get("stdout", ""))
        if results.get("stderr"):
            print(results["stderr"], file=sys.stderr)


if __name__ == "__main__":
    main()
