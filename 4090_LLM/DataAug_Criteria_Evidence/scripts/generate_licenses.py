#!/usr/bin/env python3
"""
Generate license compliance report.

Uses pip-licenses to list all dependencies and their licenses.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def generate_license_report(output_format: str = "markdown") -> str:
    """Generate license report using pip-licenses."""
    format_map = {
        "markdown": "markdown",
        "json": "json",
        "csv": "csv",
        "html": "html",
    }

    try:
        result = subprocess.run(
            [
                "pip-licenses",
                "--format",
                format_map[output_format],
                "--with-urls",
                "--with-description",
                "--order=license",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running pip-licenses: {e}", file=sys.stderr)
        return ""


def add_header(report: str, output_format: str) -> str:
    """Add header to license report."""
    if output_format == "markdown":
        header = """# Third-Party Licenses

This document lists all third-party dependencies and their licenses.

---

"""
        return header + report
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate license compliance report")
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "csv", "html"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("THIRD_PARTY_LICENSES.md"),
        help="Output file path",
    )

    args = parser.parse_args()

    print("Generating license report...")
    report = generate_license_report(args.format)

    if report:
        # Add header for markdown format
        if args.format == "markdown":
            report = add_header(report, args.format)

        args.output.write_text(report)
        print(f"✅ License report saved to: {args.output}")
        print(f"   Format: {args.format}")
        print(f"   Size: {args.output.stat().st_size} bytes")
    else:
        print("❌ Failed to generate license report", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
