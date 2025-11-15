#!/usr/bin/env python3
"""
Generate Software Bill of Materials (SBOM).

Uses pipdeptree to create a dependency tree in various formats.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def generate_sbom_json() -> dict[str, Any]:
    """Generate SBOM in JSON format using pipdeptree."""
    try:
        result = subprocess.run(
            ["pipdeptree", "--json-tree"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running pipdeptree: {e}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing pipdeptree JSON: {e}", file=sys.stderr)
        return {}


def generate_sbom_tree() -> str:
    """Generate SBOM in tree format using pipdeptree."""
    try:
        result = subprocess.run(
            ["pipdeptree"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running pipdeptree: {e}", file=sys.stderr)
        return ""


def generate_metadata(sbom_data: dict[str, Any]) -> dict[str, Any]:
    """Generate SBOM metadata summary."""
    if not sbom_data:
        return {}

    total_packages = len(sbom_data)
    total_dependencies = sum(len(pkg.get("dependencies", [])) for pkg in sbom_data)

    return {
        "total_packages": total_packages,
        "total_dependencies": total_dependencies,
        "packages": [
            {
                "name": pkg.get("package", {}).get("key", "unknown"),
                "version": pkg.get("package", {}).get("installed_version", "unknown"),
            }
            for pkg in sbom_data
        ],
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Software Bill of Materials")
    parser.add_argument(
        "--format",
        choices=["json", "tree"],
        default="json",
        help="SBOM format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sbom.json"),
        help="Output file path",
    )
    parser.add_argument(
        "--with-metadata",
        action="store_true",
        help="Include metadata summary",
    )

    args = parser.parse_args()

    print(f"Generating SBOM in {args.format} format...")

    if args.format == "json":
        sbom = generate_sbom_json()

        if args.with_metadata:
            metadata = generate_metadata(sbom)
            output_data = {
                "metadata": metadata,
                "dependencies": sbom,
            }
            args.output.write_text(json.dumps(output_data, indent=2))
        else:
            args.output.write_text(json.dumps(sbom, indent=2))

        print(f"✅ SBOM saved to: {args.output}")
        print(f"   Format: {args.format}")
        print(f"   Size: {args.output.stat().st_size} bytes")

        if sbom:
            print(f"   Total packages: {len(sbom)}")

    elif args.format == "tree":
        sbom = generate_sbom_tree()
        args.output.write_text(sbom)

        print(f"✅ SBOM saved to: {args.output}")
        print(f"   Format: {args.format}")
        print(f"   Size: {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
