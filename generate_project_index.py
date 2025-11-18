#!/usr/bin/env python3
"""
Generate a comprehensive project index JSON for all LLM projects.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# Base directory
BASE_DIR = Path("/home/user/LLM_Projects")

# GPU directories to scan
GPU_DIRS = ["2080_LLM", "3090_LLM", "4070ti_LLM", "4090_LLM"]


def read_readme(project_path: Path) -> Optional[str]:
    """Read README.md file from project directory."""
    readme_path = project_path / "README.md"
    if readme_path.exists():
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Return first 500 chars as summary
                return content[:500].strip()
        except Exception as e:
            print(f"Error reading {readme_path}: {e}")
            return None
    return None


def determine_task_type(project_name: str, readme: Optional[str]) -> str:
    """Determine task type based on project name and README content."""
    name_lower = project_name.lower()
    readme_lower = (readme or "").lower()

    # Check for specific task types
    if "rag" in name_lower:
        return "rag"
    elif "reranker" in name_lower or "gemini_reranker" in name_lower:
        return "reranker_llm_judge"
    elif "augmentation" in name_lower or "dataaug" in name_lower:
        return "data_augmentation_pipeline"
    elif "agent" in name_lower and ("psy" in name_lower or "multi" in name_lower):
        return "psy_multi_agent"
    elif ("criteria" in name_lower and "evidence" in name_lower) or "both" in name_lower:
        return "multi_task_criteria_evidence"
    elif "criteria" in name_lower:
        return "criteria_matching"
    elif "evidence" in name_lower:
        if "span" in name_lower or "spanbert" in name_lower:
            return "evidence_span"
        else:
            return "evidence_sentence"
    elif "baseline" in name_lower:
        return "baseline"
    else:
        return "other"


def determine_model_family(project_name: str, readme: Optional[str], project_path: Path) -> str:
    """Determine model family based on project name, README, and code files."""
    name_lower = project_name.lower()
    readme_lower = (readme or "").lower()

    # Check project name first
    if "deberta" in name_lower or "deberta" in readme_lower:
        if "v3" in name_lower or "v3" in readme_lower:
            return "deberta_v3"
        return "deberta"
    elif "spanbert" in name_lower or "spanbert" in readme_lower:
        return "spanbert"
    elif "roberta" in name_lower or "roberta" in readme_lower:
        return "roberta"
    elif "llama" in name_lower or "llama" in readme_lower:
        return "llama"
    elif "qwen" in name_lower or "qwen" in readme_lower:
        return "qwen"
    elif "gemma" in name_lower or "gemma" in readme_lower:
        return "gemma"
    elif "gemini" in name_lower or "reranker" in name_lower:
        return "llm_reranker"
    elif "gnn" in name_lower or "graph" in name_lower:
        return "gnn"
    elif "bert" in name_lower or "bert" in readme_lower:
        return "bert"

    # Try to find model info in code files
    try:
        # Check for common model config patterns
        for file_path in project_path.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(2000).lower()  # Read first 2000 chars
                    if "deberta-v3" in content or "debertav3" in content:
                        return "deberta_v3"
                    elif "deberta" in content:
                        return "deberta"
                    elif "spanbert" in content:
                        return "spanbert"
                    elif "roberta" in content:
                        return "roberta"
                    elif "gemma" in content:
                        return "gemma"
                    elif "llama" in content:
                        return "llama"
                    elif "qwen" in content:
                        return "qwen"
            except:
                pass
    except:
        pass

    # Default fallback
    if "multi" in name_lower:
        return "deberta_v3"  # Multi usually means DeBERTa-v3

    return "bert"  # Default to BERT


def determine_status(project_name: str, readme: Optional[str]) -> str:
    """Determine project status based on project name and README."""
    name_lower = project_name.lower()
    readme_lower = (readme or "").lower()

    if "baseline" in name_lower:
        return "baseline"
    elif "rebuild" in name_lower or "refactor" in name_lower:
        return "mainline"
    elif "test" in name_lower or "jupyter" in name_lower:
        return "prototype"
    elif "deprecated" in readme_lower:
        return "deprecated"
    elif any(x in name_lower for x in ["dataaug_multi", "dataaug_deberta", "psy_agent", "psy_rag"]):
        return "mainline"
    elif "noaug" in name_lower:
        return "baseline"
    else:
        return "unknown"


def generate_readme_summary(readme: Optional[str], project_name: str) -> str:
    """Generate a concise README summary."""
    if not readme:
        return f"Project: {project_name}"

    # Try to extract first meaningful line or paragraph
    lines = readme.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and len(line) > 20:
            # Limit to 200 chars
            if len(line) > 200:
                return line[:197] + "..."
            return line

    # Fallback to first 200 chars
    if len(readme) > 200:
        return readme[:197].strip() + "..."
    return readme.strip()


def scan_projects() -> List[Dict]:
    """Scan all projects and generate index data."""
    projects = []

    for gpu_group in GPU_DIRS:
        gpu_path = BASE_DIR / gpu_group
        if not gpu_path.exists():
            print(f"Warning: {gpu_path} does not exist")
            continue

        # Get all subdirectories
        for project_dir in sorted(gpu_path.iterdir()):
            if not project_dir.is_dir():
                continue

            project_name = project_dir.name
            relative_path = f"{gpu_group}/{project_name}"

            print(f"Processing: {relative_path}")

            # Read README
            readme = read_readme(project_dir)

            # Determine properties
            task_type = determine_task_type(project_name, readme)
            model_family = determine_model_family(project_name, readme, project_dir)
            status = determine_status(project_name, readme)
            readme_summary = generate_readme_summary(readme, project_name)

            # Create project entry
            project_entry = {
                "gpu_group": gpu_group,
                "project_path": relative_path,
                "project_name": project_name,
                "task_type": task_type,
                "model_family": model_family,
                "status": status,
                "readme_summary": readme_summary
            }

            projects.append(project_entry)

            print(f"  -> Task: {task_type}, Model: {model_family}, Status: {status}")

    return projects


def main():
    """Main function to generate project index."""
    print("=" * 80)
    print("Generating Project Index for LLM_Projects")
    print("=" * 80)

    projects = scan_projects()

    # Sort by gpu_group and project_name
    projects.sort(key=lambda x: (x['gpu_group'], x['project_name']))

    # Write to JSON file
    output_file = BASE_DIR / "project_index.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(projects, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Project index generated: {output_file}")
    print(f"Total projects: {len(projects)}")
    print("=" * 80)

    # Print summary statistics
    gpu_counts = {}
    task_counts = {}
    model_counts = {}
    status_counts = {}

    for project in projects:
        gpu_counts[project['gpu_group']] = gpu_counts.get(project['gpu_group'], 0) + 1
        task_counts[project['task_type']] = task_counts.get(project['task_type'], 0) + 1
        model_counts[project['model_family']] = model_counts.get(project['model_family'], 0) + 1
        status_counts[project['status']] = status_counts.get(project['status'], 0) + 1

    print("\nSummary Statistics:")
    print("-" * 80)
    print("\nBy GPU Group:")
    for gpu, count in sorted(gpu_counts.items()):
        print(f"  {gpu}: {count} projects")

    print("\nBy Task Type:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} projects")

    print("\nBy Model Family:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count} projects")

    print("\nBy Status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} projects")

    print("=" * 80)


if __name__ == "__main__":
    main()
