#!/usr/bin/env python
"""SUPERMAX Phase 10: HPO Meta-Learning Analysis Tool.

Analyze completed HPO studies to extract meta-learning insights:
- Parameter importance across studies
- Convergence patterns
- Transfer learning recommendations
- Warm-start potential

Usage:
    # Analyze a single study
    python scripts/analyze_hpo.py --study criteria-maximal-hpo

    # Compare multiple studies
    python scripts/analyze_hpo.py \
        --studies criteria-maximal-hpo evidence-maximal-hpo \
        --compare

    # Export analysis for later use
    python scripts/analyze_hpo.py \
        --study criteria-maximal-hpo \
        --export outputs/hpo_analysis/criteria.json

    # Get transfer learning recommendations
    python scripts/analyze_hpo.py \
        --transfer-to joint \
        --available-studies criteria-maximal-hpo evidence-maximal-hpo

    # Analyze convergence patterns
    python scripts/analyze_hpo.py \
        --studies criteria-maximal-hpo evidence-maximal-hpo share-maximal-hpo \
        --analyze-convergence
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from psy_agents_noaug.hpo.meta_learning import (
    TrialHistoryAnalyzer,
    compare_study_performance,
    export_importance_csv,
    print_convergence_analysis,
    print_importance_comparison,
    print_study_summary,
    print_transfer_recommendations,
)

LOGGER = logging.getLogger("analyze_hpo")


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SUPERMAX HPO Meta-Learning Analysis - Extract insights from completed studies"
    )

    # Study selection
    parser.add_argument(
        "--study",
        type=str,
        help="Single study to analyze",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        help="Multiple studies to analyze and compare",
    )

    # Storage
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna.db",
        help="Optuna storage URL (default: sqlite:///optuna.db)",
    )

    # Analysis modes
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple studies (requires --studies)",
    )
    parser.add_argument(
        "--analyze-convergence",
        action="store_true",
        help="Analyze convergence patterns",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed trial information",
    )

    # Transfer learning
    parser.add_argument(
        "--transfer-to",
        type=str,
        choices=["criteria", "evidence", "share", "joint"],
        help="Get transfer learning recommendations for this target task",
    )
    parser.add_argument(
        "--available-studies",
        nargs="+",
        help="Available source studies for transfer learning",
    )

    # Export
    parser.add_argument(
        "--export",
        type=Path,
        help="Export analysis to JSON file",
    )
    parser.add_argument(
        "--export-importance-csv",
        type=Path,
        help="Export parameter importance comparison to CSV",
    )

    # Other options
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def analyze_single_study(
    study_name: str,
    storage: str,
    detailed: bool,
    export_path: Path | None,
) -> None:
    """Analyze a single study.

    Args:
        study_name: Name of study to analyze
        storage: Optuna storage URL
        detailed: Show detailed information
        export_path: Path to export analysis (optional)
    """
    LOGGER.info("Analyzing study: %s", study_name)

    # Create analyzer
    analyzer = TrialHistoryAnalyzer(storage=storage)

    # Analyze study
    try:
        analysis = analyzer.analyze_study(study_name, compute_importance=True)
    except Exception as e:
        LOGGER.error("Failed to analyze study '%s': %s", study_name, e)
        return

    # Print summary
    print_study_summary(analysis, detailed=detailed)

    # Export if requested
    if export_path:
        analyzer.export_analysis(analysis, export_path)
        print(f"\nAnalysis exported to: {export_path}")


def compare_multiple_studies(
    study_names: list[str],
    storage: str,
    analyze_convergence: bool,
    export_csv_path: Path | None,
) -> None:
    """Compare multiple studies.

    Args:
        study_names: List of study names
        storage: Optuna storage URL
        analyze_convergence: Analyze convergence patterns
        export_csv_path: Path to export importance CSV (optional)
    """
    LOGGER.info("Comparing %d studies", len(study_names))

    # Create analyzer
    analyzer = TrialHistoryAnalyzer(storage=storage)

    # Analyze all studies
    analyses = {}
    for study_name in study_names:
        try:
            analysis = analyzer.analyze_study(study_name, compute_importance=True)
            analyses[study_name] = analysis
            LOGGER.info("Analyzed: %s", study_name)
        except Exception as e:
            LOGGER.error("Failed to analyze '%s': %s", study_name, e)

    if not analyses:
        LOGGER.error("No studies successfully analyzed")
        return

    # Print comparisons
    print()
    compare_study_performance(analyses)
    print()
    print_importance_comparison(analyses)

    if analyze_convergence:
        print()
        print_convergence_analysis(analyses)

    # Export if requested
    if export_csv_path:
        export_importance_csv(analyses, export_csv_path)
        print(f"\nParameter importance exported to: {export_csv_path}")


def show_transfer_recommendations(
    target_task: str,
    available_studies: list[str],
) -> None:
    """Show transfer learning recommendations.

    Args:
        target_task: Target task name
        available_studies: Available source study names
    """
    LOGGER.info(
        "Getting transfer recommendations for '%s' from %d sources",
        target_task,
        len(available_studies),
    )

    # Build available studies dict (use study name as both key and value)
    available = {study.split("-")[0]: study for study in available_studies}

    print_transfer_recommendations(target_task, available)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    # Validate arguments
    if not args.study and not args.studies and not args.transfer_to:
        LOGGER.error("Must specify --study, --studies, or --transfer-to")
        return

    # Single study analysis
    if args.study:
        analyze_single_study(
            study_name=args.study,
            storage=args.storage,
            detailed=args.detailed,
            export_path=args.export,
        )

    # Multiple study comparison
    if args.studies and args.compare:
        compare_multiple_studies(
            study_names=args.studies,
            storage=args.storage,
            analyze_convergence=args.analyze_convergence,
            export_csv_path=args.export_importance_csv,
        )

    # Transfer learning recommendations
    if args.transfer_to:
        if not args.available_studies:
            LOGGER.error("--transfer-to requires --available-studies")
            return

        show_transfer_recommendations(
            target_task=args.transfer_to,
            available_studies=args.available_studies,
        )

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
