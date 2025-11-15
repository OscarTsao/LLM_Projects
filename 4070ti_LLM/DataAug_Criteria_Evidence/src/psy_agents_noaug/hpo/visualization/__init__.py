"""Results analysis and visualization for HPO (Phase 13).

This module provides comprehensive tools for analyzing and visualizing
hyperparameter optimization results.

Key Features:
- Optimization history plots
- Parameter importance visualization
- Parallel coordinates plots
- Hyperparameter interaction analysis
- Automated report generation
- Study comparison visualizations

Example Usage:

    # Basic visualization
    from psy_agents_noaug.hpo.visualization import HPOVisualizer
    import optuna

    study = optuna.load_study("my-study", storage="sqlite:///optuna.db")
    visualizer = HPOVisualizer(study)

    # Generate optimization history plot
    visualizer.plot_optimization_history(save_path="history.png")

    # Parameter importance
    visualizer.plot_param_importance(save_path="importance.png")

    # Parallel coordinates
    visualizer.plot_parallel_coordinates(save_path="parallel.png")

    # Generate comprehensive report
    from psy_agents_noaug.hpo.visualization import ReportGenerator

    generator = ReportGenerator(study)
    generator.generate_report(output_dir="outputs/hpo_report/")

    # Compare multiple studies
    from psy_agents_noaug.hpo.visualization import StudyComparator

    studies = [
        optuna.load_study("criteria-max", storage=storage),
        optuna.load_study("evidence-max", storage=storage),
    ]
    comparator = StudyComparator(studies)
    comparator.plot_study_comparison(save_path="comparison.png")
"""

# Visualization
from psy_agents_noaug.hpo.visualization.plots import (
    HPOVisualizer,
    plot_optimization_history,
    plot_param_importance,
    plot_parallel_coordinates,
)

# Analysis
from psy_agents_noaug.hpo.visualization.analysis import (
    ParameterAnalyzer,
    ConvergenceAnalyzer,
    analyze_hyperparameter_interactions,
)

# Reporting
from psy_agents_noaug.hpo.visualization.reports import (
    ReportGenerator,
    StudyComparator,
    export_study_summary,
)

__all__ = [
    # Visualization
    "HPOVisualizer",
    "plot_optimization_history",
    "plot_param_importance",
    "plot_parallel_coordinates",
    # Analysis
    "ParameterAnalyzer",
    "ConvergenceAnalyzer",
    "analyze_hyperparameter_interactions",
    # Reporting
    "ReportGenerator",
    "StudyComparator",
    "export_study_summary",
]
