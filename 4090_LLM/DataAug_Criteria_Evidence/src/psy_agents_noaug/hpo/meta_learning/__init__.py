"""Meta-learning and warm-starting for HPO (Phase 10).

This module provides advanced meta-learning capabilities for HPO:
- Trial history analysis and knowledge extraction
- Warm-starting new studies from previous runs
- Transfer learning across related tasks
- Parameter importance analysis
- Cross-architecture knowledge transfer
- Visualization and comparison utilities

Example Usage:

    # Analyze a completed study
    from psy_agents_noaug.hpo.meta_learning import TrialHistoryAnalyzer
    analyzer = TrialHistoryAnalyzer(storage="sqlite:///optuna.db")
    analysis = analyzer.analyze_study("criteria-maximal-hpo")
    print_study_summary(analysis)

    # Warm-start a new study from previous results
    from psy_agents_noaug.hpo.meta_learning import WarmStartStrategy
    warm_starter = WarmStartStrategy(storage="sqlite:///optuna.db")
    new_study = optuna.create_study(...)
    warm_starter.warm_start_from_study(
        target_study=new_study,
        source_study="criteria-maximal-hpo",
        n_configs=5,
        strategy="best",
    )

    # Transfer knowledge across tasks
    from psy_agents_noaug.hpo.meta_learning import TransferLearner
    transfer = TransferLearner(storage="sqlite:///optuna.db")
    evidence_study = optuna.create_study(...)
    transfer.transfer_from_task(
        target_study=evidence_study,
        source_task="criteria",
        target_task="evidence",
        source_study="criteria-maximal-hpo",
        n_configs=5,
    )

    # Compare parameter importance across studies
    from psy_agents_noaug.hpo.meta_learning import print_importance_comparison
    studies = {
        "criteria": analyzer.analyze_study("criteria-maximal-hpo"),
        "evidence": analyzer.analyze_study("evidence-maximal-hpo"),
    }
    print_importance_comparison(studies)
"""

# History analysis
from psy_agents_noaug.hpo.meta_learning.history import (
    StudyAnalysis,
    TrialHistoryAnalyzer,
    TrialSummary,
)

# Transfer learning
from psy_agents_noaug.hpo.meta_learning.transfer import (
    CrossArchitectureTransfer,
    TransferLearner,
    recommend_transfer_sources,
)

# Visualization
from psy_agents_noaug.hpo.meta_learning.visualization import (
    analyze_warm_start_impact,
    compare_study_performance,
    export_importance_csv,
    print_convergence_analysis,
    print_importance_comparison,
    print_study_summary,
    print_transfer_recommendations,
)

# Warm-starting
from psy_agents_noaug.hpo.meta_learning.warm_start import (
    AdaptiveWarmStart,
    WarmStartStrategy,
)

__all__ = [
    # History analysis
    "TrialHistoryAnalyzer",
    "StudyAnalysis",
    "TrialSummary",
    # Warm-starting
    "WarmStartStrategy",
    "AdaptiveWarmStart",
    # Transfer learning
    "TransferLearner",
    "CrossArchitectureTransfer",
    "recommend_transfer_sources",
    # Visualization
    "print_study_summary",
    "print_importance_comparison",
    "print_transfer_recommendations",
    "print_convergence_analysis",
    "compare_study_performance",
    "analyze_warm_start_impact",
    "export_importance_csv",
]
