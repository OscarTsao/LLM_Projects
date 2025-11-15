"""Data loading and processing for LLM_Evidence_Gemma."""

from .evidence_dataset import (
    EvidenceDataset,
    CachedEvidenceDataset,
    build_evidence_dataset,
    load_redsm5_evidence,
    prepare_evidence_examples,
    get_symptom_labels,
    get_symptom_distribution,
    SYMPTOM_LABELS,
    NUM_SYMPTOMS,
)
from .cv_splits import (
    create_cv_splits,
    load_fold_split,
    get_fold_statistics,
    load_fold_metadata,
    verify_fold_stratification,
)
from .tokenization import build_tokenizer, apply_pair_format, LengthBucket
from .collation import SmartBatchCollator
from .classification_dataset import (
    load_classification_dataset,
    infer_label_list,
    compute_class_weights,
)

__all__ = [
    'EvidenceDataset',
    'CachedEvidenceDataset',
    'build_evidence_dataset',
    'load_redsm5_evidence',
    'prepare_evidence_examples',
    'get_symptom_labels',
    'get_symptom_distribution',
    'SYMPTOM_LABELS',
    'NUM_SYMPTOMS',
    'create_cv_splits',
    'load_fold_split',
    'get_fold_statistics',
    'load_fold_metadata',
    'verify_fold_stratification',
    'build_tokenizer',
    'apply_pair_format',
    'LengthBucket',
    'SmartBatchCollator',
    'load_classification_dataset',
    'infer_label_list',
    'compute_class_weights',
]
