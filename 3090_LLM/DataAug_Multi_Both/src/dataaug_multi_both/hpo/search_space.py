"""Optuna search space definition for hyperparameter optimization.

This module defines the search space for HPO, mapping hyperparameters
to Optuna suggestions with support for conditional parameters.
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Try to import Optuna (optional dependency)
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


@dataclass
class SearchSpaceConfig:
    """Configuration for search space definition."""

    # Model hyperparameters
    model_name: str = "categorical"  # categorical choice
    learning_rate: str = "log_uniform"  # log scale
    batch_size: str = "categorical"  # categorical choice
    max_epochs: int = 100  # fixed

    # Loss hyperparameters
    loss_type: str = "categorical"
    focal_gamma: str = "uniform"  # only if loss_type=focal
    label_smoothing: str = "uniform"

    # Optimizer hyperparameters
    optimizer: str = "categorical"
    weight_decay: str = "log_uniform"

    # Data augmentation
    augmentation_prob: str = "uniform"
    augmentation_method: str = "categorical"


class OptunaSearchSpace:
    """Optuna search space for hyperparameter optimization."""

    def __init__(self, config: SearchSpaceConfig | None = None, fixed_epochs: int | None = None):
        """Initialize search space.

        Args:
            config: Search space configuration
            fixed_epochs: If set, use this fixed value instead of searching epochs
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for search space definition. "
                "Install with: pip install optuna"
            )

        self.config = config or SearchSpaceConfig()
        self.fixed_epochs = fixed_epochs

        logger.info(f"Initialized OptunaSearchSpace (fixed_epochs={fixed_epochs})")

    def suggest_hyperparameters(self, trial: "optuna.Trial") -> dict[str, Any]:
        """Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        # Model Architecture - Backbone (removed gated models and problematic ones)
        params["backbone"] = trial.suggest_categorical(
            "backbone",
            [
                # BERT family
                "google-bert/bert-base-uncased",
                "google-bert/bert-large-uncased",
                "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
                # DeBERTa
                "nvidia/quality-classifier-deberta",
                "microsoft/deberta-v3-base",
                # SpanBERT
                "SpanBERT/spanbert-base-cased",
                "SpanBERT/spanbert-large-cased",
                # XLM-RoBERTa
                "FacebookAI/xlm-roberta-base",
                "FacebookAI/xlm-roberta-large",
                "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
                # ELECTRA
                "google/electra-base-discriminator",
                # Longformer (removed large models to reduce memory usage)
                "allenai/longformer-base-4096",
                "allenai/longformer-base-4096-extra.pos.embd.only",
                # BioBERT (removed large model due to disk space issues)
                "dmis-lab/biobert-v1.1",
                # ClinicalBERT
                "medicalai/ClinicalBERT",
                # Mental Health Domain
                # NOTE: mnaylor/psychbert-cased removed - only has Flax weights, not PyTorch
                # NOTE: mnaylor/psychbert-finetuned-mentalhealth removed - same issue
            ],
        )

        # Keep model_name as alias for backward compatibility
        params["model_name"] = params["backbone"]

        # Tokenization
        params["max_length"] = trial.suggest_categorical("max_length", [256, 384, 512, 768, 1024])

        params["doc_stride"] = trial.suggest_categorical("doc_stride", [64, 96, 128])

        params["padding_side"] = trial.suggest_categorical("padding_side", ["left", "right"])

        params["input_format"] = trial.suggest_categorical(
            "input_format", ["binary_pairs", "multi_label"]
        )

        # Encoder freezing and layer-wise LR
        params["layerwise_lr_decay"] = trial.suggest_float("layerwise_lr_decay", 0.75, 0.95)

        params["freeze_layers_n"] = trial.suggest_categorical("freeze_layers_n", [0, 2, 6])

        params["gradient_checkpointing"] = trial.suggest_categorical(
            "gradient_checkpointing", [True, False]
        )

        # Batch / Precision
        params["grad_accum"] = trial.suggest_categorical("grad_accum", [1, 2, 4, 8])

        params["fp_precision"] = trial.suggest_categorical("fp_precision", ["fp16", "bf16", "none"])

        params["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [0.0, 1.0, 2.0, 5.0])

        # Criteria Matching Head
        params["pooling"] = trial.suggest_categorical(
            "pooling", ["cls", "mean", "max", "attention", "last_2_layer_mix"]
        )

        params["head_type"] = trial.suggest_categorical(
            "head_type", ["linear", "mlp", "mlp_residual", "gated", "multi_sample_dropout"]
        )

        params["hidden_size"] = trial.suggest_categorical("hidden_size", [256, 384, 512, 768, 1024])

        params["head_dropout"] = trial.suggest_float("head_dropout", 0.1, 0.5)

        # Backward compatibility aliases
        params["criteria_head_type"] = params["head_type"]
        params["criteria_pooling"] = params["pooling"]
        params["criteria_hidden_dim"] = params["hidden_size"]
        params["criteria_dropout"] = params["head_dropout"]

        # Evidence Binding Head
        params["span_head"] = trial.suggest_categorical(
            "span_head",
            ["start_end_linear", "start_end_mlp", "biaffine", "bio_crf", "sentence_reranker"],
        )

        params["max_spans_per_doc"] = trial.suggest_categorical("max_spans_per_doc", [3, 5, 10])

        params["max_span_len_chars"] = trial.suggest_categorical("max_span_len_chars", [128, 256])

        params["use_sentence_reranker"] = trial.suggest_categorical(
            "use_sentence_reranker", [True, False]
        )

        params["reranker_topk_sent"] = trial.suggest_categorical("reranker_topk_sent", [3, 5, 10])

        params["span_nms_iou"] = trial.suggest_float("span_nms_iou", 0.4, 0.7)

        params["char_level_metric"] = trial.suggest_categorical("char_level_metric", [True, False])

        # Backward compatibility
        params["evidence_head_type"] = params["span_head"]
        params["evidence_dropout"] = trial.suggest_float("evidence_dropout", 0.1, 0.5)

        # Multi-Task Coupling / Joint Training
        params["task_coupling"] = trial.suggest_categorical(
            "task_coupling", ["independent", "coupled"]
        )

        if params["task_coupling"] == "coupled":
            params["coupling_method"] = trial.suggest_categorical(
                "coupling_method", ["concat", "add", "mean", "weighted_sum", "weighted_mean"]
            )

            params["coupling_pooling"] = trial.suggest_categorical(
                "coupling_pooling", ["mean", "max", "attention"]
            )

            params["pool_before_combination"] = trial.suggest_categorical(
                "pool_before_combination", [True, False]
            )

        params["lambda_class"] = trial.suggest_float("lambda_class", 0.5, 2.0)

        params["lambda_span"] = trial.suggest_float("lambda_span", 0.5, 2.0)

        params["use_span2cls_concat"] = trial.suggest_categorical(
            "use_span2cls_concat", [True, False]
        )

        params["span_pooling"] = trial.suggest_categorical(
            "span_pooling", ["mean", "max", "attention"]
        )

        # Loss Function
        params["loss_type"] = trial.suggest_categorical(
            "loss_type", ["ce", "focal", "bce", "weighted_bce", "adaptive_focal", "hybrid"]
        )

        # Backward compatibility
        params["loss_function"] = params["loss_type"]

        if params["loss_type"] == "hybrid":
            params["hybrid_weight_alpha"] = trial.suggest_float("hybrid_weight_alpha", 0.1, 0.9)

        if params["loss_type"] in ["focal", "adaptive_focal", "hybrid"]:
            params["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)

        params["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.2)

        params["class_weights_enabled"] = trial.suggest_categorical(
            "class_weights_enabled", [True, False]
        )

        # Backward compatibility
        params["class_weights"] = params["class_weights_enabled"]

        params["decision_threshold_tau"] = trial.suggest_float("decision_threshold_tau", 0.3, 0.8)

        # Data Augmentation (restricted to TextAttack only)
        params["aug_library"] = trial.suggest_categorical(
            "aug_library", ["none", "textattack"]
        )

        if params["aug_library"] != "none":
            # Select augmentation operations
            aug_choices = trial.suggest_categorical(
                "aug_ops_choice", [0, 1, 2, 3]  # index into different combinations
            )
            aug_combinations = [
                ["synonym"],
                ["backtranslation"],
                ["synonym", "delete"],
                ["contextual"],
            ]
            params["aug_ops"] = aug_combinations[aug_choices]

            params["aug_protect_list_enabled"] = trial.suggest_categorical(
                "aug_protect_list_enabled", [True, False]
            )

            params["aug_prob"] = trial.suggest_float("aug_prob", 0.0, 0.3)
        else:
            params["aug_ops"] = []
            params["aug_protect_list_enabled"] = False
            params["aug_prob"] = 0.0

        # Backward compatibility
        params["augmentation_prob"] = params["aug_prob"]

        # Map to old augmentation_methods format
        augmentation_methods = []
        for op in params["aug_ops"]:
            if op == "synonym":
                augmentation_methods.append("synonym")
            elif op == "backtranslation":
                augmentation_methods.append("back_translation")
            elif op == "delete":
                augmentation_methods.append("swap")
            elif op == "contextual":
                augmentation_methods.append("insert")
        params["augmentation_methods"] = augmentation_methods

        # Activation Function
        params["activation"] = trial.suggest_categorical(
            "activation", ["gelu", "silu", "relu", "leakyrelu", "mish", "tanh"]
        )

        # Regularization
        params["encoder_dropout"] = trial.suggest_float("encoder_dropout", 0.0, 0.2)

        params["attn_dropout"] = trial.suggest_float("attn_dropout", 0.0, 0.2)

        params["token_dropout"] = trial.suggest_float("token_dropout", 0.0, 0.2)

        # Backward compatibility - kept for older code
        params["layer_wise_lr_decay"] = params["layerwise_lr_decay"]

        params["differential_lr_ratio"] = trial.suggest_float("differential_lr_ratio", 0.1, 1.0)

        params["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.0, 0.2)

        params["adversarial_epsilon"] = trial.suggest_float("adversarial_epsilon", 0.0, 0.01)

        # Optimizer and Learning Rates (removed Lion optimizer due to dependency issues)
        params["optimizer"] = trial.suggest_categorical(
            "optimizer", ["adamw_torch", "adafactor", "adam", "adamw"]
        )

        params["encoder_lr"] = trial.suggest_float("encoder_lr", 1e-6, 5e-5, log=True)

        params["head_lr"] = trial.suggest_float("head_lr", 5e-6, 2e-4, log=True)

        # Backward compatibility
        params["learning_rate"] = params["encoder_lr"]

        # Batch size: support separate train and eval batch sizes
        # If separate_batch_sizes is True, search for both independently
        # Otherwise, use the same batch_size for both
        params["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128, 256])

        params["accumulation_steps"] = params["grad_accum"]

        # Epochs: use fixed value if provided, otherwise search
        if self.fixed_epochs is not None:
            params["epochs"] = self.fixed_epochs
            logger.info(f"Using fixed epochs: {self.fixed_epochs}")
        else:
            params["epochs"] = trial.suggest_categorical("epochs", [5, 8, 10, 15])
            logger.info(f"Searching epochs from [5, 8, 10, 15], selected: {params['epochs']}")

        params["max_epochs"] = params["epochs"]

        params["weight_decay"] = trial.suggest_float("weight_decay", 0.0, 0.1)

        # Use separate parameters for beta1 and beta2 to avoid tuple warnings
        beta1 = trial.suggest_categorical("beta1", [0.9])
        beta2 = trial.suggest_categorical("beta2", [0.999, 0.98])
        params["betas"] = [beta1, beta2]  # Use list instead of tuple to avoid Optuna warnings

        params["eps"] = trial.suggest_categorical("eps", [1e-8, 1e-7, 1e-6])

        # Scheduler
        params["scheduler"] = trial.suggest_categorical(
            "scheduler", ["linear", "cosine", "cosine_restart", "one_cycle"]
        )

        if params["scheduler"] == "cosine_restart":
            params["cosine_T_mult"] = trial.suggest_categorical("cosine_T_mult", [1, 2])

        if params["scheduler"] == "one_cycle":
            params["one_cycle_pct_start"] = trial.suggest_float("one_cycle_pct_start", 0.1, 0.3)

        # PEFT: LoRA / Adapters / IA³
        params["peft_method"] = trial.suggest_categorical(
            "peft_method",
            ["none", "lora", "lora_plus", "adalora", "pfeiffer", "houlsby", "compacter", "ia3"],
        )

        if params["peft_method"] in ["lora", "lora_plus", "adalora"]:
            target_choices = trial.suggest_categorical("peft_target_choice", [0, 1, 2, 3])
            target_combinations = [
                ["attn.q", "attn.v"],
                ["attn.q", "attn.k", "attn.v", "attn.o"],
                ["attn.q", "attn.v", "ffn.up", "ffn.down"],
                ["attn.q", "attn.k", "attn.v", "attn.o", "ffn.up", "ffn.down"],
            ]
            params["peft_target_modules"] = target_combinations[target_choices]

            params["peft_r"] = trial.suggest_categorical("peft_r", [4, 8, 16, 32])

            params["peft_alpha"] = trial.suggest_categorical("peft_alpha", [8, 16, 32, 64])

            params["peft_dropout"] = trial.suggest_categorical(
                "peft_dropout", [0.0, 0.05, 0.1, 0.2]
            )

            params["peft_bias"] = trial.suggest_categorical(
                "peft_bias", ["none", "lora_only", "all"]
            )

            params["peft_lr"] = trial.suggest_float("peft_lr", 1e-4, 1e-3, log=True)

        if params["peft_method"] in ["pfeiffer", "houlsby", "compacter"]:
            params["adapter_bottleneck"] = trial.suggest_categorical(
                "adapter_bottleneck", [32, 64, 128, 192]
            )

            params["adapter_dropout"] = trial.suggest_categorical(
                "adapter_dropout", [0.1, 0.2, 0.3]
            )

            params["adapter_layers"] = trial.suggest_categorical(
                "adapter_layers", ["all", "top4", "top8"]
            )

        if params["peft_method"] == "ia3":
            params["ia3_enable_ffn"] = trial.suggest_categorical("ia3_enable_ffn", [True, False])

        params["unfreeze_layernorm"] = trial.suggest_categorical(
            "unfreeze_layernorm", [True, False]
        )

        params["unfreeze_pooler"] = trial.suggest_categorical("unfreeze_pooler", [True, False])

        # Adversarial Training
        params["adv_training"] = trial.suggest_categorical("adv_training", ["none", "fgm", "pgd"])

        if params["adv_training"] != "none":
            params["adv_eps"] = trial.suggest_float("adv_eps", 1e-6, 5e-3, log=True)

            params["adv_steps"] = trial.suggest_categorical("adv_steps", [1, 2, 3])

        # DAPT / TAPT
        params["use_dapt"] = trial.suggest_categorical("use_dapt", [True, False])

        if params["use_dapt"]:
            params["dapt_epochs"] = trial.suggest_categorical("dapt_epochs", [0, 1, 2, 3])

            params["dapt_mlm_prob"] = trial.suggest_float("dapt_mlm_prob", 0.15, 0.3)

        params["use_tapt"] = trial.suggest_categorical("use_tapt", [True, False])

        if params["use_tapt"]:
            params["tapt_epochs"] = trial.suggest_categorical("tapt_epochs", [0, 1, 2])

            params["tapt_mlm_prob"] = trial.suggest_float("tapt_mlm_prob", 0.15, 0.3)

        if params["use_dapt"] or params["use_tapt"]:
            params["masking_style"] = trial.suggest_categorical(
                "masking_style", ["token", "whole_word", "span"]
            )

        # Semi-supervised & Mining
        params["use_pseudo_labels"] = trial.suggest_categorical("use_pseudo_labels", [False, True])

        if params["use_pseudo_labels"]:
            params["pl_conf_thresh"] = trial.suggest_float("pl_conf_thresh", 0.7, 0.95)

            params["pl_weight"] = trial.suggest_float("pl_weight", 0.2, 1.0)

        params["hard_negative_mining"] = trial.suggest_categorical(
            "hard_negative_mining", [True, False]
        )

        if params["hard_negative_mining"]:
            params["hnm_ratio"] = trial.suggest_float("hnm_ratio", 0.1, 0.5)

        # Training Loop
        params["early_stop_patience"] = trial.suggest_categorical("early_stop_patience", [3, 5, 8])

        params["eval_steps"] = trial.suggest_categorical("eval_steps", [200, 500, 1000])

        # Inference & Calibration
        params["temp_scaling"] = trial.suggest_categorical("temp_scaling", [True, False])

        if params["temp_scaling"]:
            params["temperature"] = trial.suggest_float("temperature", 0.5, 5.0)

        params["ensemble_n_models"] = trial.suggest_categorical("ensemble_n_models", [1, 2, 3])

        params["mc_dropout_passes"] = trial.suggest_categorical("mc_dropout_passes", [0, 5, 10])

        params["abstain_entropy_thresh"] = trial.suggest_float("abstain_entropy_thresh", 0.5, 1.5)

        # Retrieval parameters removed - not used in core training pipeline

        # Optimization Metric
        params["optimization_metric"] = trial.suggest_categorical(
            "optimization_metric", ["val_f1_macro", "val_accuracy", "val_f1_micro"]
        )

        # Random Seeds
        params["seed"] = trial.suggest_categorical("seed", [42, 1337, 2025])

        return params

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validate suggested parameters.

        Args:
            params: Dictionary of parameters

        Returns:
            True if valid, False otherwise
        """
        # Check required parameters (with backward compatibility)
        required = ["batch_size", "optimizer"]

        # Check that either new or old parameter names exist
        if "backbone" not in params and "model_name" not in params:
            logger.error("Missing required parameter: backbone/model_name")
            return False

        if "encoder_lr" not in params and "learning_rate" not in params:
            logger.error("Missing required parameter: encoder_lr/learning_rate")
            return False

        if "loss_type" not in params and "loss_function" not in params:
            logger.error("Missing required parameter: loss_type/loss_function")
            return False

        for key in required:
            if key not in params:
                logger.error(f"Missing required parameter: {key}")
                return False

        # Validate ranges with backward compatibility
        lr = params.get("encoder_lr") or params.get("learning_rate")
        if lr and not 1e-6 <= lr <= 5e-5:
            logger.warning(f"Learning rate {lr} outside typical range [1e-6, 5e-5]")

        if params["batch_size"] not in [4, 8, 16, 32, 64, 128, 256]:
            logger.warning(f"Batch size {params['batch_size']} not in standard options")

        epochs = params.get("epochs", params.get("max_epochs", 10))
        if not 1 <= epochs <= 100:
            logger.error(f"Invalid epochs: {epochs}")
            return False

        # Validate conditional parameters
        loss_fn = params.get("loss_type") or params.get("loss_function")
        if loss_fn in ["focal", "adaptive_focal", "hybrid"]:
            if "focal_gamma" not in params:
                logger.error("focal_gamma required for focal/adaptive_focal/hybrid loss")
                return False

            if not 1.0 <= params["focal_gamma"] <= 5.0:
                logger.error(f"Invalid focal_gamma: {params['focal_gamma']}")
                return False

        if loss_fn == "hybrid":
            if "hybrid_weight_alpha" not in params:
                logger.error("hybrid_weight_alpha required for hybrid loss")
                return False

            if not 0.1 <= params["hybrid_weight_alpha"] <= 0.9:
                logger.error(f"Invalid hybrid_weight_alpha: {params['hybrid_weight_alpha']}")
                return False

        # Validate coupling parameters
        if params.get("task_coupling") == "coupled":
            required_coupling = ["coupling_method", "coupling_pooling", "pool_before_combination"]
            for key in required_coupling:
                if key not in params:
                    logger.warning(f"Missing coupling parameter: {key}")

        # Validate head dimensions (with backward compatibility)
        hidden_dim = params.get("hidden_size") or params.get("criteria_hidden_dim", 512)
        if not 128 <= hidden_dim <= 1024:
            logger.warning(f"Hidden dimension {hidden_dim} outside typical range [128, 1024]")

        # Validate dropout rates (flexible)
        for dropout_key in [
            "head_dropout",
            "criteria_dropout",
            "evidence_dropout",
            "encoder_dropout",
        ]:
            if dropout_key in params and not 0.0 <= params[dropout_key] <= 1.0:
                logger.warning(f"{dropout_key} {params[dropout_key]} outside [0.0, 1.0]")

        # Validate augmentation probability
        aug_prob = params.get("aug_prob") or params.get("augmentation_prob", 0.0)
        if not 0.0 <= aug_prob <= 1.0:
            logger.warning(f"Augmentation probability {aug_prob} outside [0.0, 1.0]")

        # Validate regularization parameters (with new ranges)
        lr_decay = params.get("layerwise_lr_decay") or params.get("layer_wise_lr_decay", 0.85)
        if not 0.5 <= lr_decay <= 1.0:
            logger.warning(f"Layer-wise LR decay {lr_decay} outside [0.5, 1.0]")

        if not 0.0 <= params.get("differential_lr_ratio", 1.0) <= 10.0:
            logger.warning("Differential LR ratio outside reasonable range")

        return True


def create_search_space(config: SearchSpaceConfig | None = None, fixed_epochs: int | None = None) -> "OptunaSearchSpace":
    """Factory function to create search space.

    Args:
        config: Search space configuration
        fixed_epochs: If set, use this fixed value instead of searching epochs

    Returns:
        Initialized search space

    Example:
        search_space = create_search_space(fixed_epochs=100)
        params = search_space.suggest_hyperparameters(trial)
    """
    return OptunaSearchSpace(config, fixed_epochs=fixed_epochs)


def suggest_trial_config(trial: "optuna.Trial", fixed_epochs: int | None = None) -> dict[str, Any]:
    """Suggest a complete trial configuration.

    Args:
        trial: Optuna trial object
        fixed_epochs: If set, use this fixed value instead of searching epochs

    Returns:
        Dictionary of trial configuration

    Example:
        def objective(trial):
            config = suggest_trial_config(trial, fixed_epochs=100)
            # Train model with config
            return validation_metric
    """
    search_space = create_search_space(fixed_epochs=fixed_epochs)
    params = search_space.suggest_hyperparameters(trial)

    # Validate
    if not search_space.validate_params(params):
        raise ValueError("Invalid trial configuration")

    return params


# Predefined search space configurations
SEARCH_SPACE_SMALL = SearchSpaceConfig(
    # Smaller search space for quick experiments
)

SEARCH_SPACE_FULL = SearchSpaceConfig(
    # Full search space for comprehensive HPO
)
