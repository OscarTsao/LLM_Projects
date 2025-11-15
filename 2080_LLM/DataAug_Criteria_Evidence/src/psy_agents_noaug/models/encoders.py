"""Transformer encoder wrappers for text representation.

These thin wrappers centralise a few cross‑model choices:
  - pooling strategy (``cls`` vs. ``mean``)
  - optional gradient checkpointing (memory vs. speed trade‑off)
  - optional LoRA adapters (via ``peft``) for parameter‑efficient finetuning
"""

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

try:  # Optional dependency for LoRA
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional feature
    PEFT_AVAILABLE = False


class TransformerEncoder(nn.Module):
    """
    Base transformer encoder using HuggingFace models.

    Supports:
    - BERT, RoBERTa, DeBERTa variants
    - Frozen or fine-tunable encoders
    - CLS token pooling or mean pooling
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze_encoder: bool = False,
        pooling_strategy: str = "cls",
        max_length: int = 512,
        gradient_checkpointing: bool = False,
        lora_config: dict[str, Any] | None = None,
    ):
        """
        Initialize encoder.

        Args:
            model_name: HuggingFace model name
            freeze_encoder: Whether to freeze encoder weights
            pooling_strategy: Pooling method ('cls' or 'mean')
            max_length: Maximum sequence length
        """
        super().__init__()

        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length

        # Load tokenizer and model from HuggingFace hub/local cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()

        # Apply optional LoRA adapters
        if lora_config and lora_config.get("enabled", False):
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "peft is required when enabling LoRA. Install with `pip install peft`."
                )

            target_modules = lora_config.get("target_modules", ["query", "value"])
            lora_cfg = LoraConfig(
                r=int(lora_config.get("r", 8)),
                lora_alpha=int(lora_config.get("alpha", 16)),
                lora_dropout=float(lora_config.get("dropout", 0.1)),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=target_modules,
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)

        # Get hidden size
        self.hidden_size = self.encoder.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode input text.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Encoded representations [batch_size, hidden_size]
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Apply pooling
        if self.pooling_strategy == "cls":
            # Use CLS token representation
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            # Mean pooling over sequence
            last_hidden = outputs.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
                last_hidden.size()
            )
            sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, dim=1)
            sum_mask = attention_mask_expanded.sum(dim=1)
            pooled = sum_embeddings / sum_mask.clamp(min=1e-9)
        else:
            raise ValueError(
                f"Unknown pooling strategy: {self.pooling_strategy}. "
                "Must be 'cls' or 'mean'"
            )

        return pooled

    def encode_texts(
        self,
        texts: list[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Encode a batch of texts.

        Args:
            texts: List of input texts
            device: Target device

        Returns:
            Encoded representations [batch_size, hidden_size]
        """
        # Tokenize a list of strings; batch encode for efficiency
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if device:
            encoded = {k: v.to(device) for k, v in encoded.items()}

        # Encode
        with torch.no_grad():
            return self.forward(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )


class BERTEncoder(TransformerEncoder):
    """BERT-specific encoder."""

    def __init__(
        self,
        variant: str = "base",
        freeze_encoder: bool = False,
        pooling_strategy: str = "cls",
        max_length: int = 512,
        gradient_checkpointing: bool = False,
        lora_config: dict[str, Any] | None = None,
    ):
        """
        Initialize BERT encoder.

        Args:
            variant: BERT variant ('base', 'large')
            freeze_encoder: Whether to freeze encoder weights
            pooling_strategy: Pooling method ('cls' or 'mean')
            max_length: Maximum sequence length
        """
        model_name = f"bert-{variant}-uncased"
        super().__init__(
            model_name=model_name,
            freeze_encoder=freeze_encoder,
            pooling_strategy=pooling_strategy,
            max_length=max_length,
            gradient_checkpointing=gradient_checkpointing,
            lora_config=lora_config,
        )


class RoBERTaEncoder(TransformerEncoder):
    """RoBERTa-specific encoder."""

    def __init__(
        self,
        variant: str = "base",
        freeze_encoder: bool = False,
        pooling_strategy: str = "cls",
        max_length: int = 512,
        gradient_checkpointing: bool = False,
        lora_config: dict[str, Any] | None = None,
    ):
        """
        Initialize RoBERTa encoder.

        Args:
            variant: RoBERTa variant ('base', 'large')
            freeze_encoder: Whether to freeze encoder weights
            pooling_strategy: Pooling method ('cls' or 'mean')
            max_length: Maximum sequence length
        """
        model_name = f"roberta-{variant}"
        super().__init__(
            model_name=model_name,
            freeze_encoder=freeze_encoder,
            pooling_strategy=pooling_strategy,
            max_length=max_length,
            gradient_checkpointing=gradient_checkpointing,
            lora_config=lora_config,
        )


class DeBERTaEncoder(TransformerEncoder):
    """DeBERTa-v3-specific encoder."""

    def __init__(
        self,
        variant: str = "base",
        freeze_encoder: bool = False,
        pooling_strategy: str = "cls",
        max_length: int = 512,
        gradient_checkpointing: bool = False,
        lora_config: dict[str, Any] | None = None,
    ):
        """
        Initialize DeBERTa encoder.

        Args:
            variant: DeBERTa variant ('base', 'large')
            freeze_encoder: Whether to freeze encoder weights
            pooling_strategy: Pooling method ('cls' or 'mean')
            max_length: Maximum sequence length
        """
        model_name = f"microsoft/deberta-v3-{variant}"
        super().__init__(
            model_name=model_name,
            freeze_encoder=freeze_encoder,
            pooling_strategy=pooling_strategy,
            max_length=max_length,
            gradient_checkpointing=gradient_checkpointing,
            lora_config=lora_config,
        )
