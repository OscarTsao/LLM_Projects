from collections.abc import Sequence
from typing import Any

import torch
import transformers


class ClassificationHead(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        dropout_prob: float = 0.1,
        layer_num: int = 1,
        hidden_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if layer_num < 1:
            raise ValueError("layer_num must be at least 1.")

        if hidden_dims is not None:
            hidden_dims = tuple(hidden_dims)
            if len(hidden_dims) != layer_num - 1:
                raise ValueError(
                    "hidden_dims length must be equal to layer_num - 1 when provided."
                )
        else:
            hidden_dims = tuple([input_dim] * (layer_num - 1))

        dims = (input_dim,) + hidden_dims

        hidden_layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            hidden_layers.append(torch.nn.Linear(in_dim, out_dim))
            hidden_layers.append(torch.nn.GELU())
            hidden_layers.append(torch.nn.Dropout(dropout_prob))

        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.output_layer = torch.nn.Linear(dims[-1], num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        if len(self.hidden_layers) > 0:
            x = self.hidden_layers(x)
        return self.output_layer(x)


class Model(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        head_cfg: dict[str, Any] | None = None,
        task_cfg: dict[str, Any] | None = None,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        classifier_layer_num: int = 1,
        classifier_hidden_dims: Sequence[int] | None = None,
    ) -> None:
        """
        Initialize Criteria classification model.

        Args:
            model_name: HuggingFace model identifier
            head_cfg: Head configuration dict (optional, for HPO compatibility)
                Expected keys: layers, hidden, activation, dropout, pooling
            task_cfg: Task configuration dict (optional, for HPO compatibility)
                Expected keys: num_labels
            num_labels: Number of output classes (overridden by task_cfg)
            classifier_dropout: Dropout probability (overridden by head_cfg)
            classifier_layer_num: Number of layers (overridden by head_cfg)
            classifier_hidden_dims: Hidden dimensions (overridden by head_cfg)
        """
        super().__init__()

        # Extract from head_cfg if provided (for HPO compatibility)
        if head_cfg:
            classifier_dropout = head_cfg.get("dropout", classifier_dropout)
            classifier_layer_num = head_cfg.get("layers", classifier_layer_num)
            hidden = head_cfg.get("hidden", None)
            if hidden is not None:
                if isinstance(hidden, int):
                    # Single value: create uniform hidden dims
                    classifier_hidden_dims = (hidden,) * (classifier_layer_num - 1)
                elif isinstance(hidden, (list, tuple)):
                    # Sequence: use as provided
                    classifier_hidden_dims = tuple(hidden)

        # Extract from task_cfg if provided
        if task_cfg:
            num_labels = task_cfg.get("num_labels", num_labels)

        self.encoder = transformers.AutoModel.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,  # Prevent meta device issues in parallel HPO
            device_map=None,          # Explicitly prevent device_map auto-assignment
            torch_dtype=None          # Let model use its default dtype (not auto)
        )
        hidden_size = self.encoder.config.hidden_size

        if classifier_hidden_dims is not None:
            classifier_layer_num = len(classifier_hidden_dims) + 1

        self.classifier = ClassificationHead(
            input_dim=hidden_size,
            num_labels=num_labels,
            dropout_prob=classifier_dropout,
            layer_num=classifier_layer_num,
            hidden_dims=classifier_hidden_dims,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        # Handle models without pooler_output (DeBERTa, ELECTRA, etc.)
        # Check attribute existence first to avoid AttributeError
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Fallback: Use [CLS] token representation (first token)
            pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits
