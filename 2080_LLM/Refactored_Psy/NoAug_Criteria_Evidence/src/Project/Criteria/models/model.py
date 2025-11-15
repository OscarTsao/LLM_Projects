from typing import Optional, Sequence

import torch
import transformers


class ClassificationHead(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        dropout_prob: float = 0.1,
        layer_num: int = 1,
        hidden_dims: Optional[Sequence[int]] = None,
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
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        classifier_layer_num: int = 1,
        classifier_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.encoder = transformers.AutoModel.from_pretrained(model_name)
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
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits
