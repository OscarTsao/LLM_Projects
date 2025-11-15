import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from src.data.tokenization import build_tokenizer
from src.data.collation import SmartBatchCollator
from src.models.llm_classification import LLMClassificationModel


def build_tiny_backbone():
    config = AutoConfig.for_model(
        "gpt2",
        vocab_size=64,
        n_positions=32,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    return AutoModelForCausalLM.from_config(config)


def test_tokenizer_right_padding(monkeypatch):
    class DummyTok:
        def __init__(self):
            self.padding_side = "left"
            self.pad_token = None
            self.eos_token = "<eos>"

    monkeypatch.setattr(
        "src.data.tokenization.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTok(),
    )
    tok = build_tokenizer("any-model")
    assert tok.padding_side == "right"
    assert tok.pad_token == "<eos>"


def test_causal_classifier_logits_shape():
    backbone = build_tiny_backbone()
    model = LLMClassificationModel(backbone=backbone, num_labels=3, mode="causal")
    batch = {
        "input_ids": torch.randint(0, 10, (2, 8)),
        "attention_mask": torch.ones(2, 8),
    }
    outputs = model(**batch)
    assert outputs.logits.shape == (2, 3)


def test_encoderized_pooling_mean():
    backbone = build_tiny_backbone()
    model = LLMClassificationModel(backbone=backbone, num_labels=2, mode="encoderized", pooler="mean")
    batch = {
        "input_ids": torch.randint(0, 10, (1, 6)),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]]),
    }
    outputs = model(**batch)
    assert outputs.logits.shape == (1, 2)


def test_smart_collator_adds_attention_mask():
    backend = Tokenizer(WordLevel({"<pad>": 0, "a": 1, "b": 2}, unk_token="<pad>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<pad>")
    tok.pad_token = "<pad>"
    tok.pad_token_id = 0
    tok.padding_side = "right"
    collator = SmartBatchCollator(tok, pad_to_multiple_of=8)
    features = [
        {"input_ids": [1, 2, 3], "labels": 0},
        {"input_ids": [1, 2], "labels": 1},
    ]
    batch = collator(features)
    assert "attention_mask" in batch
    assert batch["input_ids"].shape[1] % 8 == 0
