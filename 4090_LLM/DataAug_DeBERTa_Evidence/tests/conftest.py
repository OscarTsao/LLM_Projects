from __future__ import annotations

import shutil
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ta_module = types.ModuleType("textattack")
ta_augmentation = types.ModuleType("textattack.augmentation")
ta_recipes = types.ModuleType("textattack.augmentation.recipes")


class DummyAugmenter:
    def __init__(self, transformations_per_example=1, **kwargs):
        self.transformations_per_example = transformations_per_example

    def augment(self, text: str, **kwargs):
        return [text + "_aug"]


for name in [
    "EasyDataAugmenter",
    "CharSwapAugmenter",
    "EmbeddingAugmenter",
    "BackTranslationAugmenter",
    "CheckListAugmenter",
    "CLAREAugmenter",
    "TextFoolerJin2019",
    "PWWSRen2019",
    "DeepWordBugGao2018",
    "HotFlipEbrahimi2017",
    "IGAWang2019",
    "Kuleshov2017",
    "CheckList2020",
    "BAEGarg2019",
]:
    setattr(ta_recipes, name, DummyAugmenter)

ta_module.augmentation = ta_augmentation
ta_augmentation.Augmenter = DummyAugmenter
sys.modules.setdefault("textattack", ta_module)
sys.modules.setdefault("textattack.augmentation", ta_augmentation)
sys.modules.setdefault("textattack.augmentation.recipes", ta_recipes)

swap_module = types.ModuleType(
    "textattack.transformations.word_swaps.word_swap_random_character_substitute"
)


class DummySwap:
    pass


swap_module.WordSwapRandomCharacterSubstitute = DummySwap
sys.modules.setdefault(
    "textattack.transformations.word_swaps.word_swap_random_character_substitute",
    swap_module,
)

from dataaug_multi_both.config import load_project_config


@pytest.fixture
def workspace_tmp_path() -> Path:
    base = Path.cwd() / ".pytest_workspace"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def test_config(workspace_tmp_path: Path) -> dict:
    cfg = load_project_config()
    cfg["train"]["num_epochs"] = 1
    cfg["train"]["per_device_batch_size"] = 2
    cfg["train"]["grad_accum_steps"] = 1
    cfg["train"]["num_workers"] = 0
    cfg["train"]["logging_steps"] = 1
    cfg["data"]["cache_dir"] = str(workspace_tmp_path / "data_cache")
    cfg["data"]["token_cache_dir"] = str(workspace_tmp_path / "token_cache")
    cfg["checkpoint"]["dir"] = str(workspace_tmp_path / "checkpoints")
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{workspace_tmp_path / 'mlflow.db'}"
    cfg["mlflow"]["artifact_location"] = str(workspace_tmp_path / "mlruns")
    cfg.setdefault("mlflow", {}).setdefault("buffer", {})
    cfg["mlflow"]["buffer"]["dir"] = str(workspace_tmp_path / "mlflow_buffer")
    cfg["augmentation"]["simple"]["enabled_mask"] = {method: False for method in cfg["augmentation"]["simple"]["enabled_mask"]}
    cfg["augmentation"]["ta"]["enabled_mask"] = {method: False for method in cfg["augmentation"]["ta"]["enabled_mask"]}
    cfg["encoder"]["model_name"] = "hf-internal-testing/tiny-random-roberta"
    cfg["encoder"]["tokenizer_name"] = "hf-internal-testing/tiny-random-roberta"
    cfg["tokenizer"]["max_length"] = 64
    cfg["train"]["max_length"] = 64
    return cfg


@pytest.fixture(autouse=True)
def stub_transformers(monkeypatch):
    class DummyTokenizer:
        pad_token_id = 0
        name_or_path = "dummy-tokenizer"

        def __call__(self, texts, max_length=64, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            input_ids = []
            attention_masks = []
            for text in texts:
                encoded = [(ord(ch) % 50) + 1 for ch in text][:max_length]
                mask = [1] * len(encoded)
                if len(encoded) < max_length:
                    encoded.extend([0] * (max_length - len(encoded)))
                    mask.extend([0] * (max_length - len(mask)))
                input_ids.append(encoded)
                attention_masks.append(mask)
            return {"input_ids": input_ids, "attention_mask": attention_masks}

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 32)
            self.config = SimpleNamespace(hidden_size=32)

        def forward(self, input_ids, attention_mask=None):
            embedded = self.embedding(input_ids)
            return SimpleNamespace(last_hidden_state=embedded)

        def gradient_checkpointing_enable(self):
            return None

    tokenizer = DummyTokenizer()
    encoder = DummyEncoder()

    monkeypatch.setattr(
        "dataaug_multi_both.training.train_loop.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "dataaug_multi_both.model.multitask.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "dataaug_multi_both.evaluation.reporting.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "dataaug_multi_both.model.multitask.AutoModel.from_pretrained",
        lambda *args, **kwargs: DummyEncoder(),
    )
