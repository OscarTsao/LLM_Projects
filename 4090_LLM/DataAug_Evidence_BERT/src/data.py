from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


@dataclass
class QAExample:
    """Container for a single QA example."""

    id: str
    question: str
    context: str
    answer_text: str
    answer_start: int


def _normalize_symptom(symptom: str) -> str:
    """Normalize DSM-5 symptom tags for natural language questions."""
    return symptom.replace("_", " ").lower()


def _build_examples(data_cfg: DictConfig) -> List[QAExample]:
    posts = pd.read_csv(data_cfg.post_path)
    annotations = pd.read_csv(data_cfg.annotation_path)

    posts = posts.set_index(data_cfg.post_id_column)
    examples: List[QAExample] = []
    skipped = 0

    for row in annotations.itertuples(index=False):
        if getattr(row, "status") != data_cfg.positive_status_value:
            continue

        post_id = getattr(row, data_cfg.post_id_column)
        if post_id not in posts.index:
            skipped += 1
            continue

        context = posts.loc[post_id, data_cfg.context_column]
        question = data_cfg.question_template.format(
            symptom=_normalize_symptom(getattr(row, "DSM5_symptom"))
        )
        answer_text = getattr(row, "sentence_text")
        start_char = context.find(answer_text)

        if start_char == -1:
            LOGGER.debug("Skipping example - unable to align answer for post_id=%s", post_id)
            skipped += 1
            continue

        example = QAExample(
            id=f"{post_id}",
            question=question,
            context=context,
            answer_text=answer_text,
            answer_start=start_char,
        )
        examples.append(example)

    if skipped:
        LOGGER.warning("Skipped %d annotation rows due to alignment issues.", skipped)

    return examples


def _train_val_test_split(
    examples: List[QAExample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[QAExample]]:
    train_examples, temp_examples = train_test_split(
        examples, train_size=train_ratio, random_state=seed, shuffle=True
    )
    if val_ratio <= 0:
        return {"train": train_examples, "validation": temp_examples, "test": []}

    remaining_ratio = 1.0 - train_ratio
    test_ratio = max(1e-6, remaining_ratio - val_ratio)
    if test_ratio <= 0:
        val_examples = temp_examples
        test_examples: List[QAExample] = []
    else:
        relative_val_ratio = val_ratio / remaining_ratio
        val_examples, test_examples = train_test_split(
            temp_examples, train_size=relative_val_ratio, random_state=seed, shuffle=True
        )

    return {"train": train_examples, "validation": val_examples, "test": test_examples}


def _examples_to_dataset(examples: List[QAExample]) -> Dataset:
    data = {
        "id": [ex.id for ex in examples],
        "question": [ex.question for ex in examples],
        "context": [ex.context for ex in examples],
        "answers": [
            {"text": [ex.answer_text], "answer_start": [ex.answer_start]} for ex in examples
        ],
    }
    return Dataset.from_dict(data)


def load_datasets(data_cfg: DictConfig, seed: int) -> DatasetDict:
    """Load the REDSM5 dataset and return a DatasetDict with splits."""
    examples = _build_examples(data_cfg)
    if not examples:
        raise ValueError("No QA examples could be constructed from the provided data.")

    split_examples = _train_val_test_split(
        examples, data_cfg.train_ratio, data_cfg.val_ratio, seed
    )

    datasets = {}
    for split_name, split_data in split_examples.items():
        if not split_data:
            continue
        datasets[split_name] = _examples_to_dataset(split_data)

    return DatasetDict(datasets)


def create_tokenized_datasets(
    raw_datasets: DatasetDict,
    tokenizer,
    data_cfg: DictConfig,
) -> Tuple[DatasetDict, DatasetDict]:
    """Tokenize datasets for question answering using Hugging Face tooling."""

    max_length = data_cfg.max_context_length
    stride = data_cfg.stride
    padding = data_cfg.padding

    column_names = raw_datasets["train"].column_names

    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=padding,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                sequence_ids = tokenized_examples.sequence_ids(i)
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

        tokenized_examples["start_positions"] = start_positions
        tokenized_examples["end_positions"] = end_positions
        return tokenized_examples

    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=padding,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    train_dataset = raw_datasets["train"].map(
        prepare_train_features,
        batched=True,
        remove_columns=column_names,
    )

    eval_examples = raw_datasets["validation"]
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        remove_columns=column_names,
    )

    processed = DatasetDict(
        train=train_dataset,
        validation=eval_dataset,
    )

    extra_examples = DatasetDict(validation=eval_examples)

    test_dataset = None
    if "test" in raw_datasets:
        test_examples = raw_datasets["test"]
        test_dataset = test_examples.map(
            prepare_validation_features,
            batched=True,
            remove_columns=column_names,
        )
        processed["test"] = test_dataset
        extra_examples["test"] = test_examples

    return processed, extra_examples
