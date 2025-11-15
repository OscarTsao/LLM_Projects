from __future__ import annotations

import pandas as pd
from hydra import compose, initialize

from criteriabind.config_schemas import AppConfig, parse_config
from criteriabind.data.real_loader import load_samples


def test_csv_loader(tmp_path):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "note": "Patient reports fatigue and low mood.",
                "criterion": "Depressed mood most days",
                "definition": "Depressed mood most days for two weeks",
                "split": "train",
            },
            {
                "sample_id": "s1",
                "note": "Patient reports fatigue and low mood.",
                "criterion": "Loss of interest in activities",
                "definition": "Loss of interest for two weeks",
                "split": "train",
            },
            {
                "sample_id": "s2",
                "note": "Patient sleeps well and has stable appetite.",
                "criterion": "Insomnia or hypersomnia",
                "definition": "Sleep disturbance (insomnia or hypersomnia)",
                "split": "validation",
            },
        ]
    )
    df.to_csv(train_csv, index=False)

    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.path={data_dir.as_posix()}",
                "data.source=csv",
                "+data.path_or_name=train.csv",
                "+data.mapping.sample_id=sample_id",
                "+data.mapping.note=note",
                "+data.mapping.criterion_name=criterion",
                "+data.mapping.criterion_definition=definition",
                "+data.mapping.split=split",
            ],
        )

    app_cfg: AppConfig = parse_config(cfg)
    samples = load_samples(app_cfg, "train")
    assert len(samples) == 1
    sample = samples[0]
    assert sample.id == "s1"
    assert sample.split == "train"
    assert len(sample.criteria) == 2
    names = {criterion.name for criterion in sample.criteria}
    assert "Depressed mood most days" in names
    assert "Loss of interest in activities" in names
