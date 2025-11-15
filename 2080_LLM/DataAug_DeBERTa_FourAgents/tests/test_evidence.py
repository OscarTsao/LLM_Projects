from pathlib import Path

from src.evidence.train_pairclf import run_training
from src.evidence.infer_pairclf import run_inference
from src.utils.data import load_dataset

DATA_PATH = Path("data/redsm5_sample.jsonl")


def test_training_creates_artifacts(tmp_path):
    dataset = list(load_dataset(DATA_PATH))
    info = run_training(dataset, output_dir=tmp_path, dry_run=False)

    assert (tmp_path / "model.ckpt").exists()
    assert (tmp_path / "config.yaml").exists()
    metrics_path = Path(info["metrics_path"])
    metrics = metrics_path.read_text()
    assert "evidence_macro_f1_present" in metrics


def test_inference_produces_predictions(tmp_path):
    dataset = list(load_dataset(DATA_PATH))
    train_info = run_training(dataset, output_dir=tmp_path / "train")
    preds_info = run_inference(
        dataset,
        checkpoint_path=Path(train_info["model_path"]),
        output_path=tmp_path / "predictions.jsonl",
    )

    preds_path = Path(preds_info["output_path"])
    content = preds_path.read_text().strip().splitlines()
    assert len(content) >= 2
    assert "eu_id" in content[0]
