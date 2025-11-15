from __future__ import annotations

from pathlib import Path
import json

from criteriabind.cli import prefs_to_dpo
from criteriabind.schemas import Candidate, CriterionSpec, Judgment, Preference


def test_prefs_to_dpo_conversion(tmp_path: Path, monkeypatch) -> None:
    judgment_path = tmp_path / "judgments.jsonl"
    output_path = tmp_path / "dpo.jsonl"

    judgment = Judgment(
        job_id="job-1",
        note_id="note-1",
        criterion_id="crit-1",
        criterion=CriterionSpec(id="crit-1", name="Criterion A", definition="Criterion A definition"),
        criterion_text="Criterion A definition",
        note_text="Patient reports persistent low mood.",
        candidates=[
            Candidate(text="Low mood recorded over two weeks."),
            Candidate(text="Patient enjoys hobbies."),
        ],
        best_idx=0,
        preferences=[Preference(winner_idx=0, loser_idx=1)],
        provider="mock",
    )
    judgment_path.write_text(f"{judgment.to_json()}\n", encoding="utf-8")

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
        "seed": 0,
        "input_path": judgment_path.as_posix(),
        "output_path": output_path.as_posix(),
        "sample_rate": 1.0,
        "max_pairs": None,
        "shuffle": False,
    }
    )

    prefs_to_dpo.main.__wrapped__(cfg)  # type: ignore[arg-type]

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["prompt"].startswith("Criterion ID: crit-1")
    assert record["chosen"] == "Low mood recorded over two weeks."
    assert record["rejected"] == "Patient enjoys hobbies."
