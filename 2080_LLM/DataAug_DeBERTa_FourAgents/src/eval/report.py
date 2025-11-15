from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def write_metrics(metrics: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

