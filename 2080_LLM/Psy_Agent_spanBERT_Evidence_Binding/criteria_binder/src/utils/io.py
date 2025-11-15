# File: src/utils/io.py
"""File I/O utilities for JSONL and configuration handling."""

import json
import yaml
import numpy as np
from typing import Any, Dict, Iterator, Union, Optional
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy data types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """Write list of dictionaries to JSONL file with support for numpy types.

    Args:
        data: List of dictionaries to write
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, cls=NumpyEncoder) + '\n')


def iter_jsonl(file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Iterate over JSONL file without loading all into memory.

    Args:
        file_path: Path to JSONL file

    Yields:
        Parsed JSON objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed configuration dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def safe_mkdir(path: Union[str, Path]) -> Path:
    """Create directory safely, including parents.

    Args:
        path: Directory path to create

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON object
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """Save data to JSON file with support for numpy types.

    Args:
        data: Data to save (dictionary, list, etc.)
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def merge_configs(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """(Deprecated) Recursively merge configuration dictionaries."""
    raise NotImplementedError(
        "Manual config merging is deprecated. Use Hydra overrides instead."
    )


def parse_config_overrides(overrides):  # type: ignore[override]
    """(Deprecated) Parse command line config overrides."""
    raise NotImplementedError(
        "Manual config overrides are deprecated. Use Hydra overrides instead."
    )