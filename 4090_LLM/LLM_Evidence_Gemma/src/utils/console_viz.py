"""Console-friendly visualization helpers for training progress."""

from typing import Dict, List, Sequence
import math


_SPARKLINE_CHARS = ".:-=+*#%@"


def _safe_get(values: Sequence[float], idx: int) -> float:
    if idx < 0 or idx >= len(values):
        return math.nan
    return values[idx]


def _format_value(value: float, precision: int = 4) -> str:
    if value is None or isinstance(value, str):
        return "-"
    if math.isnan(value) or math.isinf(value):
        return "-"
    return f"{value:.{precision}f}"


def _sparkline(values: Sequence[float]) -> str:
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    if not numeric_values:
        return ""

    v_min = min(numeric_values)
    v_max = max(numeric_values)
    if math.isclose(v_min, v_max):
        return _SPARKLINE_CHARS[len(_SPARKLINE_CHARS) // 2] * len(values)

    scale = len(_SPARKLINE_CHARS) - 1
    spark_chars: List[str] = []
    for value in values:
        if not isinstance(value, (int, float)) or math.isnan(value):
            spark_chars.append(" ")
            continue
        normalized = (value - v_min) / (v_max - v_min)
        idx = min(scale, max(0, int(round(normalized * scale))))
        spark_chars.append(_SPARKLINE_CHARS[idx])
    return "".join(spark_chars)


def render_training_progress(history: Dict[str, List[float]], last_n: int = 6) -> str:
    """Return a formatted table plus sparklines summarizing recent progress."""
    if not history or not history.get('train_loss'):
        return ""

    epoch_count = len(history['train_loss'])
    start_idx = max(0, epoch_count - last_n)
    header = (
        f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
        f"{'Val EM':>8} | {'Val F1':>8}"
    )

    rows = []
    for idx in range(start_idx, epoch_count):
        train_loss = _format_value(_safe_get(history.get('train_loss', []), idx))
        val_loss = _format_value(_safe_get(history.get('val_loss', []), idx))
        val_em = _format_value(_safe_get(history.get('val_em', []), idx))
        val_f1 = _format_value(_safe_get(history.get('val_f1', []), idx))
        rows.append(
            f"{idx + 1:>5} | {train_loss:>10} | {val_loss:>10} | "
            f"{val_em:>8} | {val_f1:>8}"
        )

    spark_train = _sparkline(history.get('train_loss', [])[-last_n:])
    spark_val = _sparkline(history.get('val_f1', [])[-last_n:])

    spark_section = ""
    if spark_train:
        spark_section += f"Train loss trend : {spark_train}\n"
    if spark_val:
        spark_section += f"Val F1 trend    : {spark_val}\n"

    separator = "-" * len(header)
    lines = [
        separator,
        header,
        separator,
        *rows,
        separator,
    ]
    if spark_section:
        lines.append(spark_section.rstrip())
    return "\n".join(lines)
