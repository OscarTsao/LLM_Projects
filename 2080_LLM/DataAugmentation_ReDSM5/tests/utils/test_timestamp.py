from __future__ import annotations

import re

from src.utils.timestamp import utc_timestamp


def test_utc_timestamp_format() -> None:
    stamp = utc_timestamp()
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)
