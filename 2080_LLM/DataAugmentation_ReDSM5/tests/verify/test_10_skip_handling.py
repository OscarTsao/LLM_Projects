"""Test evidence skip handling."""
from tests.verify_utils import load_fixture

def test_fixture_has_unskippable():
    """Fixture row 11 has evidence not in post_text."""
    df = load_fixture()
    row11 = df.iloc[10]  # 0-indexed
    assert row11["evidence"] not in row11["post_text"], "Row 11 should be unskippable"
