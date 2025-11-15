from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.augment.evidence import EvidenceReplacer


def test_exact_span_replacement() -> None:
    post_text = "Intro sentence. Target evidence lives here. Outro."
    evidence = "Target evidence lives here."
    replacer = EvidenceReplacer()

    match = replacer.locate(post_text, evidence)
    assert match is not None
    assert match.match_type == "exact"
    replacement = "Augmented proof arrives."

    new_post = replacer.replace(post_text, match, replacement)
    assert new_post == post_text.replace(evidence, replacement, 1)
    # Ensure prefix and suffix remain unchanged byte-for-byte.
    assert new_post[: match.start] == post_text[: match.start]
    assert new_post[match.start + len(replacement) :] == post_text[match.end :]


def test_fuzzy_match_and_skip() -> None:
    post_text = "One line. The quick brown fox leaps over a log. Final line."
    evidence = "quick brown fox jumps over a log"
    replacer = EvidenceReplacer()

    match = replacer.locate(post_text, evidence)
    assert match is not None
    assert match.match_type == "fuzzy"

    new_text = "evidence replaced"
    replaced = replacer.replace(post_text, match, new_text)
    assert new_text in replaced
    assert post_text[: match.start] == replaced[: match.start]
    assert post_text[match.end :] == replaced[match.start + len(new_text) :]

    # Similarity below threshold should force a skip.
    assert replacer.locate(post_text, "unrelated sentence about cats") is None
