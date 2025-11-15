#!/usr/bin/env python3
"""
Builds a post–criteria pairwise dataset for ReDSM5 by merging
`data/redsm5/redsm5_posts.csv` and `data/redsm5/redsm5_annotations.csv`.

Output: `data/redsm5_post_criteria_pairs.csv`

Rules:
- For every `post_id` and each of the 10 criteria, emit one row.
- If there is at least one positive (status=1) annotation for a criterion,
  set `status=1` and set `evidence_sentence` to the concatenation of all
  positive evidence sentences for that pair (in file order), joined by " || ".
- If there are no positive annotations for a criterion, set `status=0` and
  `evidence_sentence` to an empty string, even if there are explicit negative
  (status=0) annotations.
"""

from __future__ import annotations

import csv
import pathlib
from collections import defaultdict, OrderedDict


CRITERIA_ORDER = [
    "DEPRESSED_MOOD",
    "ANHEDONIA",
    "APPETITE_CHANGE",
    "SLEEP_ISSUES",
    "PSYCHOMOTOR",
    "FATIGUE",
    "WORTHLESSNESS",
    "COGNITIVE_ISSUES",
    "SUICIDAL_THOUGHTS",
    "SPECIAL_CASE",
]


def read_posts(posts_path: pathlib.Path) -> OrderedDict[str, str]:
    posts: "OrderedDict[str, str]" = OrderedDict()
    with posts_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post_id = row["post_id"].strip()
            text = row["text"]
            posts[post_id] = text
    return posts


def read_annotations(ann_path: pathlib.Path):
    # Map: (post_id, symptom) -> list of annotation rows (preserving order)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with ann_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post_id = row["post_id"].strip()
            symptom = row["DSM5_symptom"].strip()
            groups[(post_id, symptom)].append(row)
    return groups


def build_pairs(posts, ann_groups, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "post_id",
        "DSM5_symptom",
        "status",
        "evidence_sentence",
        "text",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for post_id, text in posts.items():
            for symptom in CRITERIA_ORDER:
                rows = ann_groups.get((post_id, symptom), [])
                # Gather positives in order, de-duplicated preserving order
                positives: list[str] = []
                seen = set()
                for r in rows:
                    status_val = str(r.get("status", "")).strip()
                    if status_val == "1":
                        sent = r.get("sentence_text", "")
                        if sent not in seen:
                            positives.append(sent)
                            seen.add(sent)

                status = 1 if positives else 0
                evidence_sentence = " || ".join(positives) if positives else ""

                writer.writerow(
                    {
                        "post_id": post_id,
                        "DSM5_symptom": symptom,
                        "status": status,
                        "evidence_sentence": evidence_sentence,
                        "text": text,
                    }
                )


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    posts_path = root / "data" / "redsm5" / "redsm5_posts.csv"
    ann_path = root / "data" / "redsm5" / "redsm5_annotations.csv"
    out_path = root / "data" / "redsm5_post_criteria_pairs.csv"

    posts = read_posts(posts_path)
    ann_groups = read_annotations(ann_path)
    build_pairs(posts, ann_groups, out_path)


if __name__ == "__main__":
    main()

