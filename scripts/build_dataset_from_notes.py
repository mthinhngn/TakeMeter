"""Build a TakeMeter CSV from local r/berkeley discussion notes.

The source notes in ../data/raw are paraphrased summaries of public r/berkeley
threads. This script turns them into short post-like examples for the course
classifier and labels them with the project taxonomy.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT.parent / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "takemeter_labeled.csv"


LABELS = ["grounded_advice", "unsupported_take", "reactive_noise"]


def clean_text(text: str) -> str:
    text = re.sub(r"^---.*?---", "", text, flags=re.S)
    text = re.sub(r"^#\s+", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [part.strip() for part in parts if 45 <= len(part.strip()) <= 280]


def topic_from_name(path: Path) -> str:
    return "dining" if path.name.startswith("dining") else "housing"


def grounded_from_sentence(sentence: str, topic: str) -> str:
    prefixes = [
        "A useful way to frame this is that",
        "For someone making a decision, the important detail is that",
        "The practical takeaway seems to be that",
        "I would not treat this as universal, but",
        "The comparison matters because",
    ]
    prefix = prefixes[hash(sentence) % len(prefixes)]
    return f"{prefix} {sentence[0].lower() + sentence[1:]}"


def unsupported_from_sentence(sentence: str, topic: str) -> str:
    subjects = {
        "housing": ["this dorm", "that housing option", "freshman housing", "the residence hall"],
        "dining": ["this dining hall", "campus food", "that meal plan", "the dining situation"],
    }[topic]
    judgments = [
        "is obviously overrated",
        "is honestly terrible",
        "is the only choice that makes sense",
        "is not worth defending",
        "proves Berkeley does not have it together",
    ]
    return f"{subjects[hash(sentence) % len(subjects)]} {judgments[hash(sentence[::-1]) % len(judgments)]}. {sentence}"


def reactive_from_sentence(sentence: str, topic: str) -> str:
    reactions = {
        "housing": [
            "Dorm decisions are stressing me out again.",
            "Why does choosing housing feel like a final exam?",
            "Freshman housing discourse is exhausting.",
            "Every dorm thread makes me more confused.",
            "I am tired of everyone arguing about rooms.",
        ],
        "dining": [
            "Dining hall discourse is so dramatic.",
            "I just wanted dinner, not a campus-wide debate.",
            "Every food thread makes me lose hope.",
            "The dining takes are getting ridiculous again.",
            "I cannot believe we are arguing about rice this much.",
        ],
    }[topic]
    return reactions[hash(sentence) % len(reactions)]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    source_files = sorted(SOURCE_DIR.glob("*.md"))
    all_sentences: list[tuple[str, str, str]] = []
    for path in source_files:
        topic = topic_from_name(path)
        for sentence in sentences(path.read_text(encoding="utf-8")):
            all_sentences.append((sentence, topic, path.name))

    for sentence, topic, source in all_sentences:
        rows.append(
            {
                "text": grounded_from_sentence(sentence, topic),
                "label": "grounded_advice",
                "source_note": source,
            }
        )
        rows.append(
            {
                "text": unsupported_from_sentence(sentence, topic),
                "label": "unsupported_take",
                "source_note": source,
            }
        )
        rows.append(
            {
                "text": reactive_from_sentence(sentence, topic),
                "label": "reactive_noise",
                "source_note": source,
            }
        )

    balanced: list[dict[str, str]] = []
    for label in LABELS:
        label_rows = [row for row in rows if row["label"] == label]
        balanced.extend(label_rows[:80])
    return balanced


def main() -> None:
    rows = build_rows()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label", "source_note"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")
    counts = {label: sum(1 for row in rows if row["label"] == label) for label in LABELS}
    print(counts)


if __name__ == "__main__":
    main()
