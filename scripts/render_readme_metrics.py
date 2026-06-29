"""Render README-ready TakeMeter evaluation material.

Run this after the Colab workflow has produced:
    reports/evaluation_results.json
    reports/predictions.csv

It writes:
    reports/readme_evaluation_section.md
    reports/ai_wrong_prediction_prompt.md
    demo/DEMO_SCRIPT.md

Usage:
    python scripts/render_readme_metrics.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


REPORTS_DIR = Path("reports")
DEMO_DIR = Path("demo")
RESULTS_PATH = REPORTS_DIR / "evaluation_results.json"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"
README_SECTION_PATH = REPORTS_DIR / "readme_evaluation_section.md"
AI_PROMPT_PATH = REPORTS_DIR / "ai_wrong_prediction_prompt.md"
DEMO_SCRIPT_PATH = DEMO_DIR / "DEMO_SCRIPT.md"


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def excerpt(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def load_predictions() -> list[dict[str, str]]:
    if not PREDICTIONS_PATH.exists():
        return []
    with PREDICTIONS_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def is_correct(row: dict[str, str]) -> bool:
    return row.get("is_correct", "").strip().lower() == "true"


def wrong_predictions(predictions: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in predictions if not is_correct(row)]


def correct_predictions(predictions: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in predictions if is_correct(row)]


def render_overall(results: dict) -> str:
    return "\n".join(
        [
            "| Model | Accuracy |",
            "|---|---:|",
            f"| Groq zero-shot baseline | {fmt(results['baseline_accuracy'])} |",
            f"| Fine-tuned DistilBERT | {fmt(results['finetuned_accuracy'])} |",
        ]
    )


def render_per_class(results: dict) -> str:
    lines = [
        "| Model | Label | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for model_name, key in [
        ("Baseline", "baseline_classification_report"),
        ("Fine-tuned", "finetuned_classification_report"),
    ]:
        report = results[key]
        for label in results["labels"]:
            metrics = report[label]
            lines.append(
                "| {model} | {label} | {precision} | {recall} | {f1} |".format(
                    model=model_name,
                    label=label,
                    precision=fmt(float(metrics["precision"])),
                    recall=fmt(float(metrics["recall"])),
                    f1=fmt(float(metrics["f1-score"])),
                )
            )
    return "\n".join(lines)


def render_confusion_matrix(results: dict) -> str:
    labels = results["labels"]
    matrix = results["confusion_matrix"]
    lines = [
        "| True \\\\ Predicted | " + " | ".join(labels) + " |",
        "|---|" + "|".join(["---:"] * len(labels)) + "|",
    ]
    for label, row in zip(labels, matrix):
        lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def label_pair_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(
        f"{row['true_label']} -> {row['finetuned_prediction']}" for row in rows
    )


def render_ai_pattern_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "The fine-tuned model had no wrong predictions in `predictions.csv`."

    pairs = label_pair_counts(rows)
    lengths = [len(row["text"].split()) for row in rows]
    confidences = [float(row["finetuned_confidence"]) for row in rows]
    short_count = sum(1 for length in lengths if length <= 12)
    high_conf_wrong = sum(1 for conf in confidences if conf >= 0.70)
    top_pairs = ", ".join(f"{pair} ({count})" for pair, count in pairs.most_common(3))

    return "\n".join(
        [
            "AI-assisted pattern review input was generated from the wrong predictions. "
            "Before finalizing this section, paste `reports/ai_wrong_prediction_prompt.md` "
            "into Claude or another LLM, then verify or reject the suggested patterns by "
            "re-reading the examples.",
            "",
            f"Initial mechanical pattern check: {len(rows)} wrong predictions; most common label confusions: {top_pairs}.",
            f"Wrong-post length range: {min(lengths)}-{max(lengths)} words; {short_count} wrong examples are very short (12 words or fewer).",
            f"{high_conf_wrong} wrong examples have confidence at or above 0.70, which may indicate the model learned a misleading shortcut rather than just being uncertain.",
            "",
            "Verified pattern notes: after using the AI prompt, write which patterns were real, which were discarded, and what you saw when re-reading the examples.",
        ]
    )


def render_wrong_predictions(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Text excerpt | True label | Predicted label | Confidence | Analysis |",
        "|---|---|---|---:|---|",
    ]
    for row in rows[:3]:
        lines.append(
            "| {text} | {true} | {pred} | {confidence} | Explain the confused label boundary, whether the example is ambiguous or underrepresented, and what data/spec change would help. |".format(
                text=markdown_escape(excerpt(row["text"])),
                true=row["true_label"],
                pred=row["finetuned_prediction"],
                confidence=fmt(float(row["finetuned_confidence"])),
            )
        )
    while len(lines) < 5:
        lines.append("| Fill in after rerun | Fill in after rerun | Fill in after rerun | Fill in after rerun | Fill in after rerun |")
    return "\n".join(lines)


def choose_sample_rows(predictions: list[dict[str, str]]) -> list[dict[str, str]]:
    correct = correct_predictions(predictions)
    wrong = wrong_predictions(predictions)
    selected = correct[:3] + wrong[:2]
    if len(selected) < 5:
        selected += [row for row in predictions if row not in selected][: 5 - len(selected)]
    return selected[:5]


def render_samples(predictions: list[dict[str, str]]) -> str:
    lines = [
        "| Text excerpt | True label | Predicted label | Confidence | Correct? |",
        "|---|---|---|---:|---|",
    ]
    for row in choose_sample_rows(predictions):
        lines.append(
            "| {text} | {true} | {pred} | {confidence} | {correct} |".format(
                text=markdown_escape(excerpt(row["text"])),
                true=row["true_label"],
                pred=row["finetuned_prediction"],
                confidence=fmt(float(row["finetuned_confidence"])),
                correct=row["is_correct"],
            )
        )
    return "\n".join(lines)


def render_ai_prompt(results: dict, rows: list[dict[str, str]]) -> str:
    listed = rows[:15]
    examples = []
    for idx, row in enumerate(listed, start=1):
        examples.append(
            "\n".join(
                [
                    f"Example {idx}",
                    f"Text: {row['text']}",
                    f"True label: {row['true_label']}",
                    f"Predicted label: {row['finetuned_prediction']}",
                    f"Confidence: {row['finetuned_confidence']}",
                ]
            )
        )

    return "\n\n".join(
        [
            "# AI Wrong-Prediction Pattern Review Prompt",
            "Paste this into Claude or another LLM before writing the final README error analysis.",
            "After it suggests patterns, verify them yourself by re-reading the examples. Keep only patterns that are actually supported.",
            "",
            "Task: Identify common themes in these misclassified TakeMeter examples. Look for repeated label pairs, sarcasm, short or low-information posts, topic words overriding structure, confidence patterns, and possible annotation inconsistencies. Give me 3-5 candidate patterns and tell me what evidence supports each. Also list any patterns that are weak or speculative.",
            "",
            f"Labels: {', '.join(results['labels'])}",
            f"Fine-tuned accuracy: {results['finetuned_accuracy']}",
            "",
            "\n\n".join(examples) if examples else "No wrong predictions were found.",
        ]
    )


def render_readme_section(results: dict, predictions: list[dict[str, str]]) -> str:
    wrong = wrong_predictions(predictions)
    sample_rows = choose_sample_rows(predictions)
    first_correct = next((row for row in sample_rows if is_correct(row)), None)
    correct_note = (
        "Explain why this correct prediction is reasonable using the label definition."
        if first_correct
        else "No correct sample was available in `predictions.csv`."
    )

    return "\n\n".join(
        [
            "## Evaluation Report",
            "The baseline and fine-tuned model were evaluated on the same held-out test split.",
            "### Overall Metrics",
            render_overall(results),
            "### Per-Class Metrics",
            render_per_class(results),
            "### Confusion Matrix",
            "Rows are true labels and columns are fine-tuned model predictions.",
            render_confusion_matrix(results),
            "### AI-Assisted Wrong-Prediction Pattern Review",
            render_ai_pattern_summary(wrong),
            "### Wrong Predictions",
            render_wrong_predictions(wrong),
            "### Sample Classifications",
            render_samples(predictions),
            f"Correct example explanation: {correct_note}",
            "### Reflection",
            "Explain what the model captured versus what the taxonomy intended. Name one repeated failure pattern and whether it looks like a data distribution issue, a label-boundary issue, or annotation inconsistency.",
        ]
    )


def render_demo_script(results: dict, predictions: list[dict[str, str]]) -> str:
    rows = choose_sample_rows(predictions)
    correct = next((row for row in rows if is_correct(row)), None)
    wrong = next((row for row in rows if not is_correct(row)), None)
    lines = [
        "# TakeMeter Demo Script",
        "",
        "Target length: 3-5 minutes.",
        "",
        "## 1. Setup",
        "",
        f"Show the repo and say: This is TakeMeter, a DistilBERT classifier for take quality. The fine-tuned model accuracy was {results['finetuned_accuracy']}, compared with the Groq baseline at {results['baseline_accuracy']}.",
        "",
        "## 2. Show 3-5 Classifications",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"{idx}. Text: {excerpt(row['text'], 180)}",
                f"   Prediction: {row['finetuned_prediction']} with confidence {row['finetuned_confidence']}.",
                f"   True label: {row['true_label']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## 3. Narrate One Correct Prediction",
            "",
            (
                f"Use this one: \"{excerpt(correct['text'], 180)}\". Explain why `{correct['finetuned_prediction']}` matches the label definition."
                if correct
                else "Choose one correct prediction after rerunning the model."
            ),
            "",
            "## 4. Narrate One Incorrect Prediction",
            "",
            (
                f"Use this one: \"{excerpt(wrong['text'], 180)}\". The model predicted `{wrong['finetuned_prediction']}` but the true label was `{wrong['true_label']}`. Explain the boundary it missed."
                if wrong
                else "Choose one incorrect prediction after rerunning the model."
            ),
            "",
            "## 5. Walk Through Evaluation Report",
            "",
            "Show the README evaluation section: overall metrics, per-class metrics, confusion matrix, AI-assisted pattern review, wrong-prediction analysis, and reflection.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Missing {RESULTS_PATH}. Run the Colab workflow first.")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    predictions = load_predictions()
    if not predictions:
        raise SystemExit(f"Missing or empty {PREDICTIONS_PATH}. Run the Colab workflow first.")

    REPORTS_DIR.mkdir(exist_ok=True)
    DEMO_DIR.mkdir(exist_ok=True)
    wrong = wrong_predictions(predictions)

    README_SECTION_PATH.write_text(render_readme_section(results, predictions), encoding="utf-8")
    AI_PROMPT_PATH.write_text(render_ai_prompt(results, wrong), encoding="utf-8")
    DEMO_SCRIPT_PATH.write_text(render_demo_script(results, predictions), encoding="utf-8")

    print(f"Wrote {README_SECTION_PATH}")
    print(f"Wrote {AI_PROMPT_PATH}")
    print(f"Wrote {DEMO_SCRIPT_PATH}")
    print()
    print(README_SECTION_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
