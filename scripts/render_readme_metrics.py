"""Render README-ready Markdown tables from TakeMeter evaluation outputs.

Usage:
    python scripts/render_readme_metrics.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPORTS_DIR = Path("reports")
RESULTS_PATH = REPORTS_DIR / "evaluation_results.json"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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
                    precision=fmt(metrics["precision"]),
                    recall=fmt(metrics["recall"]),
                    f1=fmt(metrics["f1-score"]),
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


def load_predictions() -> list[dict[str, str]]:
    if not PREDICTIONS_PATH.exists():
        return []
    with PREDICTIONS_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def excerpt(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_samples(predictions: list[dict[str, str]]) -> str:
    lines = [
        "| Text excerpt | True label | Predicted label | Confidence | Correct? |",
        "|---|---|---|---:|---|",
    ]
    selected = predictions[:5]
    for row in selected:
        lines.append(
            "| {text} | {true} | {pred} | {confidence} | {correct} |".format(
                text=excerpt(row["text"]).replace("|", "\\|"),
                true=row["true_label"],
                pred=row["finetuned_prediction"],
                confidence=fmt(float(row["finetuned_confidence"])),
                correct=row["is_correct"],
            )
        )
    return "\n".join(lines)


def render_wrong_predictions(predictions: list[dict[str, str]]) -> str:
    lines = [
        "| Text excerpt | True label | Predicted label | Confidence | Analysis |",
        "|---|---|---|---:|---|",
    ]
    wrong = [row for row in predictions if row["is_correct"].lower() == "false"][:3]
    for row in wrong:
        lines.append(
            "| {text} | {true} | {pred} | {confidence} | TODO: explain why this was hard. |".format(
                text=excerpt(row["text"]).replace("|", "\\|"),
                true=row["true_label"],
                pred=row["finetuned_prediction"],
                confidence=fmt(float(row["finetuned_confidence"])),
            )
        )
    return "\n".join(lines)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Missing {RESULTS_PATH}. Run the Colab workflow first.")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    predictions = load_predictions()

    print("\n## Overall Metrics\n")
    print(render_overall(results))
    print("\n## Per-Class Metrics\n")
    print(render_per_class(results))
    print("\n## Confusion Matrix\n")
    print(render_confusion_matrix(results))
    if predictions:
        print("\n## Wrong Predictions\n")
        print(render_wrong_predictions(predictions))
        print("\n## Sample Classifications\n")
        print(render_samples(predictions))


if __name__ == "__main__":
    main()
