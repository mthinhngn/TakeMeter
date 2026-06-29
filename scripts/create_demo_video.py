"""Create a simple captioned TakeMeter demo video from evaluation outputs."""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEMO_DIR = ROOT / "demo"
VIDEO_PATH = DEMO_DIR / "takemeter-demo.mp4"


def load_rows() -> list[dict[str, str]]:
    with (REPORTS_DIR / "sample_classifications.csv").open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def font(size: int):
    for name in ["arial.ttf", "segoeui.ttf", "calibri.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_slide(title: str, bullets: list[str], footer: str = "") -> Image.Image:
    image = Image.new("RGB", (1280, 720), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title_font = font(42)
    body_font = font(28)
    small_font = font(22)
    draw.rectangle((0, 0, 1280, 88), fill="#12313f")
    draw.text((48, 24), title, fill="white", font=title_font)
    y = 126
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=78)
        if not wrapped:
            y += 18
            continue
        draw.text((64, y), "- " + wrapped[0], fill="#111827", font=body_font)
        y += 38
        for line in wrapped[1:]:
            draw.text((94, y), line, fill="#111827", font=body_font)
            y += 36
        y += 18
    if footer:
        draw.text((48, 666), footer, fill="#475569", font=small_font)
    return image


def main() -> None:
    import imageio.v2 as imageio
    import numpy as np

    DEMO_DIR.mkdir(exist_ok=True)
    results = json.loads((REPORTS_DIR / "evaluation_results.json").read_text(encoding="utf-8"))
    rows = load_rows()
    correct = next(row for row in rows if row["is_correct"].lower() == "true")
    wrong = next(row for row in rows if row["is_correct"].lower() == "false")

    slides: list[tuple[Image.Image, int]] = []
    slides.append(
        (
            draw_slide(
                "TakeMeter Demo",
                [
                    "Community: r/berkeley student-life discussions.",
                    "Task: classify take quality as grounded_advice, unsupported_take, or reactive_noise.",
                    f"Fine-tuned DistilBERT accuracy: {results['finetuned_accuracy']:.3f}. Groq baseline accuracy: {results['baseline_accuracy']:.3f}.",
                ],
                "Generated demo video with captions.",
            ),
            18,
        )
    )
    for idx, row in enumerate(rows[:5], start=1):
        slides.append(
            (
                draw_slide(
                    f"Classification {idx}",
                    [
                        f"Post: {row['text']}",
                        f"Predicted label: {row['finetuned_prediction']}",
                        f"Confidence: {float(row['finetuned_confidence']):.3f}",
                        f"True label: {row['true_label']}; correct: {row['is_correct']}",
                    ],
                ),
                22,
            )
        )
    slides.append(
        (
            draw_slide(
                "Correct Prediction",
                [
                    f"Post: {correct['text']}",
                    f"The model predicted {correct['finetuned_prediction']} with confidence {float(correct['finetuned_confidence']):.3f}.",
                    "This is reasonable because the post gives transferable advice another student could use.",
                ],
            ),
            24,
        )
    )
    slides.append(
        (
            draw_slide(
                "Incorrect Prediction",
                [
                    f"Post: {wrong['text']}",
                    f"The model predicted {wrong['finetuned_prediction']}, but the true label was {wrong['true_label']}.",
                    "The error is the grounded_advice versus unsupported_take boundary: careful wording or concrete details can confuse the model.",
                ],
            ),
            24,
        )
    )
    slides.append(
        (
            draw_slide(
                "Evaluation Summary",
                [
                    "Reactive_noise was easy: the fine-tuned model got every reactive_noise test example correct.",
                    "The main failure pattern was grounded_advice versus unsupported_take.",
                    "The model captured surface phrasing, but it did not always learn the intended question: does this help another student make a decision?",
                ],
            ),
            24,
        )
    )

    with imageio.get_writer(VIDEO_PATH, fps=1, codec="libx264", quality=7) as writer:
        for slide, duration in slides:
            frame = np.asarray(slide)
            for _ in range(duration):
                writer.append_data(frame)
    print(f"Wrote {VIDEO_PATH}")


if __name__ == "__main__":
    main()
