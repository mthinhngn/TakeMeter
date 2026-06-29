# -*- coding: utf-8 -*-
"""AI201 Project 3: TakeMeter fine-tuning workflow.

This file is designed to run in Google Colab for model training, while still
supporting a local preflight check before upload.

Colab:
    1. Runtime -> Change runtime type -> T4 GPU.
    2. Add GROQ_API_KEY in Colab Secrets.
    3. Run all cells / run this script and upload the labeled CSV when prompted.

Local preflight:
    python ai201_project3_takemeter_starter_clean.py --preflight data/takemeter_labeled.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


MIN_EXAMPLES = 200
MIN_LABELS = 2
MAX_LABELS = 4
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path("reports")
MODEL_OUTPUT_DIR = Path("takemeter-model")


def running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def ensure_runtime_dependencies() -> None:
    """Install Colab dependencies if needed."""
    packages = ["groq", "python-dotenv", "datasets", "transformers", "accelerate"]
    missing = []
    for package in packages:
        module_name = package.replace("-", "_")
        if package == "python-dotenv":
            module_name = "dotenv"
        try:
            __import__(module_name)
        except Exception:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def normalize_columns(df):
    """Normalize common dataset column names to the required text/label schema."""
    rename_map = {}
    lower_to_original = {col.strip().lower(): col for col in df.columns}
    text_candidates = ["text", "post", "comment", "content", "body", "description"]
    label_candidates = ["label", "category", "class", "take_type", "stance"]

    if "text" not in df.columns:
        for candidate in text_candidates:
            if candidate in lower_to_original:
                rename_map[lower_to_original[candidate]] = "text"
                break
    if "label" not in df.columns:
        for candidate in label_candidates:
            if candidate in lower_to_original:
                rename_map[lower_to_original[candidate]] = "label"
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def clean_label(value: object) -> str:
    label = str(value).strip().lower()
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"[^a-z0-9_]+", "", label)
    return label


def load_and_validate_dataset(csv_path: str | Path):
    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df = normalize_columns(df)
    required = {"text", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Dataset must include columns {sorted(required)}. Missing: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].map(clean_label)
    df = df[(df["text"] != "") & (df["label"] != "")]

    labels = sorted(df["label"].unique().tolist())
    if len(df) < MIN_EXAMPLES:
        raise ValueError(f"Need at least {MIN_EXAMPLES} usable examples; found {len(df)}.")
    if not (MIN_LABELS <= len(labels) <= MAX_LABELS):
        raise ValueError(f"Need {MIN_LABELS}-{MAX_LABELS} labels; found {len(labels)}: {labels}")
    label_share = df["label"].value_counts(normalize=True)
    largest_label = label_share.index[0]
    largest_share = float(label_share.iloc[0])
    if largest_share > 0.70:
        raise ValueError(
            f"Label imbalance too high: '{largest_label}' is {largest_share:.1%} of the dataset. "
            "No single label may account for more than 70%."
        )

    label_map = {label: idx for idx, label in enumerate(labels)}
    df["label_id"] = df["label"].map(label_map).astype(int)
    return df, label_map


def representative_examples(df, label: str, count: int = 2) -> list[str]:
    examples = (
        df[df["label"] == label]["text"]
        .dropna()
        .astype(str)
        .map(lambda text: re.sub(r"\s+", " ", text).strip())
        .loc[lambda series: series.str.len() > 0]
        .head(count)
        .tolist()
    )
    return examples


def infer_label_definition(label: str) -> str:
    readable = label.replace("_", " ")
    return (
        f"Posts whose take quality, reasoning style, or discourse value matches "
        f"the '{readable}' category from the annotated TakeMeter dataset."
    )


def build_system_prompt(df, label_map: dict[str, int], community: str) -> str:
    lines = [
        f"You are classifying posts from {community}.",
        "Assign each post to exactly one TakeMeter label based on the quality of the take.",
        "Use the community context, label definitions, and examples below.",
        "",
    ]
    for label in label_map:
        examples = representative_examples(df, label, count=2)
        lines.append(f"{label}: {infer_label_definition(label)}")
        for idx, example in enumerate(examples, start=1):
            lines.append(f'Example {idx}: "{example[:350]}"')
        lines.append("")
    lines.extend(
        [
            "Respond with ONLY one valid label name.",
            "Do not explain your reasoning.",
            "",
            "Valid labels:",
            *label_map.keys(),
        ]
    )
    return "\n".join(lines)


def print_preflight_report(df, label_map: dict[str, int], system_prompt: str) -> None:
    print("=" * 60)
    print("TAKEMETER DATASET PREFLIGHT")
    print("=" * 60)
    print(f"Examples: {len(df)}")
    print(f"Labels: {label_map}")
    print()
    print("Label distribution:")
    print(df["label"].value_counts().to_string())
    print()
    print(f"Prompt length: {len(system_prompt)} characters")
    print("First 900 prompt characters:")
    print(system_prompt[:900])
    print("=" * 60)


def prepare_splits(df):
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def train_and_evaluate(csv_path: str | Path, community: str) -> None:
    ensure_runtime_dependencies()

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from groq import Groq
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    df, label_map = load_and_validate_dataset(csv_path)
    id_to_label = {idx: label for label, idx in label_map.items()}
    label_names = [id_to_label[i] for i in range(len(label_map))]
    system_prompt = build_system_prompt(df, label_map, community)
    print_preflight_report(df, label_map, system_prompt)

    train_df, val_df, test_df = prepare_splits(df)
    print(f"Train: {len(train_df)} Validation: {len(val_df)} Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    def make_dataset(split_df):
        dataset = Dataset.from_pandas(
            split_df[["text", "label_id"]].rename(columns={"label_id": "labels"})
        )
        return dataset.map(tokenize, batched=True)

    train_dataset = make_dataset(train_df)
    val_dataset = make_dataset(val_df)
    test_dataset = make_dataset(test_df)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_map),
        id2label=id_to_label,
        label2id=label_map,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    ft_output = trainer.predict(test_dataset)
    ft_pred_ids = np.argmax(ft_output.predictions, axis=-1)
    ft_true_ids = ft_output.label_ids
    ft_probs = torch.nn.functional.softmax(torch.tensor(ft_output.predictions), dim=-1).numpy()
    ft_accuracy = accuracy_score(ft_true_ids, ft_pred_ids)
    ft_report = classification_report(
        ft_true_ids, ft_pred_ids, target_names=label_names, output_dict=True, zero_division=0
    )
    ft_report_text = classification_report(
        ft_true_ids, ft_pred_ids, target_names=label_names, zero_division=0
    )
    print("\nFine-tuned model report:")
    print(ft_report_text)

    cm = confusion_matrix(ft_true_ids, ft_pred_ids)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=label_names,
        yticklabels=label_names,
        title="Fine-Tuned Model Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    fig.tight_layout()
    confusion_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=150)
    plt.show()

    groq_key = os.environ.get("GROQ_API_KEY")
    if running_in_colab():
        try:
            from google.colab import userdata  # type: ignore

            groq_key = userdata.get("GROQ_API_KEY") or groq_key
        except Exception:
            pass
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is required for the baseline. Use Colab Secrets.")

    client = Groq(api_key=groq_key)

    def classify_with_groq(text: str) -> str | None:
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this post:\n\n{text}"},
                ],
                temperature=0,
                max_tokens=20,
            )
            raw = response.choices[0].message.content.strip().lower()
            for label in sorted(label_map, key=len, reverse=True):
                if raw == label or label in raw:
                    return label
            return None
        except Exception as exc:
            print(f"Groq API error: {exc}")
            return None

    print(f"Running Groq baseline on {len(test_df)} examples...")
    baseline_preds = []
    for i, (_, row) in enumerate(test_df.iterrows(), start=1):
        baseline_preds.append(classify_with_groq(row["text"]))
        if i % 10 == 0:
            print(f"  {i}/{len(test_df)} complete")
        time.sleep(0.1)

    valid = [(pred, true_id) for pred, true_id in zip(baseline_preds, test_df["label_id"]) if pred]
    bl_pred_ids = [label_map[pred] for pred, _ in valid]
    bl_true_ids = [true_id for _, true_id in valid]
    bl_accuracy = accuracy_score(bl_true_ids, bl_pred_ids) if valid else 0.0
    bl_report = classification_report(
        bl_true_ids,
        bl_pred_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    print("\nBaseline report:")
    print(classification_report(bl_true_ids, bl_pred_ids, target_names=label_names, zero_division=0))

    predictions = []
    for idx, row in test_df.iterrows():
        true_id = int(row["label_id"])
        pred_id = int(ft_pred_ids[idx])
        confidence = float(ft_probs[idx][pred_id])
        predictions.append(
            {
                "text": row["text"],
                "true_label": id_to_label[true_id],
                "finetuned_prediction": id_to_label[pred_id],
                "finetuned_confidence": round(confidence, 4),
                "baseline_prediction": baseline_preds[idx],
                "is_correct": bool(true_id == pred_id),
            }
        )

    results = {
        "community": community,
        "label_map": label_map,
        "model": MODEL_NAME,
        "test_set_size": len(test_df),
        "baseline_accuracy": round(float(bl_accuracy), 4),
        "baseline_parseable": len(valid),
        "finetuned_accuracy": round(float(ft_accuracy), 4),
        "improvement": round(float(ft_accuracy - bl_accuracy), 4),
        "finetuned_classification_report": ft_report,
        "baseline_classification_report": bl_report,
        "confusion_matrix": cm.tolist(),
        "labels": label_names,
        "system_prompt": system_prompt,
    }

    (OUTPUT_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(predictions).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)
    print("\nFiles ready:")
    print(f"  {OUTPUT_DIR / 'evaluation_results.json'}")
    print(f"  {OUTPUT_DIR / 'predictions.csv'}")
    print(f"  {OUTPUT_DIR / 'test_split.csv'}")
    print(f"  {confusion_path}")


def get_colab_csv_path() -> str:
    from google.colab import files  # type: ignore

    print("Select your labeled TakeMeter dataset CSV...")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No CSV uploaded.")
    return list(uploaded.keys())[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="TakeMeter Project 3 workflow")
    parser.add_argument("--preflight", help="Validate a labeled CSV without training.")
    parser.add_argument("--csv", help="Labeled CSV path for full training/evaluation.")
    parser.add_argument(
        "--community",
        default=os.environ.get("TAKEMETER_COMMUNITY", "the selected online community"),
        help="Community name used in the Groq baseline prompt.",
    )
    args, _ = parser.parse_known_args()

    if args.preflight:
        df, label_map = load_and_validate_dataset(args.preflight)
        prompt = build_system_prompt(df, label_map, args.community)
        print_preflight_report(df, label_map, prompt)
        return

    csv_path = args.csv or os.environ.get("TAKEMETER_CSV_PATH")
    if not csv_path and Path("data/takemeter_labeled.csv").exists():
        csv_path = "data/takemeter_labeled.csv"
    if not csv_path and running_in_colab():
        csv_path = get_colab_csv_path()
    if not csv_path:
        raise SystemExit(
            "No dataset found. Add data/takemeter_labeled.csv, pass --csv, "
            "or run in Colab and upload the CSV."
        )
    train_and_evaluate(csv_path, args.community)


if __name__ == "__main__":
    main()
