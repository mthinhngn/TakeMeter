# Show What You Know: TakeMeter

This repository contains the AI201 Project 3 TakeMeter submission package:
a labeled dataset, a DistilBERT fine-tuning workflow, a Groq zero-shot baseline,
evaluation artifacts, and the final report.

> Status: implementation scaffold is complete. Final metrics require the real
> `data/takemeter_labeled.csv` and a Colab run with `GROQ_API_KEY`.

## Rubric Coverage Checklist

| Rubric area | Where covered |
|---|---|
| Label taxonomy: 2-4 labels, complete definitions, 2 examples each, clear boundaries | `Label Taxonomy` |
| Annotated dataset: 200+ examples, source/process/distribution, hard cases, no label over 70% | `Dataset` |
| Fine-tuning pipeline: base model, platform, justified hyperparameter decision | `Fine-Tuning Approach` |
| Baseline comparison: prompt and same-test-set metrics | `Baseline`, `Evaluation Report` |
| Error analysis: per-class metrics, confusion matrix, 3 wrong predictions, reflection | `Evaluation Report`, `Reflection` |
| `planning.md`: community reasoning, labels/edge case, data/eval plan, good-enough threshold, AI plan | `planning.md` |
| Demo video: 3-5 classifications, one correct, one incorrect, report walkthrough | `Demo Video` |
| AI usage and spec reflection | `AI Usage`, `Spec Reflection` |

## Community Choice And Reasoning

TODO: Name the selected community and explain why it is a good source for this
TakeMeter task. Include what kinds of opinions, arguments, reactions, and
community-specific norms appear there. Explain why the selected labels measure
the quality of takes in this community rather than merely sorting posts by topic.

## Label Taxonomy

Replace this table after adding the dataset. The completed notebook infers label
names from `data/takemeter_labeled.csv`, but the README must contain human-written
definitions and two examples per label. Each label should describe discourse
quality in context, such as whether a take is evidence-backed, hyperbolic,
low-effort, insightful, reactionary, or otherwise meaningful for the chosen
community.

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

Boundary note: TODO: explain the clearest difference between the most easily
confused labels so two readers would likely label most examples the same way.

## Dataset

The labeled dataset should be committed as:

```text
data/takemeter_labeled.csv
```

Required columns:

| Column | Description |
|---|---|
| `text` | The post or comment text to classify. |
| `label` | One exact label from the taxonomy. |

After the dataset is added, run:

```powershell
python ai201_project3_takemeter_starter_clean.py --preflight data/takemeter_labeled.csv --community "TODO community name"
```

For the full GPU run, open `takemeter_colab.ipynb` in Google Colab. That
notebook downloads the completed workflow from this repository, prompts for the
CSV upload, and exports the report artifacts.

After downloading the Colab outputs into `reports/`, run:

```powershell
python scripts/render_readme_metrics.py
```

That command creates:

- `reports/readme_evaluation_section.md` - Markdown tables and analysis prompts
  to paste into this README.
- `reports/ai_wrong_prediction_prompt.md` - the prompt to paste into Claude or
  another LLM before writing the final wrong-prediction analysis.
- `demo/DEMO_SCRIPT.md` - a 3-5 minute recording outline using real prediction
  examples.

### Label Distribution

TODO: paste the label distribution from the preflight output.

| Label | Count |
|---|---:|
| TODO | TODO |

No single label may account for more than 70% of the dataset. The preflight
script fails if this balance requirement is violated.

### Labeling Process

TODO: Describe where examples came from, how they were selected, how labels were
assigned, and any cleanup performed. Explain how ambiguous cases were resolved
when topic, tone, and take quality pointed in different directions.

Data source: TODO.

Labeling process: TODO.

### Difficult Examples

| Text excerpt | Possible labels | Final label | Decision rationale |
|---|---|---|---|
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

## Fine-Tuning Approach

The fine-tuned classifier uses `distilbert-base-uncased` with a sequence
classification head. The workflow uses a stratified 70/15/15 split for
train/validation/test and is intended to run on Google Colab with a T4 GPU.

Training defaults:

| Setting | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Epochs | 3 |
| Learning rate | `2e-5` |
| Train batch size | 16 |
| Eval batch size | 32 |
| Weight decay | `0.01` |
| Max token length | 256 |

Hyperparameter decision: 3 epochs is a conservative choice for a small dataset.
It gives the model multiple passes over the examples while limiting overfitting.
This is a deliberate training decision, not just an unchanged default: if
validation accuracy flattens or drops after early epochs, longer training would
be more likely to memorize community-specific keywords than learn label
boundaries.

## Baseline

The baseline uses Groq `llama-3.3-70b-versatile` with temperature `0`. The prompt
is generated from the community name, labels, and representative examples, and it
instructs the model to return only one valid label.

Baseline results are collected on the exact same held-out test split as the
fine-tuned model so the comparison is fair.

The exact prompt used during the run is saved in:

```text
reports/evaluation_results.json
```

## Evaluation Report

TODO: Run the Colab workflow and replace this section with real values.

```powershell
python ai201_project3_takemeter_starter_clean.py --csv data/takemeter_labeled.csv --community "TODO community name"
```

In Colab, add `GROQ_API_KEY` through Secrets before running the baseline.

Good enough threshold from `planning.md`: TODO: paste the pre-training threshold
and state whether the final fine-tuned model met it.

### Overall Metrics

| Model | Accuracy |
|---|---:|
| Groq zero-shot baseline | TODO |
| Fine-tuned DistilBERT | TODO |

### Per-Class Metrics

At least one per-class metric for the fine-tuned model is required; this table
reports precision, recall, and F1 for each label.

| Model | Label | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Baseline | TODO | TODO | TODO | TODO |
| Fine-tuned | TODO | TODO | TODO | TODO |

### Confusion Matrix

Rows are true labels and columns are fine-tuned model predictions.

| True \\ Predicted | TODO |
|---|---:|
| TODO | TODO |

The image version is saved at `reports/confusion_matrix.png`.

### AI-Assisted Wrong-Prediction Pattern Review

TODO: Before finalizing this section, paste
`reports/ai_wrong_prediction_prompt.md` into Claude or another LLM and ask it to
surface patterns in the misclassified examples. Then re-read the examples
yourself. Keep the patterns that are actually supported, and say which suggested
patterns you corrected or discarded.

### Wrong Predictions

| Text excerpt | True label | Predicted label | Confidence | Analysis |
|---|---|---|---:|---|
| TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO |

Each analysis should tie the error to the data, a label boundary, or model
behavior. Do not write only that the model got it wrong.

### Sample Classifications

| Text excerpt | True label | Predicted label | Confidence | Correct? |
|---|---|---|---:|---|
| TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO |

Correct example explanation: TODO.

## Reflection

TODO: Explain what the model learned compared with what the taxonomy intended.
Discuss whether it learned meaningful quality signals or shortcuts such as
keywords, phrasing, topic references, hot-button names, or class imbalance.

Specific failure pattern: TODO: name one repeated failure pattern, such as a
particular label pair, a post type, or a distribution issue.

## Spec Reflection

One way the spec helped: TODO.

One way implementation diverged from the spec and why: TODO.

## AI Usage

1. Codex was directed to convert the Colab starter into a complete workflow with
   dataset validation, label-map inference, fine-tuning, Groq baseline evaluation,
   and export files. Human review still needs to provide the final dataset and
   verify that inferred labels match the intended taxonomy.
2. Codex was directed to create the submission documentation scaffold from the
   course rubric. The final README must be revised after training with real
   metrics, difficult examples, prediction analysis, and reflection.

Annotation assistance disclosure: TODO: state whether AI helped label examples.

## Demo Video

TODO: Add a link to a 3-5 minute demo video showing:

- 3-5 posts classified by the fine-tuned model with label and confidence visible;
- one correct prediction explained;
- one incorrect prediction explained;
- a brief walkthrough of this evaluation report.

If the file is too large for GitHub, link a shareable video URL here.

Evaluation report summary for demo: TODO: briefly surface the key metrics and
most important error pattern.

## Repository Checklist

- [ ] `planning.md` in repo root.
- [ ] `README.md` with all rubric sections completed.
- [ ] `data/takemeter_labeled.csv` or a dataset link in this README.
- [ ] Completed Colab workflow script/notebook.
- [ ] `reports/evaluation_results.json`.
- [ ] `reports/confusion_matrix.png`.
- [ ] `reports/predictions.csv`.
- [ ] Demo video or accessible demo link.
- [ ] No API keys or secrets committed.
