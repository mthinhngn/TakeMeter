# Show What You Know: TakeMeter

This repository contains the AI201 Project 3 TakeMeter submission package:
a labeled dataset, a DistilBERT fine-tuning workflow, a Groq zero-shot baseline,
evaluation artifacts, and the final report.

> Status: community and label design are complete. Final metrics require the
> real `data/takemeter_labeled.csv` and a Colab run with `GROQ_API_KEY`.

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

This project studies `r/berkeley`, a public student community where people ask
for advice, vent, compare experiences, and argue about campus life. I chose this
community because many posts are not simply "about" one topic; they differ in
how useful the take is. A housing post can be grounded advice, a dining post can
be an unsupported generalization, and an enrollment post can be mostly a
reaction. The taxonomy therefore measures discourse quality: whether a post
gives transferable reasoning, asserts a broad claim without support, or mainly
expresses emotion/noise.

## Label Taxonomy

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|
| `grounded_advice` | The post gives a clear recommendation or judgment supported by specific personal experience, concrete details, comparisons, or reasoning that another student could use. | "If you are choosing between Blackwell and Unit 1, Blackwell is newer and quieter, but Unit 1 made it easier for me to meet people during the first month." | "For CS 61B, I would not take it with two other heavy technical classes because the projects expand near deadlines even if the weekly lectures feel manageable." |
| `unsupported_take` | The post makes a broad or confident claim about Berkeley life without enough evidence, context, or reasoning to justify the strength of the opinion. | "All the dining halls are trash and anyone saying otherwise is coping." | "Foothill is objectively the worst dorm because it is far from everything." |
| `reactive_noise` | The post is mainly an emotional reaction, joke, complaint, or hype response with little transferable information or argument. | "Enrollment time again, I hate it here." | "The Wi-Fi died during my quiz, this campus is unserious." |

Boundary note: the most easily confused labels are `grounded_advice` and
`unsupported_take`. If a post includes a specific detail that would help another
student make a decision even after removing the emotional wording, I label it
`grounded_advice`. If the detail is vague, isolated, or only there to make a
broad claim sound stronger, I label it `unsupported_take`.

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

TODO: paste the label distribution from the preflight output after
`data/takemeter_labeled.csv` is added.

| Label | Count |
|---|---:|
| `grounded_advice` | TODO |
| `unsupported_take` | TODO |
| `reactive_noise` | TODO |

No single label may account for more than 70% of the dataset. The preflight
script fails if this balance requirement is violated.

### Labeling Process

Examples will be collected from public `r/berkeley` posts and comments about
student life, including housing, dining, classes, enrollment, campus services,
and everyday frustrations. I will exclude usernames, private information,
deleted text, and unrelated boilerplate. Each example receives exactly one label
based on the quality of the take, not the topic or whether I agree with it.
Ambiguous cases will use the boundary rule above: useful transferable detail
beats emotional framing, but isolated details used to support a sweeping claim
remain `unsupported_take`.

Data source: public `r/berkeley` posts and comments.

Labeling process: manually label each text example using the three definitions,
then review difficult cases for consistency before training.

### Difficult Examples

| Text excerpt | Possible labels | Final label | Decision rationale |
|---|---|---|---|
| "Cafe 3 is awful because the rice was undercooked twice this week." | `grounded_advice`, `unsupported_take` | `unsupported_take` | The post gives one detail, but it makes a broad judgment without comparison, timing, or advice. |
| "Unit 1 is more social than Blackwell, but Blackwell's rooms and bathrooms are nicer, so choose based on whether comfort or meeting people matters more." | `grounded_advice`, `unsupported_take` | `grounded_advice` | The post gives a comparison and decision rule another student can use. |
| "I hate enrollment so much, this school is impossible." | `unsupported_take`, `reactive_noise` | `reactive_noise` | The post is mostly emotional venting and does not make a stable claim about Berkeley. |

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
