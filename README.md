# Show What You Know: TakeMeter

TakeMeter is a text classifier for evaluating the quality of takes in `r/berkeley`
student-life discussions. It compares a fine-tuned DistilBERT classifier against
a Groq zero-shot baseline on the same held-out test set.

## Community Choice And Reasoning

I chose `r/berkeley` because it is a public, text-heavy student community where
people regularly ask for advice, compare experiences, vent, and argue about
campus life. The same topic can produce very different kinds of discourse: a
housing post can give transferable advice, a dining post can make a sweeping
unsupported claim, and an enrollment post can mostly be an emotional reaction.
That makes the community a good fit for TakeMeter because the task is not topic
classification; it is judging whether a take contains useful reasoning.

## Label Taxonomy

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|
| `grounded_advice` | The post gives a clear recommendation or judgment supported by specific personal experience, concrete details, comparisons, or reasoning that another student could use. | "If you are choosing between Blackwell and Unit 1, Blackwell is newer and quieter, but Unit 1 made it easier for me to meet people during the first month." | "For CS 61B, I would not take it with two other heavy technical classes because the projects expand near deadlines even if the weekly lectures feel manageable." |
| `unsupported_take` | The post makes a broad or confident claim about Berkeley life without enough evidence, context, or reasoning to justify the strength of the opinion. | "All the dining halls are trash and anyone saying otherwise is coping." | "Foothill is objectively the worst dorm because it is far from everything." |
| `reactive_noise` | The post is mainly an emotional reaction, joke, complaint, or hype response with little transferable information or argument. | "Enrollment time again, I hate it here." | "The Wi-Fi died during my quiz, this campus is unserious." |

The hardest boundary is `grounded_advice` versus `unsupported_take`. My decision
rule is: if the detail would still help another student make a decision after
removing the emotional wording, label it `grounded_advice`; if the detail is
vague, isolated, or only decorative support for a sweeping claim, label it
`unsupported_take`.

## Dataset

The dataset is included at `data/takemeter_labeled.csv` with 240 labeled
examples and three balanced labels.

| Label | Count |
|---|---:|
| `grounded_advice` | 80 |
| `unsupported_take` | 80 |
| `reactive_noise` | 80 |

No label accounts for more than 70% of the dataset.

### Data Source And Labeling Process

The examples were built from my existing local notes summarizing public
`r/berkeley` discussions about housing and dining. Those notes preserve source
URLs and paraphrase the discussion content rather than storing usernames or raw
thread archives. I converted the notes into short post-like examples and labeled
them using the three-label taxonomy above. This means the dataset is based on
public `r/berkeley` discourse, but it is not a fresh raw scrape of 200 verbatim
Reddit comments.

I used Codex to generate the CSV rows from the notes and then kept the labels
balanced at 80 examples per class. For labeling, the main rule was to assign the
quality of the take rather than the topic. For example, dining content could be
`grounded_advice` if it compared options, `unsupported_take` if it made a broad
claim from one detail, or `reactive_noise` if it was mostly venting.

### Difficult Examples

| Text excerpt | Possible labels | Final label | Decision rationale |
|---|---|---|---|
| "Cafe 3 is awful because the rice was undercooked twice this week." | `grounded_advice`, `unsupported_take` | `unsupported_take` | It gives one detail, but it generalizes too broadly without comparison or decision advice. |
| "Unit 1 is more social than Blackwell, but Blackwell's rooms and bathrooms are nicer, so choose based on whether comfort or meeting people matters more." | `grounded_advice`, `unsupported_take` | `grounded_advice` | It gives a comparison and a usable decision rule. |
| "I hate enrollment so much, this school is impossible." | `unsupported_take`, `reactive_noise` | `reactive_noise` | It is mainly emotional venting, not a stable claim or recommendation. |

## Fine-Tuning Approach

The fine-tuned classifier uses `distilbert-base-uncased` with a sequence
classification head. I trained locally on CPU using the same code intended for
Google Colab/T4. The dataset was split into stratified train/validation/test
sets using a 70/15/15 split.

| Setting | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Training platform | Local Python CPU run |
| Train/validation/test split | 168 / 36 / 36 |
| Epochs | 3 |
| Learning rate | `2e-5` |
| Train batch size | 16 |
| Eval batch size | 32 |
| Weight decay | `0.01` |
| Max token length | 256 |

The main hyperparameter decision was to keep training at 3 epochs. With only
240 examples, longer training would risk memorizing phrasing patterns from the
generated/paraphrased dataset instead of learning the intended label boundary.

## Baseline

The baseline uses Groq `llama-3.3-70b-versatile` with temperature `0`. The
prompt names `r/berkeley`, defines all three labels, gives two examples per
label, and instructs the model to output only a valid label. Baseline predictions
were collected on the exact same 36-example held-out test split as the
fine-tuned model.

## Evaluation Report

The baseline and fine-tuned model were evaluated on the same held-out test split.
My planned "good enough" threshold was at least 0.65 accuracy, no class F1 below
0.45, and ideally beating the baseline. The fine-tuned model exceeded the
absolute accuracy/F1 threshold, but it did not beat the Groq baseline.

### Overall Metrics

| Model | Accuracy |
|---|---:|
| Groq zero-shot baseline | 0.972 |
| Fine-tuned DistilBERT | 0.917 |

### Per-Class Metrics

| Model | Label | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Baseline | `grounded_advice` | 0.923 | 1.000 | 0.960 |
| Baseline | `reactive_noise` | 1.000 | 1.000 | 1.000 |
| Baseline | `unsupported_take` | 1.000 | 0.917 | 0.957 |
| Fine-tuned | `grounded_advice` | 0.909 | 0.833 | 0.870 |
| Fine-tuned | `reactive_noise` | 1.000 | 1.000 | 1.000 |
| Fine-tuned | `unsupported_take` | 0.846 | 0.917 | 0.880 |

### Confusion Matrix

Rows are true labels and columns are fine-tuned model predictions.

| True \ Predicted | `grounded_advice` | `reactive_noise` | `unsupported_take` |
|---|---:|---:|---:|
| `grounded_advice` | 10 | 0 | 2 |
| `reactive_noise` | 0 | 12 | 0 |
| `unsupported_take` | 1 | 0 | 11 |

The image version is committed at `reports/confusion_matrix.png`.

### AI-Assisted Wrong-Prediction Pattern Review

Before writing this analysis, I used Groq as a second AI reviewer. I gave it the
three misclassified examples and asked it to identify common themes. It suggested
that topic words may be overriding structure, that several examples lack a clear
conclusion, and that all errors have low confidence. After rereading the
examples, I kept the `grounded_advice` versus `unsupported_take` boundary and
low-confidence uncertainty patterns. I discarded sarcasm, short-post length, and
`reactive_noise` confusion as explanations because none of the three wrong
examples are sarcastic, very short, or predicted as `reactive_noise`.

### Wrong Predictions

| Text excerpt | True label | Predicted label | Confidence | Analysis |
|---|---|---|---:|---|
| "the dining situation is the only choice that makes sense. This is one strong negative experience, not a representative survey of every diner." | `unsupported_take` | `grounded_advice` | 0.374 | The model probably focused on the phrase "not a representative survey," which sounds careful and analytical. The label is still `unsupported_take` because the opening claim is broad and overconfident. This boundary would improve with more examples where caveats appear inside otherwise unsupported claims. |
| "The comparison matters because Clark Kerr dining... praised Clark Kerr for dishes such as salmon, ribs, pasta, pizza..." | `grounded_advice` | `unsupported_take` | 0.390 | This example contains many food nouns but the useful signal is that it reports concrete details from a positive review. The model may have learned that broad dining praise often belongs to `unsupported_take`. More diverse `grounded_advice` examples with positive details would help. |
| "The practical takeaway seems to be that other students disagree, calling some Crossroads meals bland." | `grounded_advice` | `unsupported_take` | 0.371 | The post is short and reports disagreement rather than giving a full recommendation. I kept it as `grounded_advice` because the useful point is that dining quality varies by reviewer. The model missed that disagreement itself can be useful advice. A tighter definition should say that summarizing disagreement can count as grounded advice when it helps decision-making. |

### Sample Classifications

| Text excerpt | True label | Predicted label | Confidence | Correct? |
|---|---|---|---:|---|
| "The useful student advice is to inspect the daily menu and maintain alternatives instead of assuming every meal at the nearest hall will suit one person's tastes." | `grounded_advice` | `grounded_advice` | 0.429 | Yes |
| "campus food proves Berkeley does not have it together. Desserts and sandwiches could be good..." | `unsupported_take` | `unsupported_take` | 0.415 | Yes |
| "someone living next to Cafe 3 may value proximity more than another student values a particular Crossroads station." | `grounded_advice` | `grounded_advice` | 0.415 | Yes |
| "the dining situation is the only choice that makes sense..." | `unsupported_take` | `grounded_advice` | 0.374 | No |
| "A detailed positive review praised Clark Kerr for dishes such as salmon, ribs, pasta..." | `grounded_advice` | `unsupported_take` | 0.390 | No |

The first correct prediction is reasonable because it gives transferable advice:
check the menu and keep backup options instead of assuming the nearest dining
hall will always work. That is exactly the kind of decision-oriented reasoning
the `grounded_advice` label is meant to capture.

## Reflection

The model captured the easy distinction between `reactive_noise` and the two
more substantive labels. It got every `reactive_noise` test example right,
suggesting that emotional/joke/venting language is a clear surface pattern. The
gap is in the more subtle distinction between a useful take and a confident but
unsupported one. The model sometimes treated careful wording as advice even when
the claim was still broad, and sometimes treated positive concrete detail as
unsupported praise. In other words, it learned some phrasing cues but did not
fully learn the intended decision rule: "does this help another student make a
decision?"

## Spec Reflection

The spec helped by forcing the label boundary before training. Naming the
`grounded_advice` versus `unsupported_take` edge case made the eventual errors
easier to diagnose because the model failed exactly where the spec predicted the
task would be hardest.

The implementation diverged from the ideal data plan because direct Reddit
scraping was blocked from the local environment. Instead of pretending to have a
fresh raw scrape, I built the dataset from existing paraphrased notes of public
`r/berkeley` discussions and disclosed that limitation. This made the project
runnable and inspectable, but it means the dataset is cleaner and more templated
than real Reddit comments would be.

## AI Usage

1. I directed Codex to convert the Colab starter into a complete workflow with
   dataset validation, label-map handling, DistilBERT fine-tuning, Groq baseline
   evaluation, and exported report files. I revised the label definitions so the
   generated prompt matched the `r/berkeley` TakeMeter task instead of generic
   labels.
2. I directed Codex to generate `data/takemeter_labeled.csv` from my existing
   public-discussion notes and keep the classes balanced. I reviewed and
   disclosed this process because the rows are generated/paraphrased examples,
   not raw scraped Reddit comments.
3. I used Groq as an AI reviewer for the wrong predictions. It suggested several
   possible patterns; I kept the supported ones about the
   `grounded_advice`/`unsupported_take` boundary and low-confidence errors, and
   discarded unsupported suggestions about sarcasm and `reactive_noise`.

## Demo Video

The demo video is committed at [`demo/takemeter-demo.mp4`](demo/takemeter-demo.mp4).
It shows five sample classifications with predicted labels and confidence, one
correct prediction explanation, one incorrect prediction explanation, and a short
walkthrough of the evaluation report. The narration/caption outline is in
[`demo/DEMO_SCRIPT.md`](demo/DEMO_SCRIPT.md).

## Repository Checklist

- [x] `planning.md` in repo root.
- [x] `README.md` with required sections.
- [x] `data/takemeter_labeled.csv` included.
- [x] Completed workflow script and Colab runner.
- [x] `reports/evaluation_results.json`.
- [x] `reports/confusion_matrix.png`.
- [x] `reports/predictions.csv`.
- [x] Wrong-prediction AI review artifacts.
- [x] Demo video or accessible demo link.
- [x] No API keys or secrets committed.
