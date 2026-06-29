# TakeMeter Planning Specification

## Community Choice

I chose `r/berkeley`, a public student community where people ask for advice,
compare experiences, vent, and debate campus life. The community is a good fit
for TakeMeter because students often use it for real decisions about housing,
dining, classes, and campus logistics. A useful classifier should not simply
detect the topic of a post; it should distinguish whether the post gives
transferable reasoning, makes an unsupported claim, or mostly expresses a
reaction.

## Label Taxonomy

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|
| `grounded_advice` | The post gives a clear recommendation or judgment supported by specific personal experience, concrete details, comparisons, or reasoning that another student could use. | "If you are choosing between Blackwell and Unit 1, Blackwell is newer and quieter, but Unit 1 made it easier for me to meet people during the first month." | "For CS 61B, I would not take it with two other heavy technical classes because the projects expand near deadlines even if the weekly lectures feel manageable." |
| `unsupported_take` | The post makes a broad or confident claim about Berkeley life without enough evidence, context, or reasoning to justify the strength of the opinion. | "All the dining halls are trash and anyone saying otherwise is coping." | "Foothill is objectively the worst dorm because it is far from everything." |
| `reactive_noise` | The post is mainly an emotional reaction, joke, complaint, or hype response with little transferable information or argument. | "Enrollment time again, I hate it here." | "The Wi-Fi died during my quiz, this campus is unserious." |

The hardest anticipated edge case is `grounded_advice` versus
`unsupported_take`. Some posts contain one concrete detail inside a broad
emotional claim. My decision rule is: if the detail would still help another
student make a decision after removing the emotional wording, label it
`grounded_advice`; if the detail is vague, isolated, cherry-picked, or only
decorative support for a sweeping claim, label it `unsupported_take`.

## Data Collection And Annotation Plan

The planned dataset format is a CSV at `data/takemeter_labeled.csv` with at
least `text` and `label` columns. Optional provenance columns such as
`source_note` are allowed. The goal is at least 200 examples with no single label
over 70% of the dataset.

Direct Reddit scraping was attempted later but blocked from the local
environment, so the final implementation used existing local notes summarizing
public `r/berkeley` discussions. Those notes preserve source URLs and paraphrase
public thread content without storing usernames or raw thread archives. I then
generated short post-like examples from those notes and labeled them using the
taxonomy above. This is disclosed in the README because it is not the same as a
fresh scrape of 200 verbatim comments.

Annotation rules:

- Label the quality of the take, not whether I agree with it.
- Prefer `grounded_advice` when the post gives concrete comparison, context, or
  decision guidance.
- Prefer `unsupported_take` when the post makes a strong claim without enough
  support.
- Prefer `reactive_noise` when the post is mainly a vent, joke, or immediate
  emotional response.

## Modeling Plan

The fine-tuned model is `distilbert-base-uncased` with a sequence classification
head. The planned split is stratified 70/15/15 train/validation/test. The main
training configuration is 3 epochs, learning rate `2e-5`, train batch size 16,
eval batch size 32, and max token length 256.

The key hyperparameter decision is to keep training at 3 epochs. With a small
dataset, more epochs could overfit to repeated phrasing in the examples instead
of learning the intended label boundary.

## Baseline Plan

The baseline is Groq `llama-3.3-70b-versatile` with temperature `0`. The prompt
will name `r/berkeley`, define each label, include two examples per label, and
require the model to output only one valid label. The baseline and fine-tuned
model will be evaluated on the same held-out test split.

## Evaluation Plan

I will report overall accuracy for both models, per-class precision/recall/F1
for both models, a confusion matrix for the fine-tuned model, at least three
wrong predictions with analysis, and 3-5 sample classifications with confidence.

Accuracy is useful because each example has one label. Per-class metrics are
necessary because a high overall score could hide failure on a smaller or harder
label. The confusion matrix is especially important because the expected failure
is a boundary problem between `grounded_advice` and `unsupported_take`.

Good-enough threshold: the fine-tuned model should reach at least 0.65 accuracy
and keep every class F1 at or above 0.45. Ideally it should beat the Groq
baseline, but if it does not, the report should explain what the baseline did
better and what the fine-tuned model failed to learn.

## AI Tool Plan

I planned to use AI tools for implementation support, label stress-testing, and
failure-pattern analysis. In practice, Codex helped build the workflow, generate
the dataset from existing notes, run the model, and write the report structure.
Groq was also used as a second reviewer for wrong predictions so I could compare
its suggested patterns with my own rereading of the examples.
