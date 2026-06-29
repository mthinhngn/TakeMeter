# Show What You Know: TakeMeter Planning Specification

## Community Choice

This project builds **TakeMeter**, a fine-tuned text classifier that evaluates
the quality of takes in one selected online community. The goal is not generic
topic classification. The labels should capture what counts as a good, weak,
reactionary, insightful, overconfident, or otherwise meaningful take in the
specific discourse norms of the chosen community.

The community should be narrow enough that members share recognizable norms, but
broad enough to collect at least 200 labeled examples.

**Final community:** `r/berkeley`.

**Reasoning:** `r/berkeley` is active, text-heavy, and full of practical
student opinions about housing, dining, classes, safety, registration, social
life, and campus policy. It is a good fit for TakeMeter because the same topic
can produce very different kinds of takes: some posts explain tradeoffs from
experience, some make confident claims without support, and some are mostly
immediate venting or joking. Those distinctions matter because students often
use the community to make real decisions, so the quality of the reasoning is
more important than whether the post is positive or negative.

## Label Taxonomy

The final taxonomy must contain 2-4 labels. Labels should describe discourse
quality, not incidental topic words. A good taxonomy should make it possible to
explain why one post is a strong take and another is noise, exaggeration, or a
low-effort reaction within this community.

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|
| `grounded_advice` | The post gives a clear recommendation or judgment supported by specific personal experience, concrete details, comparisons, or reasoning that another student could use. | "If you are choosing between Blackwell and Unit 1, Blackwell is newer and quieter, but Unit 1 made it easier for me to meet people during the first month." | "For CS 61B, I would not take it with two other heavy technical classes because the projects expand near deadlines even if the weekly lectures feel manageable." |
| `unsupported_take` | The post makes a broad or confident claim about Berkeley life without enough evidence, context, or reasoning to justify the strength of the opinion. | "All the dining halls are trash and anyone saying otherwise is coping." | "Foothill is objectively the worst dorm because it is far from everything." |
| `reactive_noise` | The post is mainly an emotional reaction, joke, complaint, or hype response with little transferable information or argument. | "Enrollment time again, I hate it here." | "The Wi-Fi died during my quiz, this campus is unserious." |

**Hardest anticipated edge case:** The hardest boundary is between
`grounded_advice` and `unsupported_take` when a post has a strong opinion plus
one concrete detail. Decision rule: if the detail would still help another
student make a decision after removing the emotional wording, label it
`grounded_advice`; if the detail is vague, cherry-picked, or only decorative,
label it `unsupported_take`. Example: "Cafe 3 is awful because the rice was
undercooked twice this week" is borderline, but it becomes `grounded_advice` if
the post compares options or explains when/why the issue happens; by itself it
is closer to `unsupported_take` because it generalizes from a narrow complaint.

## Data Collection Plan

- Collect at least 200 examples from the selected community.
- Store the dataset as `data/takemeter_labeled.csv`.
- Required columns: `text`, `label`.
- Optional useful columns: `source_url`, `post_id`, `created_at`, `notes`.
- Remove usernames, private information, deleted content, and unrelated boilerplate.
- Keep labels exactly consistent with the taxonomy.
- Keep the dataset balanced enough that no single label is more than 70% of all
  examples.

## Labeling Process

Each example will be assigned exactly one label. Ambiguous examples should be
resolved by the label that best captures the quality of the take in context, not
just the topic being discussed. The final README will document three difficult
examples, the chosen label, and why another label was rejected.

## Modeling Plan

The fine-tuned model will use `distilbert-base-uncased` with a sequence
classification head. The default training setup is 3 epochs, learning rate
`2e-5`, batch size 16, weight decay `0.01`, and a stratified 70/15/15
train/validation/test split.

The hyperparameter choice to report is the conservative 3-epoch setup: it gives
the small dataset enough passes to learn label patterns while reducing the risk
of overfitting compared with longer training.

## Baseline Plan

The baseline will use Groq `llama-3.3-70b-versatile` in a zero-shot prompt. The
prompt will describe the community, define each take-quality label, include two
examples per label when available, and strictly require one valid label as the
output.

## Evaluation Plan

The report will compare the Groq baseline against the fine-tuned DistilBERT
model on the same held-out test set using:

- overall accuracy for both models;
- per-class precision, recall, and F1;
- a confusion matrix written as a Markdown table in `README.md`;
- three wrong predictions with analysis;
- three to five sample classifications with label and confidence.

Accuracy is useful because each example receives one label. Per-class precision,
recall, and F1 are necessary because a good overall score could hide failure on
a smaller label. The confusion matrix will show which label boundaries are most
fragile.

**Good enough threshold:** The fine-tuned model should beat the Groq baseline on
the same test split, reach at least 0.65 accuracy, and keep each class F1 at or
above 0.45. This threshold is realistic for a small 200-example dataset while
still requiring the model to learn more than the majority label or obvious
keywords.

## Anticipated Challenges

- Short posts may lack context and be hard to label consistently.
- `grounded_advice` and `unsupported_take` may overlap when a post mixes a
  useful detail with exaggerated framing.
- Label imbalance may make minority classes harder for the model to learn.
- The Groq baseline may output extra text unless the prompt strongly constrains
  the response format.

## AI Usage Plan

Codex will be used to complete the notebook workflow, build validation and
reporting helpers, and draft the submission documentation. AI may also be used
for label stress-testing, such as asking whether two definitions are too
overlapping or whether a difficult example exposes a weak boundary. Human review
will provide the dataset, verify the taxonomy, revise misleading examples, and
ensure the README honestly reflects the collected results.
