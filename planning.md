# Show What You Know: TakeMeter Planning Specification

## Community Choice

This project builds **TakeMeter**, a fine-tuned text classifier that evaluates
the quality of takes in one selected online community. The goal is not generic
topic classification. The labels should capture what counts as a good, weak,
reactionary, insightful, overconfident, or otherwise meaningful take in the
specific discourse norms of the chosen community.

The community should be narrow enough that members share recognizable norms, but
broad enough to collect at least 200 labeled examples.

**Final community:** TODO: replace with the exact subreddit/forum/community used.

**Reasoning:** TODO: explain why this community produces useful, labelable posts
and why the classification task matters. This should be more than naming the
community: describe the recurring opinion style, disagreements, or quality
signals that make TakeMeter useful there.

## Label Taxonomy

The final taxonomy must contain 2-4 labels. Labels should describe discourse
quality, not incidental topic words. A good taxonomy should make it possible to
explain why one post is a strong take and another is noise, exaggeration, or a
low-effort reaction within this community.

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

**Hardest anticipated edge case:** TODO: name one boundary that will be hard to
label, such as evidence-backed criticism versus low-effort negativity, funny
hyperbole versus empty noise, or an insightful unpopular opinion versus a bad
take. State the rule that will resolve that boundary.

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

**Good enough threshold:** TODO: state the concrete target before final tuning.
Suggested default: the fine-tuned model should beat the Groq baseline on the
same test split and reach at least 0.65 accuracy, with no class F1 below 0.45.
If the dataset is especially ambiguous, revise this threshold before training
and explain why.

## Anticipated Challenges

- Short posts may lack context and be hard to label consistently.
- Some labels may overlap if the taxonomy is based on tone rather than intent.
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
