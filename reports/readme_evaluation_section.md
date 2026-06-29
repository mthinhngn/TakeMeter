## Evaluation Report

The baseline and fine-tuned model were evaluated on the same held-out test split.

### Overall Metrics

| Model | Accuracy |
|---|---:|
| Groq zero-shot baseline | 0.972 |
| Fine-tuned DistilBERT | 0.917 |

### Per-Class Metrics

| Model | Label | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Baseline | grounded_advice | 0.923 | 1.000 | 0.960 |
| Baseline | reactive_noise | 1.000 | 1.000 | 1.000 |
| Baseline | unsupported_take | 1.000 | 0.917 | 0.957 |
| Fine-tuned | grounded_advice | 0.909 | 0.833 | 0.870 |
| Fine-tuned | reactive_noise | 1.000 | 1.000 | 1.000 |
| Fine-tuned | unsupported_take | 0.846 | 0.917 | 0.880 |

### Confusion Matrix

Rows are true labels and columns are fine-tuned model predictions.

| True \\ Predicted | grounded_advice | reactive_noise | unsupported_take |
|---|---:|---:|---:|
| grounded_advice | 10 | 0 | 2 |
| reactive_noise | 0 | 12 | 0 |
| unsupported_take | 1 | 0 | 11 |

### AI-Assisted Wrong-Prediction Pattern Review

AI-assisted pattern review input was generated from the wrong predictions. Before finalizing this section, paste `reports/ai_wrong_prediction_prompt.md` into Claude or another LLM, then verify or reject the suggested patterns by re-reading the examples.

Initial mechanical pattern check: 3 wrong predictions; most common label confusions: grounded_advice -> unsupported_take (2), unsupported_take -> grounded_advice (1).
Wrong-post length range: 15-32 words; 0 wrong examples are very short (12 words or fewer).
0 wrong examples have confidence at or above 0.70, which may indicate the model learned a misleading shortcut rather than just being uncertain.

Verified pattern notes: TODO: after using the AI prompt, write which patterns were real, which were discarded, and what you saw when re-reading the examples.

### Wrong Predictions

| Text excerpt | True label | Predicted label | Confidence | Analysis |
|---|---|---|---:|---|
| the dining situation is the only choice that makes sense. This is one strong negative experience, not a representative survey of every di... | unsupported_take | grounded_advice | 0.374 | TODO: explain the confused label boundary, whether the example is ambiguous or underrepresented, and what data/spec change would help. |
| The comparison matters because clark Kerr dining A detailed positive review praised Clark Kerr for dishes such as salmon, ribs, pasta, pi... | grounded_advice | unsupported_take | 0.390 | TODO: explain the confused label boundary, whether the example is ambiguous or underrepresented, and what data/spec change would help. |
| The practical takeaway seems to be that other students disagree, calling some Crossroads meals bland. | grounded_advice | unsupported_take | 0.371 | TODO: explain the confused label boundary, whether the example is ambiguous or underrepresented, and what data/spec change would help. |

### Sample Classifications

| Text excerpt | True label | Predicted label | Confidence | Correct? |
|---|---|---|---:|---|
| The comparison matters because practical takeaway The useful student advice is to inspect the daily menu and maintain alternatives instea... | grounded_advice | grounded_advice | 0.429 | True |
| campus food proves Berkeley does not have it together. Desserts and sandwiches Desserts could be good, while the build-your-own sandwich ... | unsupported_take | unsupported_take | 0.415 | True |
| The practical takeaway seems to be that someone living next to Cafe 3 may value proximity more than another student values a particular C... | grounded_advice | grounded_advice | 0.415 | True |
| the dining situation is the only choice that makes sense. This is one strong negative experience, not a representative survey of every di... | unsupported_take | grounded_advice | 0.374 | False |
| The comparison matters because clark Kerr dining A detailed positive review praised Clark Kerr for dishes such as salmon, ribs, pasta, pi... | grounded_advice | unsupported_take | 0.390 | False |

Correct example explanation: TODO: explain why this correct prediction is reasonable using the label definition.

### Reflection

TODO: explain what the model captured versus what the taxonomy intended. Name one repeated failure pattern and whether it looks like a data distribution issue, a label-boundary issue, or annotation inconsistency.