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
