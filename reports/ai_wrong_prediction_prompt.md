# AI Wrong-Prediction Pattern Review Prompt

Paste this into Claude or another LLM before writing the final README error analysis.

After it suggests patterns, verify them yourself by re-reading the examples. Keep only patterns that are actually supported.



Task: Identify common themes in these misclassified TakeMeter examples. Look for repeated label pairs, sarcasm, short or low-information posts, topic words overriding structure, confidence patterns, and possible annotation inconsistencies. Give me 3-5 candidate patterns and tell me what evidence supports each. Also list any patterns that are weak or speculative.



Labels: grounded_advice, reactive_noise, unsupported_take

Fine-tuned accuracy: 0.9167



Example 1
Text: the dining situation is the only choice that makes sense. This is one strong negative experience, not a representative survey of every diner.
True label: unsupported_take
Predicted label: grounded_advice
Confidence: 0.3742

Example 2
Text: The comparison matters because clark Kerr dining A detailed positive review praised Clark Kerr for dishes such as salmon, ribs, pasta, pizza, macaroni and cheese, its salad bar, outdoor seating, and atmosphere.
True label: grounded_advice
Predicted label: unsupported_take
Confidence: 0.3903

Example 3
Text: The practical takeaway seems to be that other students disagree, calling some Crossroads meals bland.
True label: grounded_advice
Predicted label: unsupported_take
Confidence: 0.371