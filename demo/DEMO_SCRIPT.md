# TakeMeter Demo Script

Target length: 3-5 minutes.

## 1. Setup

Show the repo and say: This is TakeMeter, a DistilBERT classifier for take quality. The fine-tuned model accuracy was 0.9167, compared with the Groq baseline at 0.9722.

## 2. Show 3-5 Classifications

1. Text: The comparison matters because practical takeaway The useful student advice is to inspect the daily menu and maintain alternatives instead of assuming every meal at the nearest ...
   Prediction: grounded_advice with confidence 0.4291.
   True label: grounded_advice.

2. Text: campus food proves Berkeley does not have it together. Desserts and sandwiches Desserts could be good, while the build-your-own sandwich station used generic ingredients and fel...
   Prediction: unsupported_take with confidence 0.415.
   True label: unsupported_take.

3. Text: The practical takeaway seems to be that someone living next to Cafe 3 may value proximity more than another student values a particular Crossroads station.
   Prediction: grounded_advice with confidence 0.4147.
   True label: grounded_advice.

4. Text: the dining situation is the only choice that makes sense. This is one strong negative experience, not a representative survey of every diner.
   Prediction: grounded_advice with confidence 0.3742.
   True label: unsupported_take.

5. Text: The comparison matters because clark Kerr dining A detailed positive review praised Clark Kerr for dishes such as salmon, ribs, pasta, pizza, macaroni and cheese, its salad bar,...
   Prediction: unsupported_take with confidence 0.3903.
   True label: grounded_advice.

## 3. Narrate One Correct Prediction

Use this one: "The comparison matters because practical takeaway The useful student advice is to inspect the daily menu and maintain alternatives instead of assuming every meal at the nearest ...". Explain why `grounded_advice` matches the label definition.

## 4. Narrate One Incorrect Prediction

Use this one: "the dining situation is the only choice that makes sense. This is one strong negative experience, not a representative survey of every diner.". The model predicted `grounded_advice` but the true label was `unsupported_take`. Explain the boundary it missed.

## 5. Walk Through Evaluation Report

Show the README evaluation section: overall metrics, per-class metrics, confusion matrix, AI-assisted pattern review, wrong-prediction analysis, and reflection.