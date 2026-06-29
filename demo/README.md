# Demo Video

Record a 3-5 minute demo after the Colab run.

Show:

- 3-5 posts classified by the fine-tuned model with labels and confidence visible.
- One correct prediction with a short explanation.
- One incorrect prediction with why it likely went wrong.
- A brief walkthrough of the README evaluation report.

If the video is too large for GitHub, upload it elsewhere and link it from
`README.md`.

After Colab outputs are copied into `reports/`, run:

```powershell
python scripts/render_readme_metrics.py
```

This creates `demo/DEMO_SCRIPT.md` with real examples from the test predictions.
