# Reports

Generated Colab outputs go here after training:

- `evaluation_results.json`
- `confusion_matrix.png`
- `predictions.csv`
- `wrong_predictions.csv`
- `sample_classifications.csv`
- `test_split.csv`

These files are produced by:

```powershell
python ai201_project3_takemeter_starter_clean.py --csv data/takemeter_labeled.csv --community "YOUR COMMUNITY"
```

For the actual model run, use Google Colab with a T4 GPU and `GROQ_API_KEY`
stored in Colab Secrets.

After those files exist, render README-ready Markdown tables with:

```powershell
python scripts/render_readme_metrics.py
```

The renderer also creates:

- `reports/readme_evaluation_section.md`
- `reports/ai_wrong_prediction_prompt.md`
- `demo/DEMO_SCRIPT.md`
