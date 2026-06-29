# Data

Place the final labeled TakeMeter dataset here:

```text
data/takemeter_labeled.csv
```

Required columns:

- `text`
- `label`

Optional columns such as `source_url`, `post_id`, `created_at`, and `notes` are
fine, but the training workflow uses only `text` and `label`.

Run the local preflight before uploading to Colab:

```powershell
python ai201_project3_takemeter_starter_clean.py --preflight data/takemeter_labeled.csv --community "YOUR COMMUNITY"
```
