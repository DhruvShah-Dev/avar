# Safe split example

`sample_foul_features.csv` is synthetic. It demonstrates the window-level schema without redistributing SoccerNet video or annotations. It is too small to evaluate a foul model.

From the repository root, with dependencies installed:

```powershell
$env:PYTHONPATH = "src"
python -m avar.cli split-foul-dataset --master-csv examples/sample_foul_features.csv --out-dir examples/generated --stratify 0
```

The generated `split_manifest.json` reports row counts, game counts, label balance, missing values, and duplicate window keys. Training, validation, and test games are disjoint. The generated directory is disposable.
