# AVAR: SoccerNet video analysis prototype

AVAR is a research prototype for analyzing broadcast soccer video with the [SoccerNet](https://www.soccer-net.org/) dataset. It combines YOLO-based player detection, simple tracking, tactical visualizations, and a proximity-feature model that proposes foul candidate windows. It does not make referee decisions.

## Pipeline

1. Detect players and save detections or tracked positions as JSON.
2. Render bounding-box overlays or a 2D tactical minimap.
3. Align detections with SoccerNet foul annotations to build a labeled dataset.
4. Split data by game, train a baseline classifier, and score candidate foul windows.
5. Consolidate nearby predictions into proposed events.

The command-line entry point is `src/avar/cli.py`. Its subcommands cover download, tracking, detection, minimap rendering, dataset preparation, model training, and prediction.

## Setup

Use Python and install the dependencies listed in [requirements.txt](requirements.txt). The repository uses a `src/` layout and does not include packaging metadata, so set `PYTHONPATH` when running the CLI.

PowerShell:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m avar.cli --help
```

Bash:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m avar.cli --help
```

Run `python -m avar.cli <command> --help` after setting `PYTHONPATH` to see a command's required inputs and output options. Video processing is compute intensive; a supported GPU can speed up detection.

## Dataset and outputs

SoccerNet videos and annotations are **not included** here. Obtain access through the [official SoccerNet site](https://www.soccer-net.org/) and follow its terms. The pipeline expects downloaded data under `data/raw/soccernet/`. Generated data, models, images, and videos are excluded from Git, so outputs must be recreated locally.

## Method and limits

- Game-level train, validation, and test splits reduce leakage between matches.
- Camera calibration and pitch projection are approximations.
- Foul candidates use weak supervision and engineered proximity features. A candidate score does not establish that a foul occurred.
- Performance depends on the licensed dataset, detection quality, and hardware.

## Author

Dhruv Shah · Binghamton University