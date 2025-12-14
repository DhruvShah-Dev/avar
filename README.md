# Robot Perception: Automated Video Assistant Referee (AVAR)

This project provides tools for automated video analysis in sports, focusing on player detection, tracking, event (foul) detection, and visualization. It is designed for use with the SoccerNet dataset and includes a command-line interface (CLI) for various tasks.

## Features
- Download SoccerNet data
- Player detection and tracking (YOLOv8-based)
- Project tracks to 2D pitch coordinates
- Generate heatmaps from player tracks
- Visualize tracks and detections on video
- Build and train foul detection models
- Predict and postprocess foul events

## Installation
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd robot-perception
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. (Optional) Install SoccerNet tools:
   ```sh
   pip install SoccerNet
   ```

## Usage
Run the CLI from the `src/avar/` directory:
```sh
python -m avar.cli <command> [options]
```

Example: Run player tracking on a video
```sh
python -m avar.cli track --video <input.mp4> --out <tracks.json>
```

For a full list of commands and options:
```sh
python -m avar.cli --help
```

## Directory Structure
- `src/avar/` - Main source code
- `data/` - Data, outputs, and visualizations
- `models/` - Trained models

## License
See [LICENSE](LICENSE) for details.
