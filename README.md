# Automated Video Assistant Referee (AVAR) – SoccerNet

## Project Overview
This project implements a simplified Automated Video Assistant Referee (AVAR) pipeline for broadcast soccer videos using the SoccerNet dataset. The system focuses on three tasks:
1. Player detection from broadcast video.
2. Visualization through overlay videos and a tactical minimap.
3. Foul candidate prediction using engineered proximity-based features.

The project is designed as an end-to-end, reproducible computer vision and machine learning pipeline suitable for large-scale soccer video analysis.

---

## Dataset
This project uses the **SoccerNet** dataset.

- Dataset access requires an official SoccerNet account and password.
- Due to licensing restrictions, raw videos and annotations are **not included** in this repository.
- Users must download the dataset separately from:
  https://www.soccer-net.org/

The expected directory structure after downloading SoccerNet is:
```
data/raw/soccernet/
```

---

## Code Structure
```
avar/
 ├── cli.py                  # Command-line interface
 ├── detections/             # Player detection logic
 ├── fouls/                  # Foul dataset creation, training, prediction
 ├── viz/                    # Overlay and minimap visualizations
data/
 ├── raw/                    # SoccerNet data (not included)
 ├── processed/              # Detection outputs
 ├── analytics/              # Foul datasets, splits, metrics
 ├── visualizations/         # Generated videos and figures
models/
 ├── foul_baseline.pkl
 ├── foul_lr_split.pkl
```

---

## Pipeline Steps
1. **Player Detection**  
   Detect players using a YOLO-based object detector on broadcast video.

2. **Visualization**  
   - Overlay detected player bounding boxes on the original video.
   - Generate a 2D tactical minimap representation.

3. **Foul Dataset Construction**  
   Align player detections with SoccerNet foul annotations to build a supervised dataset.

4. **Model Training and Evaluation**  
   Train a baseline logistic regression classifier using proximity-based features and evaluate using a game-level train/validation/test split.

5. **Foul Prediction**  
   Generate frame-level foul probabilities and post-process them into temporal foul event candidates.

---

## Hardware Requirements
- CPU > i5
- GPU (any)

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- NumPy, Pandas, Scikit-learn
- Ultralytics YOLO

Exact package versions are listed in the environment setup used during development.

---

## Reproducibility Notes
- Detection outputs are cached to avoid repeated video processing.
- Dataset splits are performed **by game** to prevent data leakage.
- All experiments can be reproduced using the provided CLI commands once SoccerNet data is available.

---

## Limitations
- Camera calibration and metric pitch projection are approximated.
- Foul prediction relies on weak supervision and engineered features.
- The system is intended as a research prototype, not a production referee system.

---

## Author
Dhruv Shah  
Department of Computer Science  
Binghamton University (SUNY)
