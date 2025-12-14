# src/avar/fouls/train_model.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score


# Default features engineered in build_dataset.py
DEFAULT_FEATURE_COLUMNS: List[str] = [
    "min_pair_dist",
    "avg_min_pair_dist",
    "max_players_in_radius",
    "num_frames_with_close_contact",
    "window_size_frames",
]

# Backwards compatibility: older code imports FEATURE_COLUMNS
FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS


def _parse_features_arg(features: str, fallback: List[str]) -> List[str]:
    if not features:
        return fallback
    cols = [c.strip() for c in features.split(",") if c.strip()]
    return cols or fallback


def load_foul_datasets(
    csv_paths: Sequence[Path],
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load one or more foul dataset CSVs and return (X, y).
    """
    feature_cols = feature_cols or DEFAULT_FEATURE_COLUMNS
    dfs: List[pd.DataFrame] = []

    for p in csv_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        dfs.append(pd.read_csv(p))

    df_all = pd.concat(dfs, ignore_index=True)

    if "label" not in df_all.columns:
        raise ValueError("Dataset must contain a 'label' column (0/1).")

    missing = [c for c in feature_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")

    X = df_all[feature_cols].astype("float32")
    y = df_all["label"].astype(int)
    return X, y


def train_foul_model(csv_paths: Sequence[Path], out_model: Path) -> None:
    """
    Legacy entrypoint: train on provided CSV(s) and save a joblib.
    (No validation/test split; kept for backward compatibility.)
    """
    X, y = load_foul_datasets(csv_paths)

    print(f"[train-fouls] Training on {len(X)} samples")
    print(f"[train-fouls] Label distribution:\n{y.value_counts()}")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)

    y_pred = clf.predict(X)
    print("[train-fouls] In-sample classification report:")
    print(classification_report(y, y_pred, digits=3))

    out_model = Path(out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    bundle = {"model": clf, "feature_names": DEFAULT_FEATURE_COLUMNS}
    dump(bundle, out_model)
    print(f"[train-fouls] Saved model to {out_model}")


@dataclass
class TrainFromSplitsResult:
    best_threshold: float
    metrics: Dict


def _choose_threshold(y_true: np.ndarray, p1: np.ndarray) -> Tuple[float, Dict]:
    """
    Tune a decision threshold on validation probabilities by maximizing F1 for the positive class.
    """
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.linspace(0.05, 0.95, 91):
        y_hat = (p1 >= thr).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, {"best_f1": float(best_f1)}


def train_from_splits(
    *,
    splits_dir: Optional[Path] = None,
    train_csv: Optional[Path] = None,
    valid_csv: Optional[Path] = None,
    test_csv: Optional[Path] = None,
    out_model: Path,
    out_metrics: Path,
    features: str = "",
    seed: int = 42,
) -> TrainFromSplitsResult:
    """
    Train a foul classifier using split CSVs (train/valid/test).

    Supports either:
      - splits_dir containing train.csv, valid.csv, test.csv
      - explicit paths via train_csv/valid_csv/test_csv

    IMPORTANT: This signature accepts keyword args train_csv/valid_csv/test_csv
    to match your existing CLI/script call sites.
    """
    if splits_dir is not None:
        splits_dir = Path(splits_dir)
        train_csv = splits_dir / "train.csv"
        valid_csv = splits_dir / "valid.csv"
        test_csv = splits_dir / "test.csv"

    if train_csv is None or valid_csv is None or test_csv is None:
        raise ValueError("Provide either splits_dir or explicit train_csv/valid_csv/test_csv paths.")

    train_csv = Path(train_csv)
    valid_csv = Path(valid_csv)
    test_csv = Path(test_csv)

    feat_cols = _parse_features_arg(features, DEFAULT_FEATURE_COLUMNS)

    X_train, y_train = load_foul_datasets([train_csv], feature_cols=feat_cols)
    X_valid, y_valid = load_foul_datasets([valid_csv], feature_cols=feat_cols)
    X_test, y_test = load_foul_datasets([test_csv], feature_cols=feat_cols)

    print(f"[train-splits] train={len(X_train)} valid={len(X_valid)} test={len(X_test)}")
    print(f"[train-splits] label counts (train): {y_train.value_counts().to_dict()}")

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    # threshold tuning on validation probabilities
    p_valid = clf.predict_proba(X_valid)[:, 1]
    best_thr, thr_summary = _choose_threshold(y_valid.to_numpy(), p_valid)

    def _eval_split(X: pd.DataFrame, y: pd.Series) -> Dict:
        p1 = clf.predict_proba(X)[:, 1]
        y_hat = (p1 >= best_thr).astype(int)

        rep = classification_report(y, y_hat, digits=4, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, y_hat).tolist()

        auc = None
        try:
            if len(np.unique(y.to_numpy())) == 2:
                auc = float(roc_auc_score(y, p1))
        except Exception:
            auc = None

        return {
            "n": int(len(y)),
            "label_counts": {int(k): int(v) for k, v in y.value_counts().to_dict().items()},
            "threshold": float(best_thr),
            "auc": auc,
            "confusion_matrix": cm,
            "classification_report": rep,
        }

    metrics = {
        "feature_names": feat_cols,
        "threshold_tuning": {"criterion": "f1_pos", "best_threshold": best_thr, **thr_summary},
        "train": _eval_split(X_train, y_train),
        "valid": _eval_split(X_valid, y_valid),
        "test": _eval_split(X_test, y_test),
    }

    out_model = Path(out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": clf, "feature_names": feat_cols, "threshold": best_thr}, out_model)

    out_metrics = Path(out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[train-splits] Saved model to {out_model}")
    print(f"[train-splits] Saved metrics to {out_metrics}")
    print(f"[train-splits] Best threshold={best_thr:.3f}")

    return TrainFromSplitsResult(best_threshold=best_thr, metrics=metrics)
