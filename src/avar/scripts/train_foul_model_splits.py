# src/avar/scripts/train_foul_model_splits.py

from __future__ import annotations

from pathlib import Path

from avar.fouls.train_model import train_from_splits


def run(
    splits_dir: str,
    out_model: str,
    out_metrics: str,
    features: str = "",
    seed: int = 42,
):
    """
    Train the foul model using precomputed split CSVs.
    """
    train_from_splits(
        splits_dir=Path(splits_dir),
        out_model=Path(out_model),
        out_metrics=Path(out_metrics),
        features=features,
        seed=int(seed),
    )
