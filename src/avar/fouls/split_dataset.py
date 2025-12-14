from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    manifest: Dict


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def _simple_game_split(
    game_ids: List[str],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)

    n = len(game_ids)
    n_train = int(round(train_ratio * n))
    n_valid = int(round(valid_ratio * n))
    n_test = n - n_train - n_valid

    if n_test < 0:
        n_test = 0
        n_valid = n - n_train

    train = game_ids[:n_train]
    valid = game_ids[n_train : n_train + n_valid]
    test = game_ids[n_train + n_valid :]

    return train, valid, test


def split_by_game(
    master_csv: Path,
    out_dir: Path,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> SplitResult:
    master_csv = Path(master_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not master_csv.exists():
        raise FileNotFoundError(f"master_csv not found: {master_csv}")

    df = pd.read_csv(master_csv)

    if df.empty:
        raise RuntimeError("Master CSV is empty.")

    _require_cols(df, ["league", "season", "game", "half", "label"])

    df["league"] = df["league"].astype(str)
    df["season"] = df["season"].astype(str)
    df["game"] = df["game"].astype(str)

    # Use existing game_id if present
    if "game_id" not in df.columns:
        df["game_id"] = df["league"] + "///" + df["season"] + "///" + df["game"]

    g = (
        df.groupby("game_id")["label"]
        .agg(["count", "mean", "sum"])
        .rename(columns={"count": "n_samples", "mean": "foul_rate", "sum": "n_fouls"})
        .reset_index()
    )

    if len(g) < 3:
        raise RuntimeError("Not enough games to split safely.")

    # -------------------------
    # Attempt stratified split
    # -------------------------
    used_stratification = False
    train_ids: list[str] = []
    valid_ids: list[str] = []
    test_ids: list[str] = []

    if stratify:
        try:
            n_bins = min(5, max(2, len(g) // 5))
            g["bin"] = pd.qcut(g["foul_rate"], q=n_bins, duplicates="drop")

            if g["bin"].isna().all():
                raise ValueError("Stratification collapsed (all bins NaN).")

            used_stratification = True
            rng = np.random.default_rng(seed)

            for b in g["bin"].dropna().unique():
                ids = g[g["bin"] == b]["game_id"].tolist()
                rng.shuffle(ids)

                t, v, te = _simple_game_split(
                    ids, train_ratio, valid_ratio, seed
                )
                train_ids.extend(t)
                valid_ids.extend(v)
                test_ids.extend(te)

        except Exception as e:
            print(f"[split-fouls] Stratification disabled: {e}")

    # -------------------------
    # Fallback: simple split
    # -------------------------
    if not train_ids and not valid_ids and not test_ids:
        all_games = g["game_id"].tolist()
        train_ids, valid_ids, test_ids = _simple_game_split(
            all_games, train_ratio, valid_ratio, seed
        )

    # Final safety check
    if not train_ids or not valid_ids or not test_ids:
        raise RuntimeError(
            "Split failed: one or more splits empty after fallback.\n"
            f"Games total={len(g)} train={len(train_ids)} valid={len(valid_ids)} test={len(test_ids)}"
        )

    s_train, s_valid, s_test = set(train_ids), set(valid_ids), set(test_ids)

    train = df[df["game_id"].isin(s_train)].copy()
    valid = df[df["game_id"].isin(s_valid)].copy()
    test = df[df["game_id"].isin(s_test)].copy()

    # Write outputs
    train.to_csv(out_dir / "train.csv", index=False)
    valid.to_csv(out_dir / "valid.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    manifest = {
        "master_csv": str(master_csv),
        "seed": seed,
        "ratios": {"train": train_ratio, "valid": valid_ratio, "test": test_ratio},
        "used_stratification": used_stratification,
        "n_games": {
            "train": len(s_train),
            "valid": len(s_valid),
            "test": len(s_test),
        },
        "n_samples": {
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
        },
        "label_counts": {
            "train": train["label"].value_counts().to_dict(),
            "valid": valid["label"].value_counts().to_dict(),
            "test": test["label"].value_counts().to_dict(),
        },
    }

    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return SplitResult(train=train, valid=valid, test=test, manifest=manifest)
