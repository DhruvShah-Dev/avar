from __future__ import annotations

from pathlib import Path

from avar.fouls.split_dataset import split_by_game


def run(
    master_csv: Path,
    out_dir: Path,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> None:
    res = split_by_game(
        master_csv=Path(master_csv),
        out_dir=Path(out_dir),
        train_ratio=float(train_ratio),
        valid_ratio=float(valid_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
        stratify=bool(stratify),
    )

    m = res.manifest
    print("[split-fouls] Done.")
    print(f"[split-fouls] Games: train={m['n_games']['train']} valid={m['n_games']['valid']} test={m['n_games']['test']}")
    print(f"[split-fouls] Samples: train={m['n_samples']['train']} valid={m['n_samples']['valid']} test={m['n_samples']['test']}")
    print(f"[split-fouls] Label counts (train): {m['label_counts']['train']}")
    print(f"[split-fouls] Label counts (valid): {m['label_counts']['valid']}")
    print(f"[split-fouls] Label counts (test):  {m['label_counts']['test']}")
    print(f"[split-fouls] Wrote: {Path(out_dir)/'train.csv'}, {Path(out_dir)/'valid.csv'}, {Path(out_dir)/'test.csv'}")
    print(f"[split-fouls] Manifest: {Path(out_dir)/'split_manifest.json'}")
