from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from avar.fouls.build_dataset import build_foul_dataset_for_half


@dataclass(frozen=True)
class Item:
    league: str
    season: str
    game: str
    half: int
    det_path: Path
    labels_path: Path


def _norm_path_str(p: str) -> str:
    # Normalize a path string for robust substring checks on Windows/Linux
    return p.replace("\\", "/").strip()


def _extract_game_from_video_path(video_path: str) -> Optional[Tuple[str, str, str]]:
    """
    Attempt to parse league/season/game from a video_path stored inside detection JSON.
    Expected pattern contains: .../data/raw/soccernet/<league>/<season>/<game>/<half_video>.mkv
    """
    vp = _norm_path_str(video_path)
    marker = "/data/raw/soccernet/"
    if marker not in vp:
        return None

    tail = vp.split(marker, 1)[1]  # <league>/<season>/<game>/...
    parts = tail.split("/")
    if len(parts) < 3:
        return None

    league, season, game = parts[0], parts[1], parts[2]
    return league, season, game


def _try_match_by_filename(
    det_name: str, raw_root: Path, league_filter: Optional[str]
) -> Optional[Tuple[str, str, str, Path]]:
    """
    Match by checking if the raw game directory name appears inside the detection filename.
    Returns (league, season, game, labels_path) if matched.
    """
    for league_dir in raw_root.iterdir():
        if not league_dir.is_dir():
            continue
        if league_filter and league_dir.name != league_filter:
            continue

        for season_dir in league_dir.iterdir():
            if not season_dir.is_dir():
                continue

            for game_dir in season_dir.iterdir():
                if not game_dir.is_dir():
                    continue
                labels_path = game_dir / "Labels-v2.json"
                if not labels_path.exists():
                    continue

                if game_dir.name in det_name:
                    return league_dir.name, season_dir.name, game_dir.name, labels_path

    return None


def _try_match_by_video_path(det_path: Path, raw_root: Path) -> Optional[Tuple[str, str, str, Path]]:
    """
    Match by reading the detection JSON metadata and parsing video_path to locate Labels-v2.json.
    """
    try:
        with det_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        video_path = data.get("video_path") or data.get("video") or ""
    except Exception:
        return None

    if not video_path:
        return None

    parsed = _extract_game_from_video_path(video_path)
    if not parsed:
        return None

    league, season, game = parsed
    labels_path = raw_root / league / season / game / "Labels-v2.json"
    if labels_path.exists():
        return league, season, game, labels_path

    return None


def find_items(
    raw_root: Path,
    det_root: Path,
    league_filter: Optional[str] = "england_epl",
) -> List[Item]:
    raw_root = Path(raw_root)
    det_root = Path(det_root)

    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")
    if not det_root.exists():
        raise FileNotFoundError(f"det_root not found: {det_root}")

    det_files = [
        p for p in det_root.iterdir()
        if p.is_file() and (p.name.endswith("_h1_dets.json") or p.name.endswith("_h2_dets.json"))
    ]

    if not det_files:
        raise FileNotFoundError(f"No *_h1_dets.json / *_h2_dets.json files found in {det_root}")

    items: List[Item] = []
    skipped: List[str] = []

    for det_path in det_files:
        half = 1 if det_path.name.endswith("_h1_dets.json") else 2

        # 1) Prefer filename-based match (fast, no JSON load)
        matched = _try_match_by_filename(det_path.name, raw_root, league_filter)

        # 2) Fallback: parse video_path from JSON (handles special names like chelsea_burnley_h1_dets.json)
        if matched is None:
            matched = _try_match_by_video_path(det_path, raw_root)

        if matched is None:
            skipped.append(det_path.name)
            continue

        league, season, game, labels_path = matched
        items.append(
            Item(
                league=league,
                season=season,
                game=game,
                half=half,
                det_path=det_path,
                labels_path=labels_path,
            )
        )

    if not items:
        msg = (
            "Detection files were found, but none could be matched to raw SoccerNet game folders.\n"
            f"raw_root={raw_root}\n"
            f"det_root={det_root}\n"
        )
        if skipped:
            msg += "Examples of skipped detection filenames:\n  - " + "\n  - ".join(skipped[:20])
        raise RuntimeError(msg)

    # Optional: print a short skip summary (not fatal)
    if skipped:
        print(f"[master-fouls] Skipped {len(skipped)} detection files that could not be matched (showing up to 5):")
        for s in skipped[:5]:
            print(f"  - {s}")

    return items


def build_master(
    raw_root: Path,
    det_root: Path,
    out_csv: Path,
    per_half_out_dir: Optional[Path] = None,
    league_filter: Optional[str] = "england_epl",
    window_sec: float = 1.0,
    contact_radius: float = 0.05,
    negatives_per_positive: int = 3,
    negative_margin_sec: float = 5.0,
    random_seed: int = 42,
) -> None:
    raw_root = Path(raw_root)
    det_root = Path(det_root)
    out_csv = Path(out_csv)

    items = find_items(raw_root=raw_root, det_root=det_root, league_filter=league_filter)
    print(f"[master-fouls] Matched {len(items)} halves")

    if per_half_out_dir is None:
        per_half_out_dir = out_csv.parent / "per_half"
    per_half_out_dir = Path(per_half_out_dir)
    per_half_out_dir.mkdir(parents=True, exist_ok=True)

    dfs: List[pd.DataFrame] = []

    for i, it in enumerate(items, 1):
        # Create a deterministic per-half filename (safe on Windows)
        safe_game = re.sub(r"[^\w\-. ]+", "_", it.game)
        per_half_csv = per_half_out_dir / f"{it.league}_{it.season}_{safe_game}_h{it.half}.csv"

        # Build per-half CSV using your existing implementation (writes CSV)
        build_foul_dataset_for_half(
            labels_path=it.labels_path,
            detections_json=it.det_path,
            half=it.half,
            out_csv=per_half_csv,
            window_sec=window_sec,
            contact_radius=contact_radius,
            negatives_per_positive=negatives_per_positive,
            negative_margin_sec=negative_margin_sec,
            random_seed=random_seed,
        )

        # If build_foul_dataset_for_half found no fouls/detections, it returns early and may not create CSV
        if not per_half_csv.exists():
            continue

        df = pd.read_csv(per_half_csv)

        # Add extra identifiers for leakage-safe splitting + debugging
        df["league"] = it.league
        df["season"] = it.season
        df["game"] = it.game
        df["half"] = it.half
        df["det_file"] = it.det_path.name

        dfs.append(df)

        if i % 10 == 0:
            print(f"[master-fouls] Processed {i}/{len(items)} halves")

    if not dfs:
        raise RuntimeError(
            "No per-half CSVs were produced. This typically means:\n"
            "- no foul events were found in Labels-v2.json for your halves, OR\n"
            "- detections JSONs were empty / missing.\n"
            "Try running build-foul-dataset for one of these halves to validate inputs."
        )

    master = pd.concat(dfs, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_csv, index=False)

    print(f"[master-fouls] Saved {len(master)} samples to {out_csv}")
    if "label" in master.columns:
        print("[master-fouls] Label counts:")
        print(master["label"].value_counts())
