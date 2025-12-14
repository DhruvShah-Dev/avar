# src/avar/datasets/soccernet_events.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import json
import re


@dataclass
class SoccerNetEvent:
    game_id: str
    half: int            # 1 or 2
    label: str
    time_sec: float      # seconds from start of half
    frame: int           # frame index at native FPS
    metadata: dict       # full raw annotation if needed


# Default labels we treat as "foul-related"
DEFAULT_FOUL_LABELS: Set[str] = {
    "foul",
    "yellow card",
    "red card",
    "penalty",
    "dangerous play",
    "hand ball",
    "violent conduct",
}


def _parse_game_id(labels_path: Path) -> str:
    """
    Derive a game_id from the directory structure, e.g.

    data/raw/soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels-v2.json
    -> england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
    """
    # Expect .../<league>/<season>/<game>/Labels-v2.json
    parts = labels_path.with_suffix("").parts
    # Find the 'soccernet' root if present and take everything under it
    if "soccernet" in parts:
        idx = parts.index("soccernet")
        game_parts = parts[idx + 1:-1]  # drop "Labels-v2"
    else:
        # Fallback: just last 3 path components
        game_parts = parts[-3:-1]
    return "/".join(game_parts)


def _parse_game_time(game_time: str) -> tuple[int, float]:
    """
    Parse SoccerNet "gameTime" string, e.g. "1 - 00:15:23.520" or "2 - 45:03".

    Returns:
        (half, time_sec)
    """
    # Format: "<half> - MM:SS" or "<half> - HH:MM:SS(.ms)"
    m = re.match(r"\s*(\d+)\s*-\s*([0-9:.\-]+)\s*", game_time)
    if not m:
        raise ValueError(f"Unexpected gameTime format: {game_time!r}")
    half = int(m.group(1))
    time_str = m.group(2)

    comps = time_str.split(":")
    comps = [c.strip() for c in comps if c.strip()]

    if len(comps) == 2:
        # MM:SS(.ms)
        mm = float(comps[0])
        ss = float(comps[1])
        total_sec = 60.0 * mm + ss
    elif len(comps) == 3:
        # HH:MM:SS(.ms)
        hh = float(comps[0])
        mm = float(comps[1])
        ss = float(comps[2])
        total_sec = 3600.0 * hh + 60.0 * mm + ss
    else:
        raise ValueError(f"Unexpected time component count in {game_time!r}")

    return half, total_sec


def _get_time_sec_from_annotation(ann: dict) -> tuple[int, float]:
    """
    Robustly extract (half, time_sec) from a SoccerNet annotation dict.

    We prefer explicit fields if present, otherwise parse 'gameTime'.
    """
    half: Optional[int] = None
    time_sec: Optional[float] = None

    if "half" in ann:
        try:
            half = int(ann["half"])
        except Exception:
            half = None

    if "position" in ann:
        # 'position' is usually in seconds from start of game or half
        try:
            time_sec = float(ann["position"])
        except Exception:
            time_sec = None

    if half is not None and time_sec is not None:
        return half, time_sec

    # Fallback to "gameTime" string
    if "gameTime" in ann:
        return _parse_game_time(ann["gameTime"])

    raise ValueError(
        f"Could not extract time from annotation, keys={list(ann.keys())}"
    )


def load_foul_events_from_labels(
    labels_path: Path,
    fps: float,
    label_filter: Optional[Sequence[str]] = None,
    half_filter: Optional[Iterable[int]] = None,
) -> List[SoccerNetEvent]:
    """
    Load foul-related events from a SoccerNet Labels-v2.json file and map them to frame indices.

    Args:
        labels_path: path to Labels-v2.json for a game.
        fps: frames per second of the corresponding 224p video (typically 25).
        label_filter: list of labels to keep (defaults to DEFAULT_FOUL_LABELS).
        half_filter: optional iterable of halves to keep (e.g., [1] or [2]).

    Returns:
        List[SoccerNetEvent] with frame index at native FPS.
    """
    labels_path = Path(labels_path)
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    anns = data.get("annotations") or data.get("events") or []

    if not isinstance(anns, list):
        raise ValueError(
            f"Unexpected Labels-v2 format in {labels_path}, "
            f"'annotations' is not a list."
        )

    game_id = _parse_game_id(labels_path)
    lf_set: Set[str] = set(
        l.lower() for l in (label_filter or DEFAULT_FOUL_LABELS)
    )
    half_keep: Optional[Set[int]] = None
    if half_filter is not None:
        half_keep = {int(h) for h in half_filter}

    events: List[SoccerNetEvent] = []

    for ann in anns:
        label = str(ann.get("label", "")).strip()
        if not label:
            continue

        if label.lower() not in lf_set:
            continue

        half, time_sec = _get_time_sec_from_annotation(ann)

        if half_keep is not None and half not in half_keep:
            continue

        frame = int(round(time_sec * fps))

        events.append(
            SoccerNetEvent(
                game_id=game_id,
                half=half,
                label=label,
                time_sec=time_sec,
                frame=frame,
                metadata=ann,
            )
        )

    events.sort(key=lambda e: (e.half, e.time_sec))
    return events
