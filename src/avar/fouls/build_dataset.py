# src/avar/fouls/build_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Set

import json
import math
import random

import numpy as np
import pandas as pd

from avar.datasets.soccernet_events import (
    SoccerNetEvent,
    load_foul_events_from_labels,
)


@dataclass
class Detection:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)


@dataclass
class WindowFeatures:
    game_id: str
    half: int
    event_frame: int
    center_frame: int
    label: int  # 1 = foul, 0 = non-foul

    # Simple geometric features
    min_pair_dist: float
    avg_min_pair_dist: float
    max_players_in_radius: int
    num_frames_with_close_contact: int
    window_size_frames: int


def load_detections_by_frame(det_json_path: Path) -> Tuple[Dict[int, List[Detection]], dict]:
    """
    Load detection JSON produced by export_detections_json and group by frame.

    Returns:
        detections_by_frame: dict[frame_idx] -> list[Detection]
        meta: dict with video metadata (width, height, fps, step, weights, conf)
    """
    det_json_path = Path(det_json_path)
    with det_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    video_w = int(data["width"])
    video_h = int(data["height"])
    fps = float(data["fps"])
    step = int(data.get("step", 1))
    video_path = data.get("video_path", "")

    dets_by_frame: Dict[int, List[Detection]] = {}
    for d in data["detections"]:
        frame = int(d["frame"])
        det = Detection(
            frame=frame,
            x1=float(d["x1"]),
            y1=float(d["y1"]),
            x2=float(d["x2"]),
            y2=float(d["y2"]),
            conf=float(d["conf"]),
            cls=int(d["cls"]),
        )
        dets_by_frame.setdefault(frame, []).append(det)

    meta = {
        "video_w": video_w,
        "video_h": video_h,
        "fps": fps,
        "step": step,
        "video_path": video_path,
    }
    return dets_by_frame, meta


def _pairwise_distances_norm(
    detections: List[Detection],
    video_w: int,
    video_h: int,
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between players in normalized coords.
    Returns an array of shape (N_pairs,) in [0, sqrt(2)].
    """
    if len(detections) < 2:
        return np.zeros((0,), dtype=float)

    xs = np.array([d.cx / max(1.0, video_w) for d in detections], dtype=float)
    ys = np.array([d.cy / max(1.0, video_h) for d in detections], dtype=float)
    coords = np.stack([xs, ys], axis=1)  # (N, 2)

    # Compute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))       # (N, N)

    # Take upper triangle without diagonal
    iu = np.triu_indices(len(detections), k=1)
    return dists[iu]


def _compute_window_features(
    game_id: str,
    half: int,
    center_frame: int,
    label: int,
    frames: Iterable[int],
    dets_by_frame: Dict[int, List[Detection]],
    video_w: int,
    video_h: int,
    contact_radius: float,
) -> WindowFeatures:
    """
    Aggregate pairwise distance features over a time window.
    """
    frames = list(frames)
    per_frame_min = []
    frames_with_contact = 0
    max_players_in_radius = 0

    for f in frames:
        dets = [d for d in dets_by_frame.get(f, []) if d.cls == 0]  # persons only
        if len(dets) < 2:
            continue

        dists = _pairwise_distances_norm(dets, video_w, video_h)
        if dists.size == 0:
            continue

        d_min = float(dists.min())
        per_frame_min.append(d_min)

        # Count close contacts
        if d_min < contact_radius:
            frames_with_contact += 1

        # For each detection, count how many neighbours are within radius
        # (rough "local density" feature)
        xs = np.array([d.cx / max(1.0, video_w) for d in dets], dtype=float)
        ys = np.array([d.cy / max(1.0, video_h) for d in dets], dtype=float)
        coords = np.stack([xs, ys], axis=1)
        diff = coords[:, None, :] - coords[None, :, :]
        d_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        # consider neighbours strictly within radius (exclude self)
        counts = (d_mat < contact_radius).sum(axis=1) - 1
        max_players_in_radius = max(max_players_in_radius, int(counts.max()))

    if per_frame_min:
        min_pair_dist = float(min(per_frame_min))
        avg_min_pair_dist = float(sum(per_frame_min) / len(per_frame_min))
    else:
        # If no frames with >=2 players, set large distance
        min_pair_dist = 1.0
        avg_min_pair_dist = 1.0

    return WindowFeatures(
        game_id=game_id,
        half=half,
        event_frame=center_frame,
        center_frame=center_frame,
        label=label,
        min_pair_dist=min_pair_dist,
        avg_min_pair_dist=avg_min_pair_dist,
        max_players_in_radius=max_players_in_radius,
        num_frames_with_close_contact=frames_with_contact,
        window_size_frames=len(frames),
    )


def build_foul_dataset_for_half(
    labels_path: Path,
    detections_json: Path,
    half: int,
    out_csv: Path,
    window_sec: float = 1.0,
    contact_radius: float = 0.05,
    negatives_per_positive: int = 3,
    negative_margin_sec: float = 5.0,
    random_seed: int = 42,
) -> None:
    """
    Build a baseline foul dataset for a single half of a game.

    Args:
        labels_path: Labels-v2.json for the game.
        detections_json: detection JSON for the half video (from detect-json).
        half: 1 or 2.
        out_csv: where to save the dataset CSV.
        window_sec: +/- seconds around event center to include in the window.
        contact_radius: normalized distance threshold for 'close contact' (0-~1).
        negatives_per_positive: number of non-foul windows per foul event.
        negative_margin_sec: minimal time distance (sec) from any foul to be considered negative.
        random_seed: RNG seed for reproducible negative sampling.
    """
    labels_path = Path(labels_path)
    detections_json = Path(detections_json)
    out_csv = Path(out_csv)

    dets_by_frame, meta = load_detections_by_frame(detections_json)
    video_w = meta["video_w"]
    video_h = meta["video_h"]
    fps = meta["fps"]
    step = meta["step"]

    # Note: we assume detection JSON uses raw frame indices (0..N-1).
    # window_size in frames at raw FPS
    window_frames = int(round(window_sec * fps))
    margin_frames = int(round(negative_margin_sec * fps))

    # Load foul events for this half and map them to frame indices
    foul_events: List[SoccerNetEvent] = load_foul_events_from_labels(
        labels_path=labels_path,
        fps=fps,
        label_filter=None,
        half_filter=[half],
    )

    if not foul_events:
        print(f"[fouls] No foul events found in {labels_path} for half={half}")
        return

    game_id = foul_events[0].game_id

    # Build a sorted list of all frames that have detections
    all_frames = sorted(dets_by_frame.keys())
    if not all_frames:
        print(f"[fouls] No detections in {detections_json}")
        return
    last_frame = all_frames[-1]

    random.seed(random_seed)

    rows: List[WindowFeatures] = []

    # Positive samples
    for ev in foul_events:
        center_frame = ev.frame
        f_start = max(0, center_frame - window_frames)
        f_end = min(last_frame, center_frame + window_frames)
        frames = range(f_start, f_end + 1)

        wf = _compute_window_features(
            game_id=game_id,
            half=half,
            center_frame=center_frame,
            label=1,
            frames=frames,
            dets_by_frame=dets_by_frame,
            video_w=video_w,
            video_h=video_h,
            contact_radius=contact_radius,
        )
        rows.append(wf)

    # Negative samples: sample random frames far from any foul
    foul_frames = np.array([ev.frame for ev in foul_events], dtype=int)

    def is_far_from_foul(f: int) -> bool:
        # quick check: if min distance in frames > margin_frames
        return int(np.min(np.abs(foul_frames - f))) > margin_frames

    candidate_frames = [f for f in all_frames if is_far_from_foul(f)]
    if not candidate_frames:
        print(
            f"[fouls] No negative candidate frames left with margin={negative_margin_sec}s"
        )
    else:
        num_neg = negatives_per_positive * len(foul_events)
        chosen_neg_frames = random.sample(
            candidate_frames, k=min(num_neg, len(candidate_frames))
        )

        for cf in chosen_neg_frames:
            f_start = max(0, cf - window_frames)
            f_end = min(last_frame, cf + window_frames)
            frames = range(f_start, f_end + 1)

            wf = _compute_window_features(
                game_id=game_id,
                half=half,
                center_frame=cf,
                label=0,
                frames=frames,
                dets_by_frame=dets_by_frame,
                video_w=video_w,
                video_h=video_h,
                contact_radius=contact_radius,
            )
            rows.append(wf)

    # Convert to DataFrame and save
    if not rows:
        print("[fouls] No rows built; nothing to save.")
        return

    df = pd.DataFrame(
        [
            {
                "game_id": r.game_id,
                "half": r.half,
                "event_frame": r.event_frame,
                "center_frame": r.center_frame,
                "label": r.label,
                "min_pair_dist": r.min_pair_dist,
                "avg_min_pair_dist": r.avg_min_pair_dist,
                "max_players_in_radius": r.max_players_in_radius,
                "num_frames_with_close_contact": r.num_frames_with_close_contact,
                "window_size_frames": r.window_size_frames,
            }
            for r in rows
        ]
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[fouls] Saved dataset with {len(df)} samples to {out_csv}")
