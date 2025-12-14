# src/avar/visualization/vis_tracking.py

from pathlib import Path
import json
import cv2
import numpy as np


def load_tracks(tracks_json: Path):
    with tracks_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def draw_track(frame, track):
    x1, y1, x2, y2 = track["bbox"]
    tid = track["track_id"]

    # Box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

    # ID text
    cv2.putText(
        frame,
        f"ID {tid}",
        (int(x1), int(y1)-8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2,
        cv2.LINE_AA
    )


def export_tracking_video(
    video_path: Path,
    tracks_json: Path,
    out_path: Path,
    fps: int = None,
):
    tracks = load_tracks(tracks_json)

    # Build index: frame -> list of detections
    frame_map = {}
    for t in tracks:
        f = t["frame"]
        frame_map.setdefault(f, []).append(t)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_fps = fps if fps is not None else video_fps

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (W, H),
    )

    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_id in frame_map:
            for tr in frame_map[frame_id]:
                draw_track(frame, tr)

        writer.write(frame)
        frame_id += 1

    writer.release()
    cap.release()

    print(f"[vis] Saved tracking video to {out_path}")
