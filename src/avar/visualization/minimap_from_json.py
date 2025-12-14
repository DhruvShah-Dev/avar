# src/avar/visualization/minimap_from_json.py

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import json


def draw_pitch(
    width: int,
    height: int,
) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Draw a simple soccer pitch on a blank image.

    Returns:
        pitch: HxWx3 uint8 image (BGR)
        x1, y1, x2, y2: inner field rectangle (for mapping player coords)
    """
    pitch = np.zeros((height, width, 3), dtype=np.uint8)
    pitch[:, :] = (0, 100, 0)  # dark green

    line_color = (255, 255, 255)
    thickness = 2

    margin_x = int(0.05 * width)
    margin_y = int(0.05 * height)

    x1, y1 = margin_x, margin_y
    x2, y2 = width - margin_x, height - margin_y

    # Outer rectangle
    cv2.rectangle(pitch, (x1, y1), (x2, y2), line_color, thickness)

    # Halfway line
    mid_x = (x1 + x2) // 2
    cv2.line(pitch, (mid_x, y1), (mid_x, y2), line_color, thickness)

    # Center circle
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    radius = int((y2 - y1) * 0.2)
    cv2.circle(pitch, center, radius, line_color, thickness)
    cv2.circle(pitch, center, 3, line_color, -1)

    # Penalty boxes (rough)
    box_w = int((x2 - x1) * 0.16)
    box_h = int((y2 - y1) * 0.4)
    top_box_y1 = y1 + (y2 - y1 - box_h) // 2
    top_box_y2 = top_box_y1 + box_h

    # Left penalty box
    cv2.rectangle(
        pitch,
        (x1, top_box_y1),
        (x1 + box_w, top_box_y2),
        line_color,
        thickness,
    )

    # Right penalty box
    cv2.rectangle(
        pitch,
        (x2 - box_w, top_box_y1),
        (x2, top_box_y2),
        line_color,
        thickness,
    )

    # Goal areas
    g_w = int((x2 - x1) * 0.03)
    g_h = int((y2 - y1) * 0.2)
    g_y1 = y1 + (y2 - y1 - g_h) // 2
    g_y2 = g_y1 + g_h

    cv2.rectangle(pitch, (x1 - g_w, g_y1), (x1, g_y2), line_color, thickness)
    cv2.rectangle(pitch, (x2, g_y1), (x2 + g_w, g_y2), line_color, thickness)

    return pitch, x1, y1, x2, y2


def export_minimap_from_json(
    det_json: Path,
    out_path: Path,
    pitch_size: Tuple[int, int] = (1050, 680),
) -> None:
    """
    Build a 2D minimap video from a detections JSON produced by export_detections_json.

    - No YOLO here; we trust the JSON boxes as the reference.
    - Players shown as dots in two colors (left vs right in image).
    """
    with det_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    video_w = int(data["width"])
    video_h = int(data["height"])
    fps = float(data["fps"])
    step = int(data.get("step", 1))

    detections = data["detections"]

    # Group detections by frame
    det_by_frame = {}
    max_frame = 0
    for d in detections:
        fidx = int(d["frame"])
        det_by_frame.setdefault(fidx, []).append(d)
        if fidx > max_frame:
            max_frame = fidx

    pitch_w, pitch_h = pitch_size
    base_pitch, fx1, fy1, fx2, fy2 = draw_pitch(pitch_w, pitch_h)
    field_w = fx2 - fx1
    field_h = fy2 - fy1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (pitch_w, pitch_h),
    )

    team_a_color = (255, 0, 0)  # blue-ish
    team_b_color = (0, 0, 255)  # red-ish

    # We assume frames are 0..max_frame; JSON may be sparse if step>1
    for frame_idx in range(0, max_frame + 1):
        pitch_frame = base_pitch.copy()

        if frame_idx in det_by_frame:
            for d in det_by_frame[frame_idx]:
                if d["cls"] != 0:
                    continue  # only persons

                x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                cx = 0.5 * (x1 + x2)
                cy = y2

                xn = cx / max(1.0, video_w)
                yn = cy / max(1.0, video_h)

                px = int(fx1 + xn * field_w)
                py = int(fy1 + (1.0 - yn) * field_h)

                team_color = team_a_color if xn < 0.5 else team_b_color
                cv2.circle(pitch_frame, (px, py), 6, team_color, -1)

        writer.write(pitch_frame)

    writer.release()
    print(f"[minimap-json] Saved minimap video to {out_path}")
