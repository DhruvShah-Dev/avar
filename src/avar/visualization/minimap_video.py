# src/avar/visualization/minimap_video.py

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


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
    # Dark green background
    pitch[:, :] = (0, 100, 0)

    line_color = (255, 255, 255)
    thickness = 2

    # Margins around the pitch
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
    cv2.circle(pitch, center, 3, line_color, -1)  # center spot

    # Penalty boxes (very approximate proportions)
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

    # Goal areas (smaller boxes)
    g_w = int((x2 - x1) * 0.03)
    g_h = int((y2 - y1) * 0.2)
    g_y1 = y1 + (y2 - y1 - g_h) // 2
    g_y2 = g_y1 + g_h

    # Left goal area
    cv2.rectangle(
        pitch,
        (x1 - g_w, g_y1),
        (x1, g_y2),
        line_color,
        thickness,
    )
    # Right goal area
    cv2.rectangle(
        pitch,
        (x2, g_y1),
        (x2 + g_w, g_y2),
        line_color,
        thickness,
    )

    return pitch, x1, y1, x2, y2


def export_minimap_video(
    video_path: Path,
    out_path: Path,
    weights: str = "yolov8n.pt",
    conf: float = 0.25,
    step: int = 1,
    device: Optional[str] = None,
    pitch_size: Tuple[int, int] = (1050, 680),
) -> None:
    """
    Build a 2D minimap video:
      - runs YOLO on the original broadcast video,
      - projects player (person) detections to a top-down pitch,
      - draws two team colors based on horizontal position.

    This does NOT use the SoccerNet calibration yet. It normalizes image
    coordinates into the field rectangle, which already looks like a
    reasonable top view for analytics demos.
    """
    model = YOLO(weights)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    pitch_w, pitch_h = pitch_size
    base_pitch, fx1, fy1, fx2, fy2 = draw_pitch(pitch_w, pitch_h)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (pitch_w, pitch_h),
    )

    team_a_color = (255, 0, 0)   # Blue-ish (BGR)
    team_b_color = (0, 0, 255)   # Red-ish (BGR)

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Optionally skip inference on some frames, but still advance minimap
        if step > 1 and (frame_idx % step != 0):
            # Just reuse empty pitch (no new detections)
            pitch_frame = base_pitch.copy()
            writer.write(pitch_frame)
            frame_idx += 1
            continue

        # Run YOLO on the original frame
        results = model.predict(
            frame,
            conf=conf,
            verbose=False,
            device=device if device is not None else None,
        )

        pitch_frame = base_pitch.copy()

        if len(results) > 0:
            r = results[0]
            boxes = (
                r.boxes.xyxy.cpu().numpy()
                if r.boxes is not None
                else []
            )
            confs = (
                r.boxes.conf.cpu().numpy()
                if r.boxes is not None and r.boxes.conf is not None
                else []
            )
            clss = (
                r.boxes.cls.cpu().numpy().astype(int)
                if r.boxes is not None and r.boxes.cls is not None
                else []
            )

            field_w = fx2 - fx1
            field_h = fy2 - fy1

            for box, score, cls in zip(boxes, confs, clss):
                # Only 'person' for COCO weights
                if cls != 0:
                    continue

                x1, y1, x2, y2 = box
                # Approximate player position at feet (center bottom of box)
                cx = 0.5 * (x1 + x2)
                cy = y2

                # Normalize in image coordinates [0,1]
                xn = float(cx) / max(1.0, img_w)
                yn = float(cy) / max(1.0, img_h)

                # Map to field rectangle; invert Y so top of image -> top of pitch
                px = int(fx1 + xn * field_w)
                py = int(fy1 + (1.0 - yn) * field_h)

                # Simple team split: left half vs right half in image space
                team_color = team_a_color if xn < 0.5 else team_b_color

                cv2.circle(pitch_frame, (px, py), 6, team_color, -1)

        writer.write(pitch_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[minimap] Saved minimap video to {out_path}")
