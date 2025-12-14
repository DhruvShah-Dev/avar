# src/avar/visualization/detect_video.py

from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO
import numpy as np


def export_detection_video(
    video_path: Path,
    out_path: Path,
    weights: str = "yolov8n.pt",
    conf: float = 0.25,
    step: int = 1,
    device: Optional[str] = None,
) -> None:
    """
    Run YOLO directly on the ORIGINAL video frames and write an MP4
    with only player/person bounding boxes drawn on top.

    - No tracking
    - No resizing mismatch
    - No JSON
    """
    model = YOLO(weights)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Optionally skip frames for speed but still write them
        if step > 1 and (frame_idx % step != 0):
            writer.write(frame)
            frame_idx += 1
            continue

        # Run YOLO on the ORIGINAL frame size
        results = model.predict(
            frame,
            conf=conf,
            verbose=False,
            device=device if device is not None else None,
        )

        # Draw only 'person' detections (class 0 for COCO models)
        if len(results) > 0:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
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

            for box, score, cls in zip(boxes, confs, clss):
                # keep only 'person' class if using COCO pretrained
                if cls != 0:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{score:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[detect-video] Saved detection video to {out_path}")
