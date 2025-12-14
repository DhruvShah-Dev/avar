# src/avar/detection/detect_to_json.py

from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
from ultralytics import YOLO


def export_detections_json(
    video_path: Path,
    out_json: Path,
    weights: str = "yolov8n.pt",
    conf: float = 0.25,
    step: int = 1,
    device: Optional[str] = None,
) -> None:
    """
    Run YOLO on the ORIGINAL video frames and save detections to JSON.

    JSON format:
    {
      "video_path": "...",
      "width": int,
      "height": int,
      "fps": float,
      "weights": "yolov8n.pt",
      "conf": 0.25,
      "step": 1,
      "detections": [
        {
          "frame": int,
          "x1": float,
          "y1": float,
          "x2": float,
          "y2": float,
          "conf": float,
          "cls": int
        },
        ...
      ]
    }
    """
    import json

    model = YOLO(weights)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    all_dets: List[Dict[str, Any]] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if step > 1 and (frame_idx % step != 0):
            frame_idx += 1
            continue

        results = model.predict(
            frame,
            conf=conf,
            verbose=False,
            device=device if device is not None else None,
        )

        if len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = (
                    r.boxes.conf.cpu().numpy()
                    if r.boxes.conf is not None
                    else []
                )
                clss = (
                    r.boxes.cls.cpu().numpy().astype(int)
                    if r.boxes.cls is not None
                    else []
                )

                for box, sc, cl in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = box.tolist()
                    all_dets.append(
                        {
                            "frame": int(frame_idx),
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "conf": float(sc),
                            "cls": int(cl),
                        }
                    )

        frame_idx += 1

    cap.release()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_path": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "weights": weights,
        "conf": conf,
        "step": step,
        "detections": all_dets,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"[detect-json] Saved detections to {out_json}")
