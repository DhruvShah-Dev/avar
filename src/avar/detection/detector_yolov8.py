# src/avar/detection/detector_yolov8.py

from typing import List

import numpy as np
import torch
from ultralytics import YOLO


class PlayerDetector:
    """
    YOLOv8-based detector for players.
    Default weights: 'yolov8n.pt' (COCO, people as 'person' class).
    """

    def __init__(self, weights_path: str = "yolov8n.pt", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        # If weights_path is 'yolov8n.pt', YOLO will download it automatically if missing
        self.model = YOLO(weights_path)
        self.model.to(self.device)

        if self.device == "cuda":
            self.model.model.half()

    @torch.inference_mode()
    def detect_batch(self, frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run detection on a batch of frames.
        Returns: list of arrays [N, 6] => [x1, y1, x2, y2, conf, cls]
        """
        results = self.model(frames_bgr, verbose=False)
        out = []
        for r in results:
            det = r.boxes.data.detach().cpu().numpy()
            out.append(det)
        return out
