from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json


class MVFoulDataset(Dataset):
    """
    Multi-view (or single-view) foul dataset from SoccerNet clips.

    meta_file JSON format (example):
    [
      {
        "video_path": "data/raw/soccernet/fouls/clip1.mp4",
        "timestamp": 12.3,     # center time in seconds (or 0 if pre-cut clip)
        "label_foul": 0        # 0 = no foul, 1 = foul
      },
      ...
    ]
    """

    def __init__(
        self,
        meta_file: str,
        num_frames: int = 16,
        fps: int = 12,
        resize: int = 224,
    ):
        self.meta_path = Path(meta_file)
        with self.meta_path.open() as f:
            self.samples: List[Dict[str, Any]] = json.load(f)

        self.num_frames = num_frames
        self.fps = fps
        self.resize = resize

    def __len__(self) -> int:
        return len(self.samples)

    def _load_clip(self, video_path: str, center_time: float) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        center_frame = int(center_time * video_fps)

        half = self.num_frames // 2
        start = max(center_frame - half, 0)
        end = min(center_frame + half, total_frames - 1)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for idx in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resize, self.resize))
            frames.append(frame)

        cap.release()

        if not frames:
            raise RuntimeError(f"No frames decoded in {video_path}")

        # Pad if clip is too short
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        frames = np.stack(frames[: self.num_frames], axis=0)  # (T,H,W,C)
        return frames

    def __getitem__(self, idx: int):
        s: Dict[str, Any] = self.samples[idx]
        clip = self._load_clip(s["video_path"], s["timestamp"])

        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float()  # (T,C,H,W)
        label = torch.tensor(s["label_foul"], dtype=torch.long)

        return clip, {"foul": label}
