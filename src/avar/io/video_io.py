# src/avar/io/video_io.py

from typing import Generator, Optional, Tuple

import cv2
import numpy as np


def iter_video_frames(
    video_path: str,
    step: int = 2,
    target_size: Optional[Tuple[int, int]] = (960, 540),
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Iterate over video frames, skipping frames for speed and resizing.

    :param video_path: path to video file (.mkv / .mp4)
    :param step: use every `step`-th frame (2 => half FPS)
    :param target_size: (width, height) for resize, or None for original size
    :yield: (frame_index, frame_bgr)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            if target_size is not None:
                frame = cv2.resize(frame, target_size)
            yield frame_id, frame

        frame_id += 1

    cap.release()
