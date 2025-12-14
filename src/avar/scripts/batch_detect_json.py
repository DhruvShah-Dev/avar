from __future__ import annotations
import os
from pathlib import Path
from avar.detection.detect_to_json import export_detections_json

SOCCERNET_ROOT = Path("data/raw/soccernet")
OUTPUT_ROOT = Path("data/processed/detections")

VIDEO_NAME_MAP = {
    1: "1_224p.mkv",
    2: "2_224p.mkv"
}

def batch_detect(weights="yolov8n.pt", conf=0.25, step=1, device=None):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for league in SOCCERNET_ROOT.iterdir():
        if not league.is_dir():
            continue

        for season in league.iterdir():
            if not season.is_dir():
                continue

            for game in season.iterdir():
                if not game.is_dir():
                    continue

                for half in [1, 2]:
                    video_path = game / VIDEO_NAME_MAP[half]
                    if not video_path.exists():
                        continue

                    out_json = OUTPUT_ROOT / f"{league.name}_{season.name}_{game.name}_h{half}_dets.json"
                    if out_json.exists():
                        print("[skip exists]", out_json)
                        continue

                    print("[detect-json]", video_path)
                    export_detections_json(
                        video_path=video_path,
                        out_json=out_json,
                        weights=weights,
                        conf=conf,
                        step=step,
                        device=device,
                    )

if __name__ == "__main__":
    batch_detect()
