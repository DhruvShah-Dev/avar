from pathlib import Path
from typing import List, Dict, Any
import json


def save_tracks(tracks: List[Dict[str, Any]], out_path: str) -> None:
    """
    Save tracking results as JSON.

    :param tracks: list of dicts, each like:
                   {"frame": int, "track_id": int, "bbox": [x1,y1,x2,y2], ...}
    :param out_path: output JSON file path
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(tracks, f)


def load_tracks(path: str) -> List[Dict[str, Any]]:
    """
    Load tracking results from JSON.
    """
    p = Path(path)
    with p.open() as f:
        return json.load(f)
