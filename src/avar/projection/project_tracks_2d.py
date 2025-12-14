# src/avar/projection/project_tracks_2d.py

from pathlib import Path
import json
from typing import List, Dict, Any, Optional

import numpy as np


def _try_get(d, *keys):
    """
    Try to follow a nested path of keys.
    Example: _try_get(data, "calibration", "homography")
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _find_homography_in_obj(obj: Any) -> Optional[List[float]]:
    """
    Recursively search an object (dict/list) for something that looks like
    a homography:
      - list/tuple of length 9 or 16 (flat matrix)
      - or dict with 'homography' / 'H' / 'homography_uv_to_xy'
    Return the first such list found, or None.
    """
    # Direct list/tuple candidate
    if isinstance(obj, (list, tuple)) and len(obj) in (9, 16) and all(
        isinstance(x, (int, float)) for x in obj
    ):
        return list(obj)

    # Dict with possible homography keys
    if isinstance(obj, dict):
        # Check common keys inside this dict
        for key in ("homography", "H", "homography_uv_to_xy"):
            if key in obj:
                val = obj[key]
                if isinstance(val, (list, tuple)) and len(val) in (9, 16):
                    return list(val)

        # Otherwise, recurse on values
        for v in obj.values():
            found = _find_homography_in_obj(v)
            if found is not None:
                return found

    # List of nested things
    if isinstance(obj, list):
        for v in obj:
            found = _find_homography_in_obj(v)
            if found is not None:
                return found

    return None


def load_soccernet_homography(calib_path: Path) -> np.ndarray:
    """
    Load CCBV calibration file and return 3x3 homography matrix H.

    For SoccerNet 'field_calib_ccbv.json', we know top-level keys may look like:
      - 'UrlLocal', 'size', 'predictions'

    We:
      1) Try a few common patterns at top level.
      2) If that fails, we **search inside 'predictions'** (dict or list),
         recursively, for any 9/16-length numeric list or homography-like key.
    """
    with calib_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) Simple top-level patterns
    candidates = [
        ("homography", lambda d: d.get("homography", None)),
        ("H", lambda d: d.get("H", None)),
        ("calibration.homography", lambda d: _try_get(d, "calibration", "homography")),
        ("calib.homography", lambda d: _try_get(d, "calib", "homography")),
        ("ccbv.homography", lambda d: _try_get(d, "ccbv", "homography")),
        ("homography_uv_to_xy", lambda d: d.get("homography_uv_to_xy", None)),
    ]

    H_flat = None
    used_name = None

    for name, getter in candidates:
        H_candidate = getter(data)
        if H_candidate is None:
            continue
        if isinstance(H_candidate, (list, tuple)) and len(H_candidate) in (9, 16):
            H_flat = H_candidate
            used_name = name
            break

    # 2) If nothing at top level, search inside 'predictions'
    if H_flat is None and "predictions" in data:
        H_candidate = _find_homography_in_obj(data["predictions"])
        if H_candidate is not None:
            H_flat = H_candidate
            used_name = "predictions (recursive search)"

    # 3) Fallback: any top-level list of length 9/16
    if H_flat is None:
        for k, v in data.items():
            if isinstance(v, (list, tuple)) and len(v) in (9, 16):
                H_flat = v
                used_name = f"auto-picked from top-level key '{k}'"
                break

    if H_flat is None:
        # Still nothing: raise with diagnostic info
        raise KeyError(
            f"Could not find homography in {calib_path}. "
            f"Top-level keys: {list(data.keys())}"
        )

    H = np.array(H_flat, dtype=float)
    if H.size == 9:
        H = H.reshape(3, 3)
    elif H.size == 16:
        # Occasionally stored as 4x4, in which case we take the upper 3x3
        H = H.reshape(4, 4)[:3, :3]
    else:
        raise ValueError(
            f"Unexpected homography size from {used_name}: {H.shape}, len={H.size}"
        )

    return H


def project_tracks_to_2d(
    tracks_json: Path,
    calib_json: Path,
    out_json: Path,
) -> None:
    """
    Input (tracks_json):
      [
        {
          "frame": int,
          "track_id": int,
          "bbox": [x1, y1, x2, y2],
          ...
        },
        ...
      ]

    Output (out_json):
      [
        {
          "frame": int,
          "track_id": int,
          "X": float,  # pitch X
          "Y": float,  # pitch Y
        },
        ...
      ]
    """
    with tracks_json.open("r", encoding="utf-8") as f:
        tracks: List[Dict[str, Any]] = json.load(f)

    H = load_soccernet_homography(calib_json)

    out: List[Dict[str, Any]] = []
    for tr in tracks:
        x1, y1, x2, y2 = tr["bbox"]
        # Approximate player position as bottom-center of bbox
        cx = 0.5 * (x1 + x2)
        cy = y2
        p = np.array([cx, cy, 1.0], dtype=float)
        P = H @ p
        P /= (P[2] + 1e-9)
        X, Y = float(P[0]), float(P[1])

        out.append(
            {
                "frame": tr["frame"],
                "track_id": tr["track_id"],
                "X": X,
                "Y": Y,
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f)
