from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from joblib import load

from avar.fouls.build_dataset import (
    load_detections_by_frame,
    _compute_window_features,
)
from avar.fouls.train_model import FEATURE_COLUMNS


def predict_fouls_for_half(
    detections_json: Path,
    half: int,
    model_path: Path,
    out_json: Path,
    window_sec: float = 1.0,
    contact_radius: float = 0.05,
    stride_sec: float = 1.0,
    step_sec: Optional[float] = None,
    score_threshold: Optional[float] = None,
) -> None:
    """
    Predict fouls over a half using detection JSON.
    Uses relative spatial features only (no absolute image size).
    """

    detections_json = Path(detections_json)
    model_path = Path(model_path)
    out_json = Path(out_json)

    if step_sec is not None:
        stride_sec = float(step_sec)

    dets_by_frame, meta = load_detections_by_frame(detections_json)

    if "fps" not in meta:
        raise KeyError("detections meta must contain 'fps'")

    fps = float(meta["fps"])

    # Load model bundle
    bundle = load(model_path)
    if isinstance(bundle, dict):
        clf = bundle["model"]
        feature_names = bundle.get("feature_names", FEATURE_COLUMNS)
        threshold = float(bundle.get("threshold", 0.5))
    else:
        clf = bundle
        feature_names = FEATURE_COLUMNS
        threshold = 0.5

    if score_threshold is not None:
        threshold = float(score_threshold)

    feature_names = list(feature_names)

    frames = sorted(dets_by_frame.keys())
    if not frames:
        raise RuntimeError("No frames in detection file")

    stride_frames = max(1, int(round(stride_sec * fps)))
    window_frames = max(1, int(round(window_sec * fps)))

    print(
        f"[fouls-predict] fps={fps} "
        f"stride_frames={stride_frames} "
        f"window_frames={window_frames}"
    )

    predictions: List[Dict[str, Any]] = []

    for center_frame in frames[::stride_frames]:
        start = center_frame - window_frames
        end = center_frame + window_frames

        window_frames_list = [
            f for f in frames if start <= f <= end
        ]

        wf = _compute_window_features(
            game_id=detections_json.stem,
            half=int(half),
            center_frame=int(center_frame),
            label=0,
            frames=window_frames_list,
            dets_by_frame=dets_by_frame,
            video_w=1.0,   # dummy (not used in logic)
            video_h=1.0,   # dummy (not used in logic)
            contact_radius=float(contact_radius),
        )

        row = {
            "min_pair_dist": wf.min_pair_dist,
            "avg_min_pair_dist": wf.avg_min_pair_dist,
            "max_players_in_radius": wf.max_players_in_radius,
            "num_frames_with_close_contact": wf.num_frames_with_close_contact,
            "window_size_frames": wf.window_size_frames,
        }

        X = pd.DataFrame([row], columns=feature_names)
        p_foul = float(clf.predict_proba(X)[0, 1])
        pred = int(p_foul >= threshold)

        predictions.append(
            {
                "center_frame": int(center_frame),
                "center_time_sec": float(center_frame / fps),
                "p_foul": p_foul,
                "pred_label": pred,
                "features": row,
            }
        )

    output = {
        "detections": str(detections_json),
        "half": int(half),
        "fps": fps,
        "threshold": threshold,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "n_predictions": len(predictions),
        "predictions": predictions,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2))
    print(f"[fouls-predict] Wrote {len(predictions)} predictions → {out_json}")
