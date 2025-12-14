# src/avar/fouls/postprocess_events.py

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class PredPoint:
    """A single probability point produced by fouls-predict."""
    t: float        # seconds from start of half
    p: float        # foul probability
    frame: Optional[int] = None  # optional: frame index


@dataclass
class FoulEvent:
    """A post-processed event interval with a peak."""
    start_sec: float
    end_sec: float
    peak_sec: float
    peak_p: float
    n_points: int

    # optional bookkeeping
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    peak_frame: Optional[int] = None

    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


# -----------------------------
# JSON parsing (very tolerant)
# -----------------------------

def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _extract_pred_list(obj: Any) -> List[Dict[str, Any]]:
    """
    Accepts common formats:
      - [ {..}, {..} ]  (list)
      - { "preds": [..] } or { "predictions": [..] } or { "items": [..] }
      - { "results": { "preds": [..] } }  (nested)
    """
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        # direct keys
        for k in ("preds", "predictions", "items", "points"):
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

        # nested common patterns
        for k in ("results", "data", "output"):
            v = obj.get(k)
            if isinstance(v, dict):
                for kk in ("preds", "predictions", "items", "points"):
                    vv = v.get(kk)
                    if isinstance(vv, list):
                        return [x for x in vv if isinstance(x, dict)]

    raise ValueError(
        "Unsupported preds JSON format. Expected list of dicts or dict containing "
        "one of keys: preds/predictions/items/points."
    )


def _parse_pred_points(pred_dicts: List[Dict[str, Any]]) -> List[PredPoint]:
    """
    Supports multiple field names per prediction element.
    Required: time (seconds) and probability.
    Common field names:
      time:   t, time, time_sec, center_time_sec, center_t
      prob:   p, prob, score, foul_prob, probability
      frame:  frame, frame_idx, center_frame
    """
    pts: List[PredPoint] = []
    for d in pred_dicts:
        # time
        t = (
            d.get("t")
            if d.get("t") is not None
            else d.get("time")
            if d.get("time") is not None
            else d.get("time_sec")
            if d.get("time_sec") is not None
            else d.get("center_time_sec")
            if d.get("center_time_sec") is not None
            else d.get("center_t")
        )
        t_sec = _safe_float(t, default=None)

        # probability
        p = (
            d.get("p")
            if d.get("p") is not None
            else d.get("prob")
            if d.get("prob") is not None
            else d.get("score")
            if d.get("score") is not None
            else d.get("foul_prob")
            if d.get("foul_prob") is not None
            else d.get("probability")
        )
        prob = _safe_float(p, default=None)

        # optional frame index
        f = (
            d.get("frame")
            if d.get("frame") is not None
            else d.get("frame_idx")
            if d.get("frame_idx") is not None
            else d.get("center_frame")
        )
        frame = _safe_int(f, default=None)

        if t_sec is None or prob is None:
            # Skip malformed entries rather than crash
            continue

        pts.append(PredPoint(t=t_sec, p=prob, frame=frame))

    # ensure sorted
    pts.sort(key=lambda x: x.t)
    return pts


# -----------------------------
# Core logic
# -----------------------------

def _group_above_threshold(
    pts: List[PredPoint],
    threshold: float,
    gap_sec: float,
) -> List[List[PredPoint]]:
    """
    Returns contiguous groups of points where p >= threshold.
    A new group starts if time gap between consecutive kept points > gap_sec.
    """
    kept = [pt for pt in pts if pt.p >= threshold]
    if not kept:
        return []

    groups: List[List[PredPoint]] = []
    cur: List[PredPoint] = [kept[0]]

    for pt in kept[1:]:
        if (pt.t - cur[-1].t) <= gap_sec:
            cur.append(pt)
        else:
            groups.append(cur)
            cur = [pt]
    groups.append(cur)
    return groups


def _summarize_group(group: List[PredPoint]) -> FoulEvent:
    peak = max(group, key=lambda x: x.p)
    ev = FoulEvent(
        start_sec=group[0].t,
        end_sec=group[-1].t,
        peak_sec=peak.t,
        peak_p=peak.p,
        n_points=len(group),
        start_frame=group[0].frame,
        end_frame=group[-1].frame,
        peak_frame=peak.frame,
    )
    return ev


def _split_long_event_by_peaks(
    group: List[PredPoint],
    max_duration_sec: float,
    min_separation_sec: float,
) -> List[List[PredPoint]]:
    """
    If a group spans too long, split it by selecting multiple peaks spaced apart,
    then cutting midpoints between adjacent peaks.

    This prevents huge events like [0, 52s] collapsing into one.
    """
    if not group:
        return []
    dur = group[-1].t - group[0].t
    if max_duration_sec is None or max_duration_sec <= 0 or dur <= max_duration_sec:
        return [group]

    # Peak picking: greedy, highest p first with time separation constraint
    sorted_by_p = sorted(group, key=lambda x: x.p, reverse=True)
    peaks: List[PredPoint] = []
    for cand in sorted_by_p:
        if all(abs(cand.t - pk.t) >= min_separation_sec for pk in peaks):
            peaks.append(cand)

    # If we only got one peak, we still need to split by time chunks
    if len(peaks) <= 1:
        # Simple time chunking: split every max_duration_sec
        splits: List[List[PredPoint]] = []
        start_t = group[0].t
        cur: List[PredPoint] = []
        for pt in group:
            if pt.t - start_t <= max_duration_sec:
                cur.append(pt)
            else:
                if cur:
                    splits.append(cur)
                start_t = pt.t
                cur = [pt]
        if cur:
            splits.append(cur)
        return splits

    peaks.sort(key=lambda x: x.t)

    # Cut boundaries at midpoints between adjacent peaks
    boundaries: List[float] = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        boundaries.append(0.5 * (a.t + b.t))

    # Assign points to segments based on boundaries
    segments: List[List[PredPoint]] = [[] for _ in range(len(peaks))]
    for pt in group:
        idx = 0
        while idx < len(boundaries) and pt.t > boundaries[idx]:
            idx += 1
        segments[idx].append(pt)

    # remove empty segments (can happen with very sparse points)
    segments = [seg for seg in segments if seg]
    return segments if segments else [group]


def build_foul_events(
    preds_json: Path,
    out_json: Path,
    threshold: float = 0.6,
    gap_sec: float = 1.0,
    min_duration_sec: float = 0.6,
    topk: int = 80,
    max_duration_sec: Optional[float] = 6.0,
    min_separation_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Reads predictions, produces merged/split events, writes JSON.

    Notes:
    - min_duration_sec applies to event duration in seconds (end-start).
    - max_duration_sec splits long above-threshold runs to avoid giant events.
      Use None or <=0 to disable splitting.
    """
    preds_obj = _load_json(preds_json)
    pred_dicts = _extract_pred_list(preds_obj)
    pts = _parse_pred_points(pred_dicts)

    if not pts:
        result = {
            "threshold": threshold,
            "gap_sec": gap_sec,
            "min_duration_sec": min_duration_sec,
            "topk": topk,
            "max_duration_sec": max_duration_sec,
            "min_separation_sec": min_separation_sec,
            "events": [],
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    groups = _group_above_threshold(pts, threshold=threshold, gap_sec=gap_sec)

    # Split long groups
    split_groups: List[List[PredPoint]] = []
    for g in groups:
        if max_duration_sec is not None and max_duration_sec > 0:
            split_groups.extend(
                _split_long_event_by_peaks(
                    g,
                    max_duration_sec=float(max_duration_sec),
                    min_separation_sec=float(min_separation_sec),
                )
            )
        else:
            split_groups.append(g)

    # Summarize into events
    events: List[FoulEvent] = []
    for g in split_groups:
        ev = _summarize_group(g)
        if ev.duration() >= float(min_duration_sec):
            events.append(ev)

    # Rank by peak probability then duration
    events.sort(key=lambda e: (e.peak_p, e.duration()), reverse=True)

    if topk is not None and topk > 0:
        events = events[: int(topk)]

    # Output
    result = {
        "threshold": threshold,
        "gap_sec": gap_sec,
        "min_duration_sec": min_duration_sec,
        "topk": topk,
        "max_duration_sec": max_duration_sec,
        "min_separation_sec": min_separation_sec,
        "events": [asdict(e) for e in events],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def summarize_events(events_json: Path, topn: int = 5) -> str:
    """Convenience function for quick printing/debugging."""
    obj = _load_json(events_json)
    events = obj.get("events", [])
    if not events:
        return "No events."
    lines = []
    for i, e in enumerate(events[:topn]):
        lines.append(
            f"{i+1:02d}) peak_p={e.get('peak_p'):.3f} "
            f"peak_t={e.get('peak_sec'):.2f}s "
            f"range=[{e.get('start_sec'):.2f},{e.get('end_sec'):.2f}] "
            f"dur={e.get('end_sec')-e.get('start_sec'):.2f}s "
            f"n={e.get('n_points')}"
        )
    return "\n".join(lines)
