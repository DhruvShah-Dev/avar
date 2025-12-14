from __future__ import annotations

from pathlib import Path
from typing import Optional

from avar.fouls.postprocess_events import build_foul_events


def run(
    preds_json: str,
    out_events_json: str,
    threshold: float = 0.60,
    gap_sec: float = 1.0,
    min_duration_sec: float = 0.6,
    topk: Optional[int] = None,
) -> None:
    res = build_foul_events(
        preds_json=Path(preds_json),
        out_json=Path(out_events_json),
        threshold=threshold,
        gap_sec=gap_sec,
        min_duration_sec=min_duration_sec,
        topk=topk,
    )

    print(f"[fouls-postprocess] events={res['n_events']}")
    print(f"[fouls-postprocess] wrote: {out_events_json}")
    if res["n_events"] > 0:
        best = res["events"][0]
        print(
            f"[fouls-postprocess] top event: "
            f"peak_p={best['peak_p']:.3f} "
            f"t={best['center_time_sec']:.2f}s "
            f"range=[{best['start_time_sec']:.2f},{best['end_time_sec']:.2f}]"
        )
