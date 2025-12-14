# src/avar/analytics/heatmap.py

from pathlib import Path
from typing import List, Dict, Any, Tuple

import json
import numpy as np
import matplotlib.pyplot as plt


def load_tracks_2d(path: Path) -> List[Dict[str, Any]]:
    """Load 2D projected tracks from JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_heatmap_grid(
    tracks_2d_path: Path,
    bins: Tuple[int, int] = (80, 52),
) -> np.ndarray:
    """
    Build a 2D occupancy grid over pitch from projected tracks.

    - bins = (gx, gy) is the number of bins along X and Y.
    - H has shape (gy, gx) so that H[y, x] is valid.
    """
    tracks = load_tracks_2d(tracks_2d_path)
    if not tracks:
        raise ValueError(f"No tracks found in {tracks_2d_path}")

    xs = np.array([t["X"] for t in tracks], dtype=float)
    ys = np.array([t["Y"] for t in tracks], dtype=float)

    # Debug: inspect coordinate ranges
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    print(f"[heatmap] {len(tracks)} points")
    print(f"[heatmap] X range: {x_min:.3f} .. {x_max:.3f}")
    print(f"[heatmap] Y range: {y_min:.3f} .. {y_max:.3f}")

    # Normalize to [0,1] so we can bin
    xs_n = (xs - x_min) / (x_max - x_min + 1e-6)
    ys_n = (ys - y_min) / (y_max - y_min + 1e-6)

    gx, gy = bins  # gx = number of cells in X, gy in Y
    # IMPORTANT: H shape is (gy, gx) so H[iy, ix] is valid
    H = np.zeros((gy, gx), dtype=float)

    for x, y in zip(xs_n, ys_n):
        ix = min(gx - 1, max(0, int(x * gx)))
        iy = min(gy - 1, max(0, int(y * gy)))
        H[iy, ix] += 1.0

    print(f"[heatmap] nnz={np.count_nonzero(H)}, max_cell={H.max():.1f}")
    return H


def save_heatmap(H: np.ndarray, out_path: Path) -> None:
    """Save heatmap array as a PNG image with reasonable contrast."""
    H_norm = H.astype(float)
    max_val = H_norm.max()
    if max_val > 0:
        H_norm /= max_val

    plt.figure(figsize=(8, 5))
    plt.imshow(
        H_norm,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[heatmap] Saved image to {out_path}")
