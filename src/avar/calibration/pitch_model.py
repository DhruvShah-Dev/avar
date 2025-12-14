import numpy as np

PITCH_LENGTH = 105.0  # meters
PITCH_WIDTH = 68.0    # meters


class Pitch2D:
    """
    Simple 2D pitch representation and homography-based projection.
    """

    def __init__(self, length: float = PITCH_LENGTH, width: float = PITCH_WIDTH):
        self.length = length
        self.width = width

    @staticmethod
    def project_point(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
        """
        Project image point (x, y) using homography H into pitch coordinates.

        H: 3x3 homography matrix (image -> pitch).
        """
        p = np.array([x, y, 1.0], dtype=float)
        P = H @ p
        P /= (P[2] + 1e-9)
        return float(P[0]), float(P[1])
