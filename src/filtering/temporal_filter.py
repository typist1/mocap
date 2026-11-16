"""Per-landmark real-time temporal filters.

One Euro Filter: Casiez et al., 2012 — https://cristal.univ-lille.fr/~casiez/1euro/
Kalman Filter: constant-velocity model for 3D point tracking.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.config.settings import FilteringConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# One Euro Filter — 1D
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """Adaptive low-pass filter for 1D signal."""

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.5,
        beta: float = 0.1,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x
        # Derivative (velocity)
        dx = (x - self._x_prev) * self.freq
        # Low-pass the derivative
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_filtered = a_d * dx + (1.0 - a_d) * self._dx_prev
        # Speed-dependent cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_filtered)
        a = self._alpha(cutoff, self.freq)
        # Low-pass the signal
        x_filtered = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_filtered
        self._dx_prev = dx_filtered
        return x_filtered

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0


# ---------------------------------------------------------------------------
# One Euro Filter — 3D
# ---------------------------------------------------------------------------

class OneEuroFilter3D:
    """Applies OneEuroFilter independently to x, y, z components."""

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.5,
        beta: float = 0.1,
        d_cutoff: float = 1.0,
    ) -> None:
        self._filters = [
            OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(3)
        ]

    def filter(self, point: np.ndarray) -> np.ndarray:
        return np.array([f.filter(float(point[i])) for i, f in enumerate(self._filters)])

    def reset(self) -> None:
        for f in self._filters:
            f.reset()


# ---------------------------------------------------------------------------
# Kalman Filter — 3D constant-velocity
# ---------------------------------------------------------------------------

class KalmanFilter3D:
    """Constant-velocity Kalman filter for a 3D point.

    State: [x, y, z, vx, vy, vz]  (6D)
    Measurement: [x, y, z]         (3D)
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-2,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        # State vector [6]
        self._x = np.zeros(6)
        # Covariance matrix [6, 6]
        self._P = np.eye(6) * 1.0
        self._initialized = False

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    @property
    def _H(self) -> np.ndarray:
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        return H

    def predict(self, dt: float = 1.0 / 30.0) -> np.ndarray:
        F = self._F(dt)
        Q = np.eye(6) * self.process_noise
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        return self._x[:3].copy()

    def update(self, measurement: np.ndarray, confidence: float = 1.0) -> np.ndarray:
        if not self._initialized:
            self._x[:3] = measurement
            self._initialized = True
            return measurement.copy()
        H = self._H
        R = np.eye(3) * (self.measurement_noise / max(confidence, 1e-3))
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        innovation = measurement - H @ self._x
        self._x = self._x + K @ innovation
        self._P = (np.eye(6) - K @ H) @ self._P
        return self._x[:3].copy()

    def reset(self) -> None:
        self._x = np.zeros(6)
        self._P = np.eye(6) * 1.0
        self._initialized = False


# ---------------------------------------------------------------------------
# LandmarkFilter — applies OneEuroFilter3D to all 33 landmarks
# ---------------------------------------------------------------------------

class LandmarkFilter:
    """Real-time filter applied per-landmark per-frame."""

    def __init__(
        self,
        n_landmarks: int = 33,
        freq: float = 30.0,
        min_cutoff: float = 1.5,
        beta: float = 0.1,
        d_cutoff: float = 1.0,
        min_visibility: float = 0.3,
        config: Optional[FilteringConfig] = None,
    ) -> None:
        if config is not None:
            min_cutoff = config.one_euro_min_cutoff
            beta = config.one_euro_beta
            d_cutoff = config.one_euro_d_cutoff
            min_visibility = config.kalman_enabled and 0.3 or 0.3
        self.min_visibility = min_visibility
        self._filters = [
            OneEuroFilter3D(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(n_landmarks)
        ]

    def filter(self, world_pts: np.ndarray, visibility: Optional[np.ndarray] = None) -> np.ndarray:
        """Filter shape [N, 3]. Low-visibility landmarks skip the update (preserving
        filter state) so the next visible frame sees a smooth history, not a cold start."""
        result = world_pts.copy()
        for i, flt in enumerate(self._filters):
            if visibility is not None and float(visibility[i]) < self.min_visibility:
                # Do NOT reset — keep filter warm so the next visible frame is continuous.
                continue
            result[i] = flt.filter(world_pts[i])
        return result

    def reset(self) -> None:
        for f in self._filters:
            f.reset()
