"""Camera model for monocular 3D reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.pose.keypoints import SKELETON_CONNECTIONS
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CameraModel:
    fx: float          # focal length in pixels (horizontal)
    fy: float          # focal length in pixels (vertical; usually == fx for monocular)
    cx: float          # principal point x
    cy: float          # principal point y
    width: int = 0
    height: int = 0

    @property
    def focal_length_px(self) -> float:
        return (self.fx + self.fy) / 2.0


def estimate_focal_length(
    landmarks_image: np.ndarray,  # [33, 3]: x,y normalized 0-1, visibility
    landmarks_world: np.ndarray,  # [33, 3]: x,y,z in meters
    width: int,
    height: int,
) -> float:
    """Estimate focal length from known bone lengths in world vs pixel space.

    For each visible bone: f ≈ d_px * d_world / max(|Δz|, ε)
    Returns median over well-conditioned estimates.
    """
    estimates = []
    px_x = landmarks_image[:, 0] * width
    px_y = landmarks_image[:, 1] * height
    vis = landmarks_image[:, 2]

    for a, b in SKELETON_CONNECTIONS:
        if vis[a] < 0.5 or vis[b] < 0.5:
            continue
        d_px = float(np.hypot(px_x[b] - px_x[a], px_y[b] - px_y[a]))
        d_world = float(np.linalg.norm(landmarks_world[b] - landmarks_world[a]))
        dz = abs(float(landmarks_world[b, 2] - landmarks_world[a, 2]))
        if d_px < 2.0 or d_world < 1e-4 or dz < 1e-4:
            continue
        f = d_px * d_world / dz
        if 100.0 < f < 20000.0:  # sanity range in pixels
            estimates.append(f)

    if len(estimates) < 2:
        fallback = 1.2 * max(width, height)
        logger.debug("Focal length estimation: too few estimates (%d), using fallback %.1f", len(estimates), fallback)
        return fallback

    return float(np.median(estimates))


def make_camera(width: int, height: int, focal_px: Optional[float] = None) -> CameraModel:
    """Construct a CameraModel with sane defaults."""
    if focal_px is None:
        focal_px = 1.2 * max(width, height)
    return CameraModel(
        fx=focal_px, fy=focal_px,
        cx=width / 2.0, cy=height / 2.0,
        width=width, height=height,
    )


def project_3d_to_2d(camera: CameraModel, points_3d: np.ndarray) -> np.ndarray:
    """Weak perspective projection. Input [N,3] or [3], output [N,2] or [2] in pixels."""
    squeeze = points_3d.ndim == 1
    pts = np.atleast_2d(points_3d)
    z = pts[:, 2].copy()
    z = np.where(np.abs(z) < 1e-4, np.sign(z + 1e-9) * 1e-4, z)
    x_px = camera.fx * pts[:, 0] / z + camera.cx
    y_px = camera.fy * pts[:, 1] / z + camera.cy
    result = np.stack([x_px, y_px], axis=-1)
    return result[0] if squeeze else result


def unproject_2d_to_3d(
    camera: CameraModel,
    points_2d: np.ndarray,  # [N,2] in pixels
    depths: np.ndarray,     # [N] in meters
) -> np.ndarray:
    """Back-project 2D pixel + depth to 3D camera space. Output [N,3]."""
    pts = np.atleast_2d(points_2d)
    d = np.asarray(depths)
    x = (pts[:, 0] - camera.cx) * d / camera.fx
    y = (pts[:, 1] - camera.cy) * d / camera.fy
    return np.stack([x, y, d], axis=-1)


def reprojection_error(
    camera: CameraModel,
    pts_3d: np.ndarray,     # [N,3]
    pts_2d: np.ndarray,     # [N,2] in pixels
    visibility: Optional[np.ndarray] = None,  # [N] 0-1
) -> float:
    """Mean L2 reprojection error on visible joints (pixels)."""
    projected = project_3d_to_2d(camera, pts_3d)
    errors = np.linalg.norm(projected - pts_2d, axis=1)
    if visibility is not None:
        mask = visibility >= 0.3
        if mask.sum() == 0:
            return float("nan")
        return float(errors[mask].mean())
    return float(errors.mean())
