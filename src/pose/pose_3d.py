"""3D pose lifting from 2D landmarks + MediaPipe world landmarks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.pose.keypoints import MP33, BODY_JOINTS
from src.pose.pose_2d import Pose2DResult
from src.reconstruction.camera import CameraModel, make_camera, reprojection_error, project_3d_to_2d
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Pose3DResult:
    positions: np.ndarray      # [33, 3] camera-relative meters, centered on hip midpoint
    root_world: np.ndarray     # [3] world-space root motion (cumulative from frame 0)
    frame_index: int
    reprojection_error_px: float


class Pose3DLifter(ABC):
    @abstractmethod
    def lift(
        self,
        pose_2d: Pose2DResult,
        camera: CameraModel,
        prev_pose3d: Optional[Pose3DResult] = None,
    ) -> Pose3DResult:
        ...


class GeometricLifter(Pose3DLifter):
    """Uses MediaPipe world landmarks directly as 3D estimate.

    Root motion is estimated from hip pixel movement scaled by the
    torso pixel-to-meter ratio. No deep learning model required.
    """

    def __init__(self) -> None:
        self._torso_meters_per_pixel: Optional[float] = None
        self._root_world_accumulated = np.zeros(3)
        self._prev_hip_px: Optional[np.ndarray] = None

    def lift(
        self,
        pose_2d: Pose2DResult,
        camera: CameraModel,
        prev_pose3d: Optional[Pose3DResult] = None,
    ) -> Pose3DResult:
        world = pose_2d.landmarks_world.copy()   # [33, 3]
        image = pose_2d.landmarks_image          # [33, 3]: x,y norm + vis

        # Hip midpoint (root)
        hip_mid = (world[MP33.LEFT_HIP] + world[MP33.RIGHT_HIP]) / 2.0

        # Center body on hip midpoint (MediaPipe world is already hip-centered,
        # but we ensure it explicitly)
        world_centered = world - hip_mid

        # Convert MediaPipe world (x-right, y-down, z-away) → Blender (x-right, y-forward, z-up)
        blender = np.empty_like(world_centered)
        blender[:, 0] = world_centered[:, 0]    # x → x
        blender[:, 1] = world_centered[:, 2]    # z → y (depth → forward)
        blender[:, 2] = -world_centered[:, 1]   # -y → z (down → up)
        world_centered = blender

        # Estimate scale: torso pixels / torso meters
        sh_mid_w = (world_centered[MP33.LEFT_SHOULDER] + world_centered[MP33.RIGHT_SHOULDER]) / 2.0
        torso_m = float(np.linalg.norm(sh_mid_w))

        hip_px = np.array([
            (image[MP33.LEFT_HIP, 0] + image[MP33.RIGHT_HIP, 0]) / 2.0 * camera.width,
            (image[MP33.LEFT_HIP, 1] + image[MP33.RIGHT_HIP, 1]) / 2.0 * camera.height,
        ])
        sh_px = np.array([
            (image[MP33.LEFT_SHOULDER, 0] + image[MP33.RIGHT_SHOULDER, 0]) / 2.0 * camera.width,
            (image[MP33.LEFT_SHOULDER, 1] + image[MP33.RIGHT_SHOULDER, 1]) / 2.0 * camera.height,
        ])
        torso_px = float(np.linalg.norm(sh_px - hip_px))

        if torso_px > 2.0 and torso_m > 1e-4:
            self._torso_meters_per_pixel = torso_m / torso_px

        mpp = self._torso_meters_per_pixel or 0.001

        # Root motion: hip pixel displacement from last frame → world displacement
        if self._prev_hip_px is not None:
            delta_px = hip_px - self._prev_hip_px
            # Positive image Y is downward; map to Blender axes (X right, Y forward, Z up)
            # MediaPipe: x right, y down → Blender: x right, z up (y = depth, unchanged)
            self._root_world_accumulated[0] += delta_px[0] * mpp   # X: right
            # Y pixel movement → Z (vertical) in Blender space
            self._root_world_accumulated[2] += -delta_px[1] * mpp  # Z: up (image y down)
        self._prev_hip_px = hip_px.copy()

        # 2D reprojection error (use image pixel coords)
        vis = image[:, 2]
        pts_2d_px = np.stack([
            image[:, 0] * camera.width,
            image[:, 1] * camera.height,
        ], axis=1)
        # For reprojection we place the pose at Z=1 (normalized)
        world_at_unit_depth = world_centered.copy()
        world_at_unit_depth[:, 2] += 1.0
        reproj_err = reprojection_error(camera, world_at_unit_depth, pts_2d_px, vis)

        return Pose3DResult(
            positions=world_centered,
            root_world=self._root_world_accumulated.copy(),
            frame_index=pose_2d.frame_index,
            reprojection_error_px=reproj_err if np.isfinite(reproj_err) else 0.0,
        )

    def reset(self) -> None:
        self._torso_meters_per_pixel = None
        self._root_world_accumulated = np.zeros(3)
        self._prev_hip_px = None
