"""Quantitative evaluation metrics for 3D pose estimation."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_mpjpe(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    joint_indices: Optional[List[int]] = None,
) -> float:
    """Mean Per Joint Position Error (meters)."""
    pred = np.atleast_2d(predicted) if predicted.ndim == 2 else predicted
    gt = np.atleast_2d(ground_truth) if ground_truth.ndim == 2 else ground_truth
    if joint_indices is not None:
        pred = pred[..., joint_indices, :]
        gt = gt[..., joint_indices, :]
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def _align_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align source to target via Procrustes (rotation + scale + translation)."""
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    s_c = source - mu_s
    t_c = target - mu_t
    ss = (s_c ** 2).sum()
    if ss < 1e-12:
        return source
    # SVD
    H = s_c.T @ t_c
    U, S, Vt = np.linalg.svd(H)
    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    scale = float(S.sum() / ss)
    aligned = scale * (s_c @ R.T) + mu_t
    return aligned


def compute_pa_mpjpe(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Procrustes-Aligned MPJPE."""
    if predicted.ndim == 3:
        errors = []
        for f in range(predicted.shape[0]):
            aligned = _align_procrustes(predicted[f], ground_truth[f])
            errors.append(np.linalg.norm(aligned - ground_truth[f], axis=-1).mean())
        return float(np.mean(errors))
    aligned = _align_procrustes(predicted, ground_truth)
    return float(np.linalg.norm(aligned - ground_truth, axis=-1).mean())


def compute_pck(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.1,
    joint_indices: Optional[List[int]] = None,
) -> float:
    """Percentage of Correct Keypoints within threshold meters."""
    pred = predicted
    gt = ground_truth
    if joint_indices is not None:
        pred = pred[..., joint_indices, :]
        gt = gt[..., joint_indices, :]
    errors = np.linalg.norm(pred - gt, axis=-1)
    return float((errors < threshold).mean())


def compute_jitter_metric(sequence: np.ndarray) -> float:
    """Mean acceleration magnitude (second discrete derivative along axis=0)."""
    if sequence.shape[0] < 3:
        return 0.0
    acc = sequence[2:] - 2 * sequence[1:-1] + sequence[:-2]
    if acc.ndim == 3:
        return float(np.linalg.norm(acc, axis=-1).mean())
    return float(np.abs(acc).mean())


def compute_bone_length_variance(
    sequence: np.ndarray,          # [F, N, 3]
    connections: Tuple,            # bone connections (a, b) index pairs
) -> float:
    """Mean coefficient of variation of bone lengths across frames."""
    cvs = []
    for a, b in connections:
        if a >= sequence.shape[1] or b >= sequence.shape[1]:
            continue
        lengths = np.linalg.norm(sequence[:, b] - sequence[:, a], axis=1)
        mean = lengths.mean()
        if mean > 1e-6:
            cvs.append(lengths.std() / mean)
    return float(np.mean(cvs)) if cvs else 0.0


def compute_foot_sliding(
    sequence: np.ndarray,     # [F, 33, 3]
    fps: float,
    velocity_threshold: float = 0.05,
) -> float:
    """Mean horizontal foot velocity during ground contact (should be ~0)."""
    from src.pose.keypoints import MP33
    if sequence.shape[0] < 2:
        return 0.0
    contact_vels = []
    for ankle_idx in [MP33.LEFT_ANKLE, MP33.RIGHT_ANKLE]:
        z = sequence[:, ankle_idx, 2]
        median_z = np.median(z)
        # Ground contact heuristic
        contact = (z < median_z + 0.1)
        vert_vel = np.abs(np.diff(z)) * fps
        contact_frames = contact[:-1] & (vert_vel < 0.3)
        if contact_frames.sum() == 0:
            continue
        horiz = sequence[:, ankle_idx, :2]
        horiz_vel = np.linalg.norm(np.diff(horiz, axis=0), axis=1) * fps
        contact_vels.extend(horiz_vel[contact_frames].tolist())
    return float(np.mean(contact_vels)) if contact_vels else 0.0


def compute_reprojection_error_metric(
    sequence_3d: np.ndarray,
    sequence_2d_px: np.ndarray,
    camera,
) -> float:
    from src.reconstruction.camera import reprojection_error
    errors = []
    for f in range(sequence_3d.shape[0]):
        err = reprojection_error(camera, sequence_3d[f], sequence_2d_px[f])
        if np.isfinite(err):
            errors.append(err)
    return float(np.mean(errors)) if errors else 0.0


@dataclass
class PoseMetrics:
    mpjpe: float = 0.0
    pa_mpjpe: float = 0.0
    pck: float = 0.0
    jitter_mean: float = 0.0
    jitter_max: float = 0.0
    bone_length_mean_cv: float = 0.0
    foot_sliding: float = 0.0
    reprojection_error: float = 0.0
    processing_fps: float = 0.0
    peak_ram_mb: float = 0.0
    frame_count: int = 0


class MetricsReporter:
    def compute_all(
        self,
        sequence_3d: np.ndarray,
        sequence_2d_px: Optional[np.ndarray] = None,
        camera=None,
        fps: float = 30.0,
        ground_truth: Optional[np.ndarray] = None,
    ) -> PoseMetrics:
        from src.pose.keypoints import SKELETON_CONNECTIONS
        metrics = PoseMetrics(frame_count=len(sequence_3d))

        if ground_truth is not None:
            metrics.mpjpe = compute_mpjpe(sequence_3d, ground_truth)
            metrics.pa_mpjpe = compute_pa_mpjpe(sequence_3d, ground_truth)
            metrics.pck = compute_pck(sequence_3d, ground_truth)

        metrics.jitter_mean = compute_jitter_metric(sequence_3d)
        metrics.bone_length_mean_cv = compute_bone_length_variance(sequence_3d, SKELETON_CONNECTIONS)
        metrics.foot_sliding = compute_foot_sliding(sequence_3d, fps)

        if sequence_2d_px is not None and camera is not None:
            metrics.reprojection_error = compute_reprojection_error_metric(sequence_3d, sequence_2d_px, camera)

        return metrics

    def save_csv(self, metrics: PoseMetrics, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.__dict__.items():
                writer.writerow([k, v])

    def print_report(self, metrics: PoseMetrics) -> None:
        print("\n=== Pose Metrics ===")
        for k, v in metrics.__dict__.items():
            print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")
        print("=" * 20)
