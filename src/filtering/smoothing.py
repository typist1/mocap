"""Offline (batch) smoothing applied after all frames are collected."""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

logger = logging.getLogger(__name__)


def savitzky_golay_smooth(
    sequence: np.ndarray,
    window: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay along axis=0. Input shape [F, ...]. Returns same shape."""
    F = sequence.shape[0]
    if F < window:
        logger.warning(
            "Sequence length %d < window %d; skipping Savitzky-Golay smoothing", F, window
        )
        return sequence
    # savgol requires window > polyorder and window odd
    w = window if window % 2 == 1 else window + 1
    if w > F:
        w = F if F % 2 == 1 else F - 1
    if w <= polyorder:
        return sequence
    flat = sequence.reshape(F, -1)
    smoothed = savgol_filter(flat, window_length=w, polyorder=polyorder, axis=0)
    return smoothed.reshape(sequence.shape)


def fill_temporal_gaps(
    sequence: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    min_visibility: float = 0.3,
) -> np.ndarray:
    """Linearly interpolate NaN values along time for each landmark/axis.

    Input sequence: [F, N, 3] or [F, ...]. NaN values in input are always
    filled. When visibility provided, entries < min_visibility are also marked
    as NaN before interpolation.
    """
    seq = sequence.copy().astype(np.float64)
    F = seq.shape[0]

    if visibility is not None and seq.ndim == 3:
        N = seq.shape[1]
        for f in range(F):
            for n in range(N):
                if float(visibility[f, n]) < min_visibility:
                    seq[f, n, :] = np.nan

    flat = seq.reshape(F, -1)
    t = np.arange(F)

    for c in range(flat.shape[1]):
        col = flat[:, c]
        good = ~np.isnan(col)
        if good.sum() == 0:
            col[:] = 0.0
        elif good.sum() < F:
            col[~good] = np.interp(t[~good], t[good], col[good])

    return flat.reshape(seq.shape)


def normalize_bone_lengths_sequence(
    sequence: np.ndarray,                  # [F, 33, 3]
    target_lengths: Dict[str, float],
    skeleton,
) -> np.ndarray:
    """Normalize bone lengths for every frame."""
    result = np.empty_like(sequence)
    for f in range(sequence.shape[0]):
        result[f] = skeleton.normalize_bone_lengths(sequence[f], target_lengths)
    return result


def compute_jitter(sequence: np.ndarray) -> float:
    """Mean acceleration magnitude (second discrete derivative along axis=0).

    Input: [F, N, 3] or [F, N]. Returns scalar mean over all joints and frames.
    """
    if sequence.shape[0] < 3:
        return 0.0
    acc = sequence[2:] - 2 * sequence[1:-1] + sequence[:-2]
    if acc.ndim == 3:
        return float(np.linalg.norm(acc, axis=-1).mean())
    return float(np.abs(acc).mean())


def smooth_root_motion(root: np.ndarray, fps: float, cutoff_hz: float = 6.0) -> np.ndarray:
    """Zero-lag Butterworth low-pass on root motion [F, 3]."""
    F = root.shape[0]
    if F < 15 or cutoff_hz <= 0 or cutoff_hz >= fps / 2:
        return root
    b, a = butter(2, cutoff_hz / (fps / 2))
    return filtfilt(b, a, root, axis=0)
