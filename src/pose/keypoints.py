"""Keypoint definitions and mappings.

Single source of truth for all landmark indices and names used throughout
the pipeline. Any new backend must map its landmarks to MediaPipe33 or
define its own KeypointSet with an explicit mapping to MediaPipe33.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class MP33(IntEnum):
    """MediaPipe 33-landmark indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Human-readable names aligned to MP33 indices
MP33_NAMES: List[str] = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

assert len(MP33_NAMES) == 33

# Indices used to score overall body detection quality
BODY_JOINTS: Tuple[int, ...] = (
    MP33.LEFT_SHOULDER, MP33.RIGHT_SHOULDER,
    MP33.LEFT_ELBOW, MP33.RIGHT_ELBOW,
    MP33.LEFT_WRIST, MP33.RIGHT_WRIST,
    MP33.LEFT_HIP, MP33.RIGHT_HIP,
    MP33.LEFT_KNEE, MP33.RIGHT_KNEE,
    MP33.LEFT_ANKLE, MP33.RIGHT_ANKLE,
)

# Bone connections used for visualization and bone-length computation
SKELETON_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (MP33.LEFT_SHOULDER, MP33.RIGHT_SHOULDER),
    (MP33.LEFT_SHOULDER, MP33.LEFT_ELBOW),
    (MP33.LEFT_ELBOW, MP33.LEFT_WRIST),
    (MP33.RIGHT_SHOULDER, MP33.RIGHT_ELBOW),
    (MP33.RIGHT_ELBOW, MP33.RIGHT_WRIST),
    (MP33.LEFT_SHOULDER, MP33.LEFT_HIP),
    (MP33.RIGHT_SHOULDER, MP33.RIGHT_HIP),
    (MP33.LEFT_HIP, MP33.RIGHT_HIP),
    (MP33.LEFT_HIP, MP33.LEFT_KNEE),
    (MP33.LEFT_KNEE, MP33.LEFT_ANKLE),
    (MP33.RIGHT_HIP, MP33.RIGHT_KNEE),
    (MP33.RIGHT_KNEE, MP33.RIGHT_ANKLE),
    (MP33.LEFT_ANKLE, MP33.LEFT_FOOT_INDEX),
    (MP33.RIGHT_ANKLE, MP33.RIGHT_FOOT_INDEX),
    (MP33.LEFT_SHOULDER, MP33.NOSE),
    (MP33.RIGHT_SHOULDER, MP33.NOSE),
)


@dataclass(frozen=True)
class BonePair:
    """A named bone between two landmark indices."""
    name: str
    parent: int   # proximal landmark index
    child: int    # distal landmark index


# Canonical bone list used for bone-length normalization and kinematics
CANONICAL_BONES: Tuple[BonePair, ...] = (
    BonePair("spine", (MP33.LEFT_HIP + MP33.RIGHT_HIP) // 2, MP33.LEFT_SHOULDER),  # approximation
    BonePair("left_upper_arm", MP33.LEFT_SHOULDER, MP33.LEFT_ELBOW),
    BonePair("left_forearm", MP33.LEFT_ELBOW, MP33.LEFT_WRIST),
    BonePair("right_upper_arm", MP33.RIGHT_SHOULDER, MP33.RIGHT_ELBOW),
    BonePair("right_forearm", MP33.RIGHT_ELBOW, MP33.RIGHT_WRIST),
    BonePair("left_thigh", MP33.LEFT_HIP, MP33.LEFT_KNEE),
    BonePair("left_shin", MP33.LEFT_KNEE, MP33.LEFT_ANKLE),
    BonePair("right_thigh", MP33.RIGHT_HIP, MP33.RIGHT_KNEE),
    BonePair("right_shin", MP33.RIGHT_KNEE, MP33.RIGHT_ANKLE),
    BonePair("left_foot", MP33.LEFT_ANKLE, MP33.LEFT_FOOT_INDEX),
    BonePair("right_foot", MP33.RIGHT_ANKLE, MP33.RIGHT_FOOT_INDEX),
    BonePair("clavicle", MP33.LEFT_SHOULDER, MP33.RIGHT_SHOULDER),
    BonePair("pelvis", MP33.LEFT_HIP, MP33.RIGHT_HIP),
    BonePair("neck", MP33.LEFT_SHOULDER, MP33.NOSE),   # approximate
)


def name_to_index(name: str) -> int:
    """Return MP33 index for a landmark name. Raises KeyError if unknown."""
    idx = MP33_NAMES.index(name) if name in MP33_NAMES else -1
    if idx < 0:
        raise KeyError(f"Unknown landmark name: {name!r}")
    return idx


def index_to_name(idx: int) -> str:
    if not (0 <= idx < len(MP33_NAMES)):
        raise IndexError(f"Index {idx} out of range [0, 32]")
    return MP33_NAMES[idx]


# Symmetric pairs used for left/right consistency checks
SYMMETRIC_PAIRS: Tuple[Tuple[int, int], ...] = (
    (MP33.LEFT_SHOULDER, MP33.RIGHT_SHOULDER),
    (MP33.LEFT_ELBOW, MP33.RIGHT_ELBOW),
    (MP33.LEFT_WRIST, MP33.RIGHT_WRIST),
    (MP33.LEFT_HIP, MP33.RIGHT_HIP),
    (MP33.LEFT_KNEE, MP33.RIGHT_KNEE),
    (MP33.LEFT_ANKLE, MP33.RIGHT_ANKLE),
)
