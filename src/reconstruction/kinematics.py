"""Hierarchical skeletal kinematics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.pose.keypoints import MP33, MP33_NAMES
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Joint:
    name: str
    index: int  # MP33 index; -1 for virtual joints
    parent: Optional["Joint"] = field(default=None, repr=False)
    children: List["Joint"] = field(default_factory=list, repr=False)
    rest_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))


class Skeleton:
    def __init__(self, root: Joint) -> None:
        self._root = root
        self._joints: Dict[str, Joint] = {}
        self._register(root)

    def _register(self, joint: Joint) -> None:
        self._joints[joint.name] = joint
        for child in joint.children:
            self._register(child)

    @property
    def root(self) -> Joint:
        return self._root

    @property
    def joints(self) -> Dict[str, Joint]:
        return self._joints

    def get_joint(self, name: str) -> Joint:
        if name not in self._joints:
            raise KeyError(f"Joint '{name}' not in skeleton. Available: {sorted(self._joints)}")
        return self._joints[name]

    def compute_bone_lengths(self, positions_3d: np.ndarray) -> Dict[str, float]:
        """Input [33,3] or [N,3]. Returns {bone_name: length_meters}."""
        lengths: Dict[str, float] = {}
        for name, joint in self._joints.items():
            if joint.parent is None or joint.index < 0 or joint.parent.index < 0:
                continue
            p_idx = joint.parent.index
            c_idx = joint.index
            if p_idx >= len(positions_3d) or c_idx >= len(positions_3d):
                continue
            bone_name = f"{joint.parent.name}_{name}"
            length = float(np.linalg.norm(positions_3d[c_idx] - positions_3d[p_idx]))
            lengths[bone_name] = length
        return lengths

    def normalize_bone_lengths(
        self, positions_3d: np.ndarray, target_lengths: Dict[str, float]
    ) -> np.ndarray:
        """Scale bones to target lengths, outward from root. Returns [33,3]."""
        pts = positions_3d.copy()
        self._normalize_recursive(self._root, pts, target_lengths)
        return pts

    def _normalize_recursive(self, joint: Joint, pts: np.ndarray, targets: Dict[str, float]) -> None:
        for child in joint.children:
            if joint.index < 0 or child.index < 0:
                self._normalize_recursive(child, pts, targets)
                continue
            bone_name = f"{joint.name}_{child.name}"
            if bone_name not in targets:
                self._normalize_recursive(child, pts, targets)
                continue
            target = targets[bone_name]
            direction = pts[child.index] - pts[joint.index]
            current_len = float(np.linalg.norm(direction))
            if current_len < 1e-6:
                self._normalize_recursive(child, pts, targets)
                continue
            scale = target / current_len
            offset = direction * (scale - 1.0)
            # Translate child and all descendants
            self._translate_subtree(child, pts, offset)
            self._normalize_recursive(child, pts, targets)

    def _translate_subtree(self, joint: Joint, pts: np.ndarray, offset: np.ndarray) -> None:
        if joint.index >= 0:
            pts[joint.index] += offset
        for child in joint.children:
            self._translate_subtree(child, pts, offset)


def build_mediapipe_skeleton() -> Skeleton:
    """Build standard skeleton hierarchy for MediaPipe 33 landmarks."""
    def j(name: str, idx: int, rest_dir: Optional[np.ndarray] = None) -> Joint:
        d = rest_dir if rest_dir is not None else np.array([0.0, 1.0, 0.0])
        return Joint(name=name, index=idx, rest_direction=d)

    # Spine chain (approximate using shoulder midpoint as top of spine)
    root = j("hip_mid", -1, np.array([0.0, 1.0, 0.0]))   # virtual root

    l_hip = j("left_hip", MP33.LEFT_HIP, np.array([-1.0, 0.0, 0.0]))
    l_knee = j("left_knee", MP33.LEFT_KNEE, np.array([0.0, -1.0, 0.0]))
    l_ankle = j("left_ankle", MP33.LEFT_ANKLE, np.array([0.0, -1.0, 0.0]))
    l_foot = j("left_foot_index", MP33.LEFT_FOOT_INDEX, np.array([0.0, 0.0, 1.0]))
    l_knee.children = [l_ankle]
    l_ankle.children = [l_foot]
    l_hip.children = [l_knee]
    l_knee.parent = l_hip
    l_ankle.parent = l_knee
    l_foot.parent = l_ankle

    r_hip = j("right_hip", MP33.RIGHT_HIP, np.array([1.0, 0.0, 0.0]))
    r_knee = j("right_knee", MP33.RIGHT_KNEE, np.array([0.0, -1.0, 0.0]))
    r_ankle = j("right_ankle", MP33.RIGHT_ANKLE, np.array([0.0, -1.0, 0.0]))
    r_foot = j("right_foot_index", MP33.RIGHT_FOOT_INDEX, np.array([0.0, 0.0, 1.0]))
    r_knee.children = [r_ankle]
    r_ankle.children = [r_foot]
    r_hip.children = [r_knee]
    r_knee.parent = r_hip
    r_ankle.parent = r_knee
    r_foot.parent = r_ankle

    spine = j("spine", -1, np.array([0.0, 1.0, 0.0]))  # virtual spine root
    shoulder_mid = j("shoulder_mid", -1, np.array([0.0, 1.0, 0.0]))

    l_shoulder = j("left_shoulder", MP33.LEFT_SHOULDER, np.array([-1.0, 0.0, 0.0]))
    l_elbow = j("left_elbow", MP33.LEFT_ELBOW, np.array([-1.0, 0.0, 0.0]))
    l_wrist = j("left_wrist", MP33.LEFT_WRIST, np.array([-1.0, 0.0, 0.0]))
    l_wrist.parent = l_elbow
    l_elbow.children = [l_wrist]
    l_elbow.parent = l_shoulder
    l_shoulder.children = [l_elbow]

    r_shoulder = j("right_shoulder", MP33.RIGHT_SHOULDER, np.array([1.0, 0.0, 0.0]))
    r_elbow = j("right_elbow", MP33.RIGHT_ELBOW, np.array([1.0, 0.0, 0.0]))
    r_wrist = j("right_wrist", MP33.RIGHT_WRIST, np.array([1.0, 0.0, 0.0]))
    r_wrist.parent = r_elbow
    r_elbow.children = [r_wrist]
    r_elbow.parent = r_shoulder
    r_shoulder.children = [r_elbow]

    neck = j("neck", -1, np.array([0.0, 1.0, 0.0]))  # virtual
    nose = j("nose", MP33.NOSE, np.array([0.0, 1.0, 0.0]))
    neck.children = [nose]
    nose.parent = neck

    shoulder_mid.children = [l_shoulder, r_shoulder, neck]
    l_shoulder.parent = shoulder_mid
    r_shoulder.parent = shoulder_mid
    neck.parent = shoulder_mid

    spine.children = [shoulder_mid]
    shoulder_mid.parent = spine

    root.children = [l_hip, r_hip, spine]
    l_hip.parent = root
    r_hip.parent = root
    spine.parent = root

    return Skeleton(root)


def forward_kinematics(
    root_pos: np.ndarray,
    joint_rotations: Dict[str, np.ndarray],  # name → 3x3 rotation matrix
    skeleton: Skeleton,
) -> np.ndarray:
    """Compute world positions via FK. Returns array indexed by joint.index."""
    n = max(j.index for j in skeleton.joints.values() if j.index >= 0) + 1
    positions = np.zeros((n, 3))
    rotations: Dict[str, np.ndarray] = {}

    def traverse(joint: Joint, parent_pos: np.ndarray, parent_rot: np.ndarray) -> None:
        R = joint_rotations.get(joint.name, np.eye(3))
        global_rot = parent_rot @ R
        rotations[joint.name] = global_rot
        if joint.index >= 0:
            world_dir = parent_rot @ joint.rest_direction
            # We'll position based on parent position + rest direction * bone_length
            # For FK from rotations, use the rotation to transform rest direction
            positions[joint.index] = parent_pos  # simplified; real FK needs bone lengths
        for child in joint.children:
            pos = parent_pos if joint.index < 0 else positions[joint.index]
            traverse(child, pos, global_rot)

    traverse(skeleton.root, root_pos, np.eye(3))
    return positions
